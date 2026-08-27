//! Per-model video frame sampling strategies.
//!
//! Which strategy applies comes from the model's `ModelProcessorSpec`
//! (`crate::registry`); the parameters ride on `VideoFetchConfig`
//! (`crate::media`). This module deliberately depends on neither so the
//! decode pipeline can plan frame indices without a module cycle.

/// Static facts about the source video that sampling strategies plan against.
#[derive(Debug, Clone, Copy)]
pub struct VideoSourceMeta {
    /// Total decodable frames reported by the demuxer.
    pub total_frames: usize,
    /// Source frame rate; `<= 0` or NaN marks it as unknown.
    pub original_fps: f64,
    /// Container/stream duration when known.
    pub duration_seconds: Option<f64>,
}

/// How frames are picked out of a source video.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum VideoSamplingStrategy {
    /// Legacy uniform sampling: `round(duration * sample_fps)` frames on a
    /// floor-linspace grid. The default; behavior is identical to the
    /// pre-strategy decode pipeline and applies to every model without a
    /// spec override.
    #[default]
    Uniform,
    /// HF `Qwen3VLVideoProcessor.sample_frames` parity: fps-scaled frame
    /// count with `int()` truncation, clamped to
    /// `[min_frames, min(max_frames, total_frames)]`, on a round-linspace
    /// grid (numpy half-to-even ties).
    Qwen3Vl,
}

/// The frames to decode, as indices into the source video.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameSamplingPlan {
    pub indices: Vec<usize>,
}

impl VideoSamplingStrategy {
    /// Plan which source frames to decode. Takes the sampling parameters as
    /// primitives (rather than `VideoFetchConfig`) to keep this module free
    /// of `crate::media`.
    pub fn plan(
        self,
        source: &VideoSourceMeta,
        min_frames: usize,
        max_frames: usize,
        sample_fps: f32,
    ) -> FrameSamplingPlan {
        let indices = match self {
            Self::Uniform => uniform_frame_indices(source, min_frames, max_frames, sample_fps),
            Self::Qwen3Vl => qwen3_vl_frame_indices(source, min_frames, max_frames, sample_fps),
        };
        FrameSamplingPlan { indices }
    }
}

/// The legacy `opencv_frame_indices` logic, unchanged: round the fps-scaled
/// frame count, clamp to `[min_frames, max_frames]`, floor-linspace over
/// `[0, total_frames - 1]`.
fn uniform_frame_indices(
    source: &VideoSourceMeta,
    min_frames: usize,
    max_frames: usize,
    sample_fps: f32,
) -> Vec<usize> {
    let total_frames = source.total_frames;
    if total_frames == 0 {
        return Vec::new();
    }
    let mut target_frames = if source.original_fps.is_finite() && source.original_fps > 0.0 {
        let duration = total_frames as f64 / source.original_fps;
        (duration * f64::from(sample_fps)).round() as usize
    } else {
        max_frames
    };
    target_frames = target_frames.clamp(min_frames, max_frames);
    target_frames = target_frames.max(1);
    if target_frames == 1 {
        return vec![0];
    }

    let last = (total_frames - 1) as f64;
    let denom = (target_frames - 1) as f64;
    (0..target_frames)
        .map(|idx| ((idx as f64 * last) / denom).floor() as usize)
        .collect()
}

/// Mirrors HF `Qwen3VLVideoProcessor.sample_frames` (and vLLM's
/// `Qwen3VLVideoBackend.compute_frames_index_to_sample`): the fps-scaled
/// count is truncated with `int()`, clamped to
/// `min(max(count, min_frames), max_frames, total_frames)`, then a linspace
/// over `[0, total_frames - 1]` is rounded with numpy's half-to-even ties.
///
/// Falls back to [`uniform_frame_indices`] when the source frame rate is
/// unknown, where the HF formula is undefined.
fn qwen3_vl_frame_indices(
    source: &VideoSourceMeta,
    min_frames: usize,
    max_frames: usize,
    sample_fps: f32,
) -> Vec<usize> {
    let total_frames = source.total_frames;
    if total_frames == 0 {
        return Vec::new();
    }
    if !(source.original_fps.is_finite() && source.original_fps > 0.0) {
        return uniform_frame_indices(source, min_frames, max_frames, sample_fps);
    }

    let count = (total_frames as f64 / source.original_fps * f64::from(sample_fps)) as usize;
    let upper = max_frames.min(total_frames).max(1);
    let count = count.max(min_frames).min(upper).max(1);
    if count == 1 {
        return vec![0];
    }

    let last = (total_frames - 1) as f64;
    let denom = (count - 1) as f64;
    (0..count)
        .map(|idx| round_half_to_even(idx as f64 * last / denom) as usize)
        .collect()
}

/// `numpy.round` parity: ties go to the even neighbor, unlike `f64::round`
/// which rounds half away from zero.
fn round_half_to_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else if floor % 2.0 == 0.0 {
        floor
    } else {
        floor + 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(total_frames: usize, original_fps: f64) -> VideoSourceMeta {
        VideoSourceMeta {
            total_frames,
            original_fps,
            duration_seconds: (original_fps.is_finite() && original_fps > 0.0)
                .then_some(total_frames as f64 / original_fps),
        }
    }

    // Uniform: parity with the legacy `opencv_frame_indices` in media.rs.

    #[test]
    fn uniform_preserves_min_frames_for_short_clips() {
        let plan = VideoSamplingStrategy::Uniform.plan(&source(1, 30.0), 4, 8, 2.0);
        assert_eq!(plan.indices, vec![0, 0, 0, 0]);
    }

    #[test]
    fn uniform_uses_rounded_fps_scaled_count_and_floor_linspace() {
        // duration = 100/25 = 4s -> round(4 * 2) = 8 frames over [0, 99].
        let plan = VideoSamplingStrategy::Uniform.plan(&source(100, 25.0), 4, 768, 2.0);
        assert_eq!(plan.indices, vec![0, 14, 28, 42, 56, 70, 84, 99]);
    }

    #[test]
    fn uniform_falls_back_to_max_frames_when_fps_unknown() {
        for bad_fps in [0.0, -1.0, f64::NAN] {
            let plan = VideoSamplingStrategy::Uniform.plan(&source(100, bad_fps), 4, 8, 2.0);
            assert_eq!(plan.indices, vec![0, 14, 28, 42, 56, 70, 84, 99]);
        }
    }

    #[test]
    fn uniform_single_target_is_frame_zero() {
        let plan = VideoSamplingStrategy::Uniform.plan(&source(100, 25.0), 1, 1, 2.0);
        assert_eq!(plan.indices, vec![0]);
    }

    #[test]
    fn uniform_empty_source_yields_no_indices() {
        let plan = VideoSamplingStrategy::Uniform.plan(&source(0, 30.0), 4, 8, 2.0);
        assert!(plan.indices.is_empty());
    }

    // Qwen3Vl: HF Qwen3VLVideoProcessor.sample_frames parity.

    #[test]
    fn qwen3_vl_truncates_fps_scaled_count() {
        // int(98/30 * 2) = int(6.533) = 6; the uniform round() would give 7.
        // linspace(0, 97, 6).round() = [0, 19, 39, 58, 78, 97].
        let plan = VideoSamplingStrategy::Qwen3Vl.plan(&source(98, 30.0), 4, 768, 2.0);
        assert_eq!(plan.indices, vec![0, 19, 39, 58, 78, 97]);
    }

    #[test]
    fn qwen3_vl_clamps_up_to_min_frames() {
        // int(10/30 * 2) = 0 -> clamped up to min_frames = 4.
        let plan = VideoSamplingStrategy::Qwen3Vl.plan(&source(10, 30.0), 4, 768, 2.0);
        assert_eq!(plan.indices, vec![0, 3, 6, 9]);
    }

    #[test]
    fn qwen3_vl_clamps_down_to_max_frames() {
        // int(100000/30 * 2) = 6666 -> clamped down to max_frames = 768.
        let plan = VideoSamplingStrategy::Qwen3Vl.plan(&source(100_000, 30.0), 4, 768, 2.0);
        assert_eq!(plan.indices.len(), 768);
        assert_eq!(plan.indices[0], 0);
        assert_eq!(*plan.indices.last().unwrap(), 99_999);
    }

    #[test]
    fn qwen3_vl_never_exceeds_total_frames_even_when_min_frames_is_higher() {
        // HF: min(max(count, min_frames), max_frames, total) — total wins.
        let plan = VideoSamplingStrategy::Qwen3Vl.plan(&source(3, 30.0), 4, 768, 2.0);
        assert_eq!(plan.indices, vec![0, 1, 2]);
    }

    #[test]
    fn qwen3_vl_rounds_half_to_even_like_numpy() {
        // int(6/4 * 2) = 3; linspace(0, 5, 3) = [0, 2.5, 5] and numpy rounds
        // 2.5 to 2 (half-to-even) where f64::round would give 3.
        let plan = VideoSamplingStrategy::Qwen3Vl.plan(&source(6, 4.0), 1, 768, 2.0);
        assert_eq!(plan.indices, vec![0, 2, 5]);
    }

    #[test]
    fn qwen3_vl_matches_hf_count_for_fractional_rates() {
        // int(493/30 * 2) = int(32.867) = 32 frames spanning the whole clip.
        let plan = VideoSamplingStrategy::Qwen3Vl.plan(&source(493, 30.0), 4, 768, 2.0);
        assert_eq!(plan.indices.len(), 32);
        assert_eq!(plan.indices[0], 0);
        assert_eq!(*plan.indices.last().unwrap(), 492);
    }

    #[test]
    fn qwen3_vl_falls_back_to_uniform_when_fps_unknown() {
        for bad_fps in [0.0, -1.0, f64::NAN] {
            let qwen = VideoSamplingStrategy::Qwen3Vl.plan(&source(100, bad_fps), 4, 8, 2.0);
            let uniform = VideoSamplingStrategy::Uniform.plan(&source(100, bad_fps), 4, 8, 2.0);
            assert_eq!(qwen.indices, uniform.indices);
        }
    }

    #[test]
    fn qwen3_vl_empty_source_yields_no_indices() {
        let plan = VideoSamplingStrategy::Qwen3Vl.plan(&source(0, 30.0), 4, 8, 2.0);
        assert!(plan.indices.is_empty());
    }

    #[test]
    fn default_strategy_is_uniform() {
        assert_eq!(
            VideoSamplingStrategy::default(),
            VideoSamplingStrategy::Uniform
        );
    }
}
