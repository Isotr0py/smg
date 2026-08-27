//! Per-model video frame sampling strategies.
//!
//! Which strategy applies comes from the model's `ModelProcessorSpec`
//! (`crate::registry`); the parameters ride on `VideoFetchConfig`
//! (`crate::media`). This module owns the generic `Uniform` sampling and the
//! strategy dispatch; model-specific index math lives next to its spec in
//! `crate::registry` (e.g. `registry::qwen3_vl_frame_indices`). The strategy
//! parameters are passed as primitives (rather than `VideoFetchConfig`) to
//! keep this module free of `crate::media`.

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
            Self::Qwen3Vl => {
                crate::registry::qwen3_vl_frame_indices(source, min_frames, max_frames, sample_fps)
            }
        };
        FrameSamplingPlan { indices }
    }
}

/// The legacy `opencv_frame_indices` logic, unchanged: round the fps-scaled
/// frame count, clamp to `[min_frames, max_frames]`, floor-linspace over
/// `[0, total_frames - 1]`. Also the fallback for model-specific strategies
/// whose inputs are undefined (e.g. unknown source fps).
pub(crate) fn uniform_frame_indices(
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

    #[test]
    fn default_strategy_is_uniform() {
        assert_eq!(
            VideoSamplingStrategy::default(),
            VideoSamplingStrategy::Uniform
        );
    }
}
