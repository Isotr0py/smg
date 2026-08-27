use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    media::VideoFetchConfig,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PromptReplacement, TokenId},
    video_sampling::{uniform_frame_indices, VideoSamplingStrategy, VideoSourceMeta},
    vision::PreProcessorConfig,
};

pub(super) struct Qwen3VLVisionSpec;

/// Mirrors HF `Qwen3VLVideoProcessor.sample_frames` (and vLLM's
/// `Qwen3VLVideoBackend.compute_frames_index_to_sample`): the fps-scaled
/// count is truncated with `int()`, clamped to
/// `min(max(count, min_frames), max_frames, total_frames)`, then a linspace
/// over `[0, total_frames - 1]` is rounded with numpy's half-to-even ties.
///
/// Falls back to [`uniform_frame_indices`] when the source frame rate is
/// unknown, where the HF formula is undefined. Dispatched from
/// [`VideoSamplingStrategy::Qwen3Vl`].
pub(crate) fn qwen3_vl_frame_indices(
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

/// Shared `video_fetch_config` implementation for the video-capable Qwen3
/// specs (Qwen3-VL, Qwen3-Omni): HF `Qwen3VLVideoProcessor` sampling
/// semantics with parameters from `video_preprocessor_config.json` (`extra`)
/// when the checkpoint provides them, HF defaults otherwise.
pub(super) fn qwen3_video_fetch_config(
    video_preprocessor_config: Option<&PreProcessorConfig>,
) -> VideoFetchConfig {
    let defaults = VideoFetchConfig::default();
    VideoFetchConfig {
        min_frames: video_preprocessor_config
            .and_then(|config| config.get_extra::<usize>("min_frames"))
            .unwrap_or(defaults.min_frames),
        max_frames: video_preprocessor_config
            .and_then(|config| config.get_extra::<usize>("max_frames"))
            .unwrap_or(defaults.max_frames),
        sample_fps: video_preprocessor_config
            .and_then(|config| config.get_extra::<f32>("fps"))
            .unwrap_or(defaults.sample_fps),
        strategy: VideoSamplingStrategy::Qwen3Vl,
    }
}

impl Qwen3VLVisionSpec {
    fn image_pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }

    fn video_pad_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["video_token_id"])
            .map(|v| v as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "video_token_id".to_string(),
            })
    }

    fn vision_start_token_id(metadata: &ModelMetadata) -> Option<TokenId> {
        metadata
            .config_u32(&["vision_start_token_id"])
            .map(|v| v as TokenId)
    }

    fn vision_end_token_id(metadata: &ModelMetadata) -> Option<TokenId> {
        metadata
            .config_u32(&["vision_end_token_id"])
            .map(|v| v as TokenId)
    }

    fn token_for_id(
        metadata: &ModelMetadata,
        token_id: TokenId,
        field: &str,
    ) -> RegistryResult<String> {
        metadata
            .tokenizer
            .id_to_token(token_id as u32)
            .ok_or_else(|| ModelRegistryError::TokenNotFound {
                token: format!("{field}:{token_id}"),
            })
    }

    fn video_grid_t(preprocessed: &PreprocessedEncoderInputs) -> Option<usize> {
        match preprocessed.model_specific.get("video_grid_thw") {
            Some(ModelSpecificValue::IntTensor { data, shape })
                if shape == &[1, 3] && !data.is_empty() =>
            {
                usize::try_from(data[0]).ok()
            }
            _ => None,
        }
    }

    fn encode_plain_text(metadata: &ModelMetadata, text: &str) -> Vec<TokenId> {
        metadata
            .tokenizer
            .encode_text(text)
            .map(|ids| ids.into_iter().map(|id| id as TokenId).collect())
            .unwrap_or_default()
    }

    /// Build the per-frame video placeholder body for the Qwen3-VL family.
    ///
    /// Qwen3-VL lays out video as one `<|vision_start|> .. <|vision_end|>` block
    /// per temporal frame with a `<seconds>` timestamp between frames. The chat
    /// template already supplies the outer `<|vision_start|>`/`<|vision_end|>`, so
    /// this emits only the inner per-frame structure (hence the `grid_idx > 0`
    /// guards that reuse the template's opener/closer for the first/last frame).
    /// Returns `None` when the layout can't apply (single-frame or ragged token
    /// counts), leaving the caller to fall back to a flat pad block.
    fn per_frame_video_tokens(
        metadata: &ModelMetadata,
        pad_token_id: TokenId,
        num_tokens: usize,
        grid_t: usize,
        sample_fps: f64,
    ) -> Option<Vec<TokenId>> {
        if grid_t <= 1 || num_tokens == 0 || !num_tokens.is_multiple_of(grid_t) {
            return None;
        }
        let vision_start = Self::vision_start_token_id(metadata)?;
        let vision_end = Self::vision_end_token_id(metadata)?;
        let tokens_per_grid = num_tokens / grid_t;
        let mut tokens = Vec::with_capacity(num_tokens + (grid_t.saturating_sub(1)) * 8);
        let temporal_patch_size = metadata
            .config_u32(&["vision_config", "temporal_patch_size"])
            .unwrap_or(2) as f64;
        // HF timestamp convention: each temporal patch is timestamped by the
        // average frame time, formatted with one decimal place. `sample_fps`
        // comes from the preprocessor's `video_second_per_grid` (see
        // `video_sample_fps`) so decode fps, processor, and timestamps agree.

        for grid_idx in 0..grid_t {
            let seconds = (grid_idx as f64 * temporal_patch_size
                + (temporal_patch_size - 1.0) / 2.0)
                / sample_fps;
            if grid_idx > 0 {
                tokens.push(vision_end);
            }
            tokens.extend(Self::encode_plain_text(
                metadata,
                &format!("<{seconds:.1} seconds>"),
            ));
            if grid_idx > 0 {
                tokens.push(vision_start);
            }
            tokens.extend(std::iter::repeat_n(pad_token_id, tokens_per_grid));
        }

        Some(tokens)
    }

    /// Effective video sampling fps for prompt timestamps, derived from the
    /// preprocessor's `video_second_per_grid` (`fps = temporal_patch_size /
    /// second_per_grid`, the inverse of how `qwen_vl_base` writes it) so the
    /// decode-time effective fps, the processor, and the prompt timestamps
    /// stay consistent. Falls back to the HF default 2.0 when the value is
    /// missing or degenerate.
    fn video_sample_fps(metadata: &ModelMetadata, preprocessed: &PreprocessedEncoderInputs) -> f64 {
        const DEFAULT_SAMPLE_FPS: f64 = 2.0;
        let second_per_grid = match preprocessed.model_specific.get("video_second_per_grid") {
            Some(ModelSpecificValue::Tensor { data, .. }) if !data.is_empty() => f64::from(data[0]),
            _ => return DEFAULT_SAMPLE_FPS,
        };
        if !second_per_grid.is_finite() || second_per_grid <= 0.0 {
            return DEFAULT_SAMPLE_FPS;
        }
        let temporal_patch_size = f64::from(
            metadata
                .config_u32(&["vision_config", "temporal_patch_size"])
                .unwrap_or(2),
        );
        let fps = temporal_patch_size / second_per_grid;
        if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            DEFAULT_SAMPLE_FPS
        }
    }
}

impl ModelProcessorSpec for Qwen3VLVisionSpec {
    fn name(&self) -> &'static str {
        "qwen3_vl"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        let model_type = metadata.config_model_type();
        let is_qwen3_vl = id.contains("qwen3") && id.contains("vl")
            || model_type.is_some_and(|mt| mt == "qwen3_vl");
        let is_qwen3_5 = id.contains("qwen3.5")
            || id.contains("qwen3.6")
            || model_type.is_some_and(|mt| mt == "qwen3_5" || mt == "qwen3_5_moe");
        let is_qwen4_exp = model_type.is_some_and(|mt| mt == "qwen4_exp");
        is_qwen3_vl || is_qwen3_5 || is_qwen4_exp
    }

    fn placeholder_token(&self, metadata: &ModelMetadata) -> RegistryResult<String> {
        Self::token_for_id(
            metadata,
            Self::image_pad_token_id(metadata)?,
            "image_token_id",
        )
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_pad_token_id(metadata)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            Modality::Video => Self::token_for_id(
                metadata,
                Self::video_pad_token_id(metadata)?,
                "video_token_id",
            ),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image => Self::image_pad_token_id(metadata),
            Modality::Video => Self::video_pad_token_id(metadata),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        let mut limits = HashMap::from([(Modality::Image, 10)]);
        if metadata.config_u32(&["video_token_id"]).is_some() {
            limits.insert(Modality::Video, 1);
        }
        Ok(limits)
    }

    fn processor_kwargs(&self, _metadata: &ModelMetadata) -> RegistryResult<Value> {
        Ok(json!({}))
    }

    fn video_fetch_config(
        &self,
        video_preprocessor_config: Option<&PreProcessorConfig>,
    ) -> VideoFetchConfig {
        qwen3_video_fetch_config(video_preprocessor_config)
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::image_pad_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        // The chat template already wraps each image with <|vision_start|> ... <|vision_end|>,
        // so we only expand the single <|image_pad|> placeholder to N pad tokens.
        Ok(preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                let tokens = vec![pad_token_id; num_tokens];
                PromptReplacement::sequence(Modality::Image, &placeholder_token, tokens)
            })
            .collect())
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        match modality {
            Modality::Image => self.prompt_replacements(metadata, preprocessed),
            Modality::Video => {
                let pad_token_id = Self::video_pad_token_id(metadata)?;
                let placeholder_token = self.placeholder_token_for(metadata, Modality::Video)?;
                let video_grid_t = Self::video_grid_t(preprocessed);
                let video_sample_fps = Self::video_sample_fps(metadata, preprocessed);
                Ok(preprocessed
                    .feature_token_counts
                    .iter()
                    .map(|&num_tokens| {
                        // Every Qwen3-VL model routed to this spec (base VL and the
                        // 3.5/3.6 family) needs the per-frame video layout: vLLM's
                        // mrope pass scans for one <|vision_start|> per temporal
                        // frame, so a single flat block crashes any multi-frame
                        // video. Fall back to a flat block only when the per-frame
                        // layout can't be built (single-frame or unknown grid_t).
                        let tokens = video_grid_t
                            .and_then(|grid_t| {
                                Self::per_frame_video_tokens(
                                    metadata,
                                    pad_token_id,
                                    num_tokens,
                                    grid_t,
                                    video_sample_fps,
                                )
                            })
                            .unwrap_or_else(|| vec![pad_token_id; num_tokens]);
                        // The chat template wraps the placeholder as
                        // <|vision_start|><|video_pad|><|vision_end|>; the leading
                        // <|vision_start|> belongs to the placeholder range so vLLM's
                        // per-frame video mrope finds one marker per frame starting at
                        // the range offset (it scans even for a single frame).
                        PromptReplacement::sequence(Modality::Video, &placeholder_token, tokens)
                            .with_structural_prefix(1)
                    })
                    .collect())
            }
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // encoder_input is patchified: [total_patches, patch_features].
        // patches_per_image tells how many patches belong to each image.
        // image_grid_thw is [num_images, 3].
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_video".to_string(), FieldLayout::Batched),
            ("video_second_per_grid".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::qwen3_video_fetch_config;
    use crate::{
        encoder_inputs::ModelSpecificValue,
        registry::{test_helpers::*, ModelMetadata, ModelRegistry},
        types::ImageSize,
        video_sampling::{VideoSamplingStrategy, VideoSourceMeta},
        vision::PreProcessorConfig,
    };

    fn source(total_frames: usize, original_fps: f64) -> VideoSourceMeta {
        VideoSourceMeta {
            total_frames,
            original_fps,
            duration_seconds: (original_fps.is_finite() && original_fps > 0.0)
                .then_some(total_frames as f64 / original_fps),
        }
    }

    // Qwen3Vl frame sampling: HF Qwen3VLVideoProcessor.sample_frames parity.
    // The same math is pinned against real transformers output by
    // tests/video_sampling_golden.rs.

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
    fn qwen3_vl_pad_only_replacement() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999), ("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653,
            "vision_config": {"patch_size": 16}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen3-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3 spec");
        assert_eq!(spec.name(), "qwen3_vl");
        // 448/16 = 28 grid, merge_size=2 => (28*28)/4 = 196 tokens
        let replacements = spec
            .prompt_replacements(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[196]),
            )
            .unwrap();
        // Only pad tokens — vision_start/vision_end are already in the chat template
        assert_eq!(replacements[0].tokens.len(), 196);
        assert_eq!(replacements[0].tokens[0], 151655); // pad (image_token_id)
        assert_eq!(*replacements[0].tokens.last().unwrap(), 151655); // pad
    }

    #[test]
    fn qwen3_vl_video_pad_replacement() {
        let tokenizer = TestTokenizer::new(&[("<|video_pad|>", 151656)]);
        let config = json!({
            "model_type": "qwen3_5",
            "image_token_id": 151655,
            "video_token_id": 151656,
        });
        let metadata = ModelMetadata {
            model_id: "Qwen3.5-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3.5 spec");
        let replacements = spec
            .prompt_replacements_for(
                &metadata,
                &test_preprocessed_with_tokens(&[ImageSize::new(448, 448)], &[128]),
                crate::types::Modality::Video,
            )
            .unwrap();

        assert_eq!(replacements[0].modality, crate::types::Modality::Video);
        assert_eq!(replacements[0].tokens.len(), 128);
        assert_eq!(replacements[0].tokens[0], 151656);
    }

    #[test]
    fn qwen3_5_video_replacement_splits_temporal_grid() {
        let tokenizer = TestTokenizer::new(&[
            ("<|video_pad|>", 151656),
            ("<|vision_start|>", 151652),
            ("<|vision_end|>", 151653),
        ]);
        let config = json!({
            "model_type": "qwen3_5",
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
        });
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3.5-4B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3.5 spec");
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(320, 256)], &[160])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![2, 16, 20], 1, 3),
            );
        let replacements = spec
            .prompt_replacements_for(&metadata, &preprocessed, crate::types::Modality::Video)
            .unwrap();

        let tokens = &replacements[0].tokens;
        assert_eq!(tokens.len(), 162);
        assert!(tokens[..80].iter().all(|&token| token == 151656));
        assert_eq!(tokens[80], 151653);
        assert_eq!(tokens[81], 151652);
        assert!(tokens[82..].iter().all(|&token| token == 151656));
    }

    #[test]
    fn qwen3_vl_video_splits_temporal_grid() {
        // Base Qwen3-VL (not the 3.5/3.6 family) must ALSO emit one vision block
        // per temporal frame. vLLM's mrope pass scans for a <|vision_start|> per
        // frame, so a flat single block crashes any multi-frame video. Regression
        // guard for the is_qwen3_5-only gate that previously left base VL flat.
        let tokenizer = TestTokenizer::new(&[
            ("<|video_pad|>", 151656),
            ("<|vision_start|>", 151652),
            ("<|vision_end|>", 151653),
        ]);
        let config = json!({
            "model_type": "qwen3_vl",
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
        });
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-VL-8B-Instruct",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3_vl spec");
        assert_eq!(spec.name(), "qwen3_vl");
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(320, 256)], &[160])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![2, 16, 20], 1, 3),
            );
        let replacements = spec
            .prompt_replacements_for(&metadata, &preprocessed, crate::types::Modality::Video)
            .unwrap();

        // Two temporal frames -> a vision_end/vision_start seam splits the 160
        // pads into two 80-token halves (mirrors the 3.5 case above).
        let tokens = &replacements[0].tokens;
        assert_eq!(tokens.len(), 162);
        assert!(tokens[..80].iter().all(|&token| token == 151656));
        assert_eq!(tokens[80], 151653);
        assert_eq!(tokens[81], 151652);
        assert!(tokens[82..].iter().all(|&token| token == 151656));
    }

    #[test]
    fn qwen2_vl_does_not_match_qwen3() {
        let tokenizer = TestTokenizer::new(&[("<image>", 999)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653,
            "vision_config": {"patch_size": 16}
        });
        let metadata = ModelMetadata {
            model_id: "Qwen3-VL-7B",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("should match qwen3");
        // Must match qwen3_vl spec, not qwen_vl
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen3_vl_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen3_vl",
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match qwen3 alias");
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen3_5_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen3_5_moe",
            "image_token_id": 151655,
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match qwen3.5 alias");
        assert_eq!(spec.name(), "qwen3_vl");
    }

    #[test]
    fn qwen4_exp_matches_alias_via_model_type() {
        let tokenizer = TestTokenizer::new(&[("<|image_pad|>", 151655)]);
        let config = json!({
            "model_type": "qwen4_exp",
            "image_token_id": 151655,
        });
        let metadata = ModelMetadata {
            model_id: "custom-model",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry
            .lookup(&metadata)
            .expect("should match qwen4_exp alias");
        assert_eq!(spec.name(), "qwen3_vl");
    }

    /// Extract the plain-text fragments a replacement spliced in (timestamps),
    /// given a tokenizer byte-encoded at `base`.
    fn replacement_text(tokens: &[crate::types::TokenId], base: u32) -> String {
        tokens
            .iter()
            .filter(|&&token| (base..base + 256).contains(&(token as u32)))
            .map(|&token| char::from((token as u32 - base) as u8))
            .collect()
    }

    fn video_timestamp_setup() -> (TestTokenizer, serde_json::Value) {
        let tokenizer = TestTokenizer::new(&[
            ("<|video_pad|>", 151656),
            ("<|vision_start|>", 151652),
            ("<|vision_end|>", 151653),
        ])
        .with_byte_encoder(1000);
        let config = json!({
            "model_type": "qwen3_vl",
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "vision_config": {"temporal_patch_size": 2}
        });
        (tokenizer, config)
    }

    #[test]
    fn qwen3_vl_video_timestamps_follow_preprocessed_fps() {
        let (tokenizer, config) = video_timestamp_setup();
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-VL-8B-Instruct",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3_vl spec");
        // second_per_grid = 0.5 with temporal_patch_size = 2 -> effective fps =
        // 4, so frame 1 is timestamped (2 + 0.5) / 4 = 0.625 -> "<0.6 seconds>"
        // instead of the 2 fps default's "<1.2 seconds>".
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(320, 256)], &[160])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![2, 16, 20], 1, 3),
            )
            .with_extra(
                "video_second_per_grid",
                ModelSpecificValue::Tensor {
                    data: vec![0.5],
                    shape: vec![1],
                },
            );
        let replacements = spec
            .prompt_replacements_for(&metadata, &preprocessed, crate::types::Modality::Video)
            .unwrap();

        let text = replacement_text(&replacements[0].tokens, 1000);
        assert!(text.contains("<0.1 seconds>"), "timestamps: {text}");
        assert!(text.contains("<0.6 seconds>"), "timestamps: {text}");
        assert!(!text.contains("<1.2 seconds>"), "timestamps: {text}");
    }

    #[test]
    fn qwen3_vl_video_timestamps_fall_back_to_default_fps() {
        let (tokenizer, config) = video_timestamp_setup();
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-VL-8B-Instruct",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3_vl spec");
        // No video_second_per_grid -> HF default 2 fps: frame 1 is
        // (2 + 0.5) / 2 = 1.25 -> "<1.2 seconds>".
        let preprocessed = test_preprocessed_with_tokens(&[ImageSize::new(320, 256)], &[160])
            .with_extra(
                "video_grid_thw",
                ModelSpecificValue::int_2d(vec![2, 16, 20], 1, 3),
            );
        let replacements = spec
            .prompt_replacements_for(&metadata, &preprocessed, crate::types::Modality::Video)
            .unwrap();

        let text = replacement_text(&replacements[0].tokens, 1000);
        assert!(text.contains("<0.2 seconds>"), "timestamps: {text}");
        assert!(text.contains("<1.2 seconds>"), "timestamps: {text}");
    }

    #[test]
    fn qwen3_vl_video_fetch_config_reads_video_preprocessor_json() {
        let tokenizer = TestTokenizer::new(&[]);
        let config = json!({"model_type": "qwen3_vl"});
        let metadata = ModelMetadata {
            model_id: "Qwen/Qwen3-VL-8B-Instruct",
            tokenizer: &tokenizer,
            config: &config,
        };
        let registry = ModelRegistry::new();
        let spec = registry.lookup(&metadata).expect("qwen3_vl spec");

        let cfg = spec.video_fetch_config(None);
        assert_eq!(cfg.strategy, VideoSamplingStrategy::Qwen3Vl);
        assert_eq!(cfg.sample_fps, 2.0);
        assert_eq!(cfg.min_frames, 4);
        assert_eq!(cfg.max_frames, 768);

        let pp_config =
            PreProcessorConfig::from_json(r#"{"fps": 1.0, "min_frames": 8, "max_frames": 64}"#)
                .expect("video preprocessor config");
        let cfg = spec.video_fetch_config(Some(&pp_config));
        assert_eq!(cfg.strategy, VideoSamplingStrategy::Qwen3Vl);
        assert_eq!(cfg.sample_fps, 1.0);
        assert_eq!(cfg.min_frames, 8);
        assert_eq!(cfg.max_frames, 64);
    }

    #[test]
    fn qwen3_video_fetch_config_uses_hf_defaults_without_json() {
        let cfg = qwen3_video_fetch_config(None);
        assert_eq!(cfg.min_frames, 4);
        assert_eq!(cfg.max_frames, 768);
        assert_eq!(cfg.sample_fps, 2.0);
        assert_eq!(cfg.strategy, VideoSamplingStrategy::Qwen3Vl);
    }

    #[test]
    fn qwen3_video_fetch_config_prefers_json_extra_fields() {
        let pp_config = PreProcessorConfig::from_json(
            r#"{"video_processor_type": "Qwen3VLVideoProcessor", "fps": 1.0, "min_frames": 8, "max_frames": 64}"#,
        )
        .expect("video preprocessor config");
        let cfg = qwen3_video_fetch_config(Some(&pp_config));
        assert_eq!(cfg.min_frames, 8);
        assert_eq!(cfg.max_frames, 64);
        assert_eq!(cfg.sample_fps, 1.0);
        assert_eq!(cfg.strategy, VideoSamplingStrategy::Qwen3Vl);

        // Partial JSON: only the given keys override, the rest keep defaults.
        let pp_config =
            PreProcessorConfig::from_json(r#"{"fps": 0.5}"#).expect("video preprocessor config");
        let cfg = qwen3_video_fetch_config(Some(&pp_config));
        assert_eq!(cfg.min_frames, 4);
        assert_eq!(cfg.max_frames, 768);
        assert_eq!(cfg.sample_fps, 0.5);
    }
}
