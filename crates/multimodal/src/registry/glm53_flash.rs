use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    media::VideoFetchConfig,
    registry::{
        MediaPartOrder, ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult,
    },
    types::{FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId},
    video_sampling::{uniform_frame_indices, VideoSamplingStrategy, VideoSourceMeta},
    vision::PreProcessorConfig,
};

const IMAGE: &str = "<|image|>";
const VIDEO: &str = "<|video|>";
const BEGIN: &str = "<|begin_of_image|>";
const END: &str = "<|end_of_image|>";

pub(super) struct Glm53FlashSpec;

/// Mirrors HF `Glm5NextVideoProcessor.sample_frames`: target count
/// `int(duration * fps)` capped by `max_frames`, a threshold walk keeping the
/// first frame at each `1 / fps` boundary (stopping at `int(duration)`
/// seconds), linspace pad/trim to the exact count, dedup, and even-count
/// padding by repeating the last frame.
///
/// `min_frames` is accepted for signature uniformity but unused: the HF
/// processor has no minimum-frame clamp. Falls back to
/// [`uniform_frame_indices`] when the source frame rate is unknown, where the
/// HF formula is undefined. Dispatched from
/// [`VideoSamplingStrategy::Glm5Next`].
pub(crate) fn glm5_next_frame_indices(
    source: &VideoSourceMeta,
    _min_frames: usize,
    max_frames: usize,
    sample_fps: f32,
    max_duration: f64,
) -> Vec<usize> {
    let total_frames = source.total_frames;
    if total_frames == 0 {
        return Vec::new();
    }
    if !(source.original_fps.is_finite() && source.original_fps > 0.0) {
        return uniform_frame_indices(source, _min_frames, max_frames, sample_fps);
    }

    let max_frame_idx = total_frames - 1;
    // HF: duration = metadata.duration or round(max_frame_idx / fps) + 1.
    let duration = source
        .duration_seconds
        .filter(|duration| duration.is_finite() && *duration > 0.0)
        .unwrap_or_else(|| (max_frame_idx as f64 / source.original_fps).round() + 1.0);
    // The walk stops at the ORIGINAL duration's whole seconds, not the capped one.
    let max_seconds = duration as u64;
    let duration = if max_duration > 0.0 {
        duration.min(max_duration)
    } else {
        duration
    };

    // Python int() truncation of a non-negative float.
    let extract_t = (duration * f64::from(sample_fps)) as usize;
    let extract_t = extract_t.min(max_frames);

    let mut frame_indices: Vec<usize> = if total_frames < extract_t {
        linspace_trunc(0, max_frame_idx, extract_t)
    } else {
        let duration_per_frame = 1.0 / source.original_fps;
        let inv_fps = 1.0 / f64::from(sample_fps);
        let mut indices = Vec::new();
        let mut current_second = 0.0_f64;
        for frame_index in 0..total_frames {
            if frame_index as f64 * duration_per_frame >= current_second {
                current_second += inv_fps;
                indices.push(frame_index);
                if current_second >= max_seconds as f64 {
                    break;
                }
            }
        }
        indices
    };

    if frame_indices.len() < extract_t {
        let (start, end) = match (frame_indices.first(), frame_indices.last()) {
            (Some(&first), Some(&last)) => (first, last),
            _ => (0, max_frame_idx),
        };
        frame_indices = linspace_trunc(start, end, extract_t);
    } else if frame_indices.len() > extract_t {
        frame_indices = linspace_trunc(0, max_frame_idx, extract_t);
    }

    let mut seen = std::collections::HashSet::with_capacity(frame_indices.len());
    let mut uniq: Vec<usize> = frame_indices
        .into_iter()
        .filter(|index| seen.insert(*index))
        .collect();
    if uniq.len() % 2 == 1 {
        // Odd counts only occur with a non-empty vec (0 is even).
        if let Some(&last) = uniq.last() {
            uniq.push(last);
        }
    }
    uniq
}

/// `np.linspace(start, end, num, dtype=int)` parity: float steps, then
/// truncation toward zero (equivalent to floor for these non-negative
/// values). Like numpy, the last element is pinned to `end` exactly —
/// float step accumulation could otherwise truncate to `end - 1`.
fn linspace_trunc(start: usize, end: usize, num: usize) -> Vec<usize> {
    match num {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (end - start) as f64 / (num - 1) as f64;
            let mut values: Vec<usize> = (0..num)
                .map(|i| (start as f64 + i as f64 * step) as usize)
                .collect();
            if let Some(last) = values.last_mut() {
                *last = end;
            }
            values
        }
    }
}

impl Glm53FlashSpec {
    fn image_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        metadata
            .config_u32(&["image_token_id"])
            .map(|id| id as TokenId)
            .ok_or_else(|| ModelRegistryError::MissingConfigField {
                field: "image_token_id".to_string(),
            })
    }

    fn encode(metadata: &ModelMetadata, text: &str) -> RegistryResult<Vec<TokenId>> {
        let ids = metadata.tokenizer.encode_text(text).ok_or_else(|| {
            ModelRegistryError::TextEncodingFailed {
                spec: "glm53_flash",
                text: text.to_string(),
            }
        })?;
        Ok(ids.into_iter().map(|id| id as TokenId).collect())
    }

    fn video_grid_t(input: &PreprocessedEncoderInputs) -> RegistryResult<Vec<usize>> {
        let Some(ModelSpecificValue::IntTensor { data, shape }) =
            input.model_specific.get("video_grid_thw")
        else {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw".to_string(),
            });
        };
        if shape.len() != 2 || shape[1] != 3 || data.len() != shape[0] * 3 {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw".to_string(),
            });
        }
        data.as_chunks::<3>()
            .0
            .iter()
            .map(|row| {
                usize::try_from(row[0]).map_err(|_| ModelRegistryError::InvalidPreprocessedField {
                    field: "video_grid_thw".to_string(),
                })
            })
            .collect()
    }

    fn unsupported(modality: Modality) -> ModelRegistryError {
        ModelRegistryError::UnsupportedModality {
            spec: "glm53_flash",
            modality,
        }
    }
}

impl ModelProcessorSpec for Glm53FlashSpec {
    fn name(&self) -> &'static str {
        "glm53_flash"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        let id = metadata.model_id.to_ascii_lowercase();
        matches!(
            metadata.config_model_type(),
            Some("glm53_flash" | "glm5_next")
        ) || ["glm-5.3-flash", "glm5.3-flash", "glm-5-next", "glm5-next"]
            .iter()
            .any(|name| id.contains(name))
    }

    fn media_part_order(&self) -> MediaPartOrder {
        MediaPartOrder::Authored
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(IMAGE.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_id(metadata)
    }

    fn placeholder_token_for(
        &self,
        _metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => Ok(IMAGE.to_string()),
            Modality::Video => Ok(VIDEO.to_string()),
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn placeholder_token_id_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<TokenId> {
        match modality {
            Modality::Image | Modality::Video => Self::image_id(metadata),
            _ => Err(Self::unsupported(modality)),
        }
    }

    fn modality_limits(
        &self,
        metadata: &ModelMetadata,
    ) -> RegistryResult<HashMap<Modality, usize>> {
        // Advertise only what the checkpoint can serve, so an incapable
        // derivative is rejected at validate_media_request instead of after
        // a full media fetch + preprocess. Images need the placeholder id;
        // video additionally splices <|begin_of_image|>/<|end_of_image|>
        // frame markers into the prompt.
        let mut limits = HashMap::new();
        if Self::image_id(metadata).is_ok() {
            limits.insert(Modality::Image, 10);
            if metadata.token_id(BEGIN).is_ok() && metadata.token_id(END).is_ok() {
                limits.insert(Modality::Video, 1);
            }
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
        VideoFetchConfig {
            // HF `Glm5NextVideoProcessor` has no minimum-frame clamp; keep the
            // connector's config validation (min >= 1) satisfied.
            min_frames: 1,
            max_frames: video_preprocessor_config
                .and_then(|config| config.get_extra::<usize>("max_frames"))
                .unwrap_or(2048),
            sample_fps: video_preprocessor_config
                .and_then(|config| config.get_extra::<f32>("fps"))
                .unwrap_or(2.0),
            strategy: VideoSamplingStrategy::Glm5Next {
                max_duration: video_preprocessor_config
                    .and_then(|config| config.get_extra::<f64>("max_duration"))
                    .unwrap_or(0.0),
            },
        }
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        input: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let id = Self::image_id(metadata)?;
        Ok(input
            .feature_token_counts
            .iter()
            .map(|&count| PromptReplacement::repeated(Modality::Image, IMAGE, id, count))
            .collect())
    }

    fn prompt_replacements_for(
        &self,
        metadata: &ModelMetadata,
        input: &PreprocessedEncoderInputs,
        modality: Modality,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        if modality == Modality::Image {
            return self.prompt_replacements(metadata, input);
        }
        if modality != Modality::Video {
            return Err(Self::unsupported(modality));
        }

        let grids = Self::video_grid_t(input)?;
        if grids.len() != input.feature_token_counts.len() {
            return Err(ModelRegistryError::InvalidPreprocessedField {
                field: "video_grid_thw item count".to_string(),
            });
        }
        let image_id = Self::image_id(metadata)?;
        let begin = metadata.token_id(BEGIN)?;
        let end = metadata.token_id(END)?;
        // The paired vision processor always emits this; a missing or
        // wrongly-typed value means the pipeline is broken, and defaulting
        // would silently caption every frame with wrong timestamps while
        // the grid field next to it fails loudly.
        let seconds = match input.model_specific.get("video_second_per_grid") {
            Some(ModelSpecificValue::Tensor { data, .. }) if !data.is_empty() => data[0],
            _ => {
                return Err(ModelRegistryError::InvalidPreprocessedField {
                    field: "video_second_per_grid".to_string(),
                })
            }
        };

        input
            .feature_token_counts
            .iter()
            .zip(grids)
            .map(|(&count, grid_t)| {
                if grid_t == 0 || !count.is_multiple_of(grid_t) {
                    return Err(ModelRegistryError::InvalidPreprocessedField {
                        field: "video_grid_thw temporal token layout".to_string(),
                    });
                }
                let per_grid = count / grid_t;
                let mut tokens = Vec::new();
                let mut ranges = Vec::with_capacity(grid_t);
                for index in 0..grid_t {
                    tokens.push(begin);
                    ranges.push(PlaceholderRange {
                        offset: tokens.len(),
                        length: per_grid,
                    });
                    tokens.extend(std::iter::repeat_n(image_id, per_grid));
                    tokens.push(end);
                    tokens.extend(Self::encode(
                        metadata,
                        &format!("{:.1} seconds", index as f32 * seconds),
                    )?);
                }
                Ok(PromptReplacement::sequence(Modality::Video, VIDEO, tokens)
                    .with_feature_ranges(ranges))
            })
            .collect()
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
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

    use super::*;
    use crate::registry::{test_helpers::*, ModelRegistry};

    const IMAGE_ID: u32 = 154854;
    const BEGIN_ID: u32 = 154830;
    const END_ID: u32 = 154831;
    fn tokenizer() -> TestTokenizer {
        TestTokenizer::new(&[(IMAGE, IMAGE_ID), (BEGIN, BEGIN_ID), (END, END_ID)])
            .with_byte_encoder(1000)
    }

    #[test]
    fn matches_names_and_builds_video_timestamps() {
        let tokenizer = tokenizer();
        let config = json!({"model_type":"glm53_flash", "image_token_id":IMAGE_ID});
        let metadata = ModelMetadata {
            model_id: "zai-org/GLM-5.3-Flash",
            tokenizer: &tokenizer,
            config: &config,
        };
        let mut input = test_preprocessed_with_tokens(&[], &[4]);
        input.model_specific.insert(
            "video_grid_thw".into(),
            ModelSpecificValue::int_2d(vec![2, 2, 4], 1, 3),
        );
        input.model_specific.insert(
            "video_second_per_grid".into(),
            ModelSpecificValue::Tensor {
                data: vec![1.0],
                shape: vec![1],
            },
        );
        let replacement = ModelRegistry::new()
            .lookup(&metadata)
            .unwrap()
            .prompt_replacements_for(&metadata, &input, Modality::Video)
            .unwrap()
            .pop()
            .unwrap();

        let ranges = replacement.feature_ranges.unwrap();
        assert_eq!(
            ranges
                .iter()
                .map(|range| (range.offset, range.length))
                .collect::<Vec<_>>(),
            vec![(1, 2), (16, 2)]
        );
        assert_eq!(replacement.tokens[0], BEGIN_ID as TokenId);
        assert_eq!(replacement.tokens[1], IMAGE_ID as TokenId);
        assert_eq!(replacement.tokens[3], END_ID as TokenId);
        let timestamp = Glm53FlashSpec::encode(&metadata, "0.0 seconds").unwrap();
        assert_eq!(&replacement.tokens[4..15], timestamp.as_slice());

        let legacy = json!({"model_type":"glm5_next"});
        assert!(Glm53FlashSpec.matches(&ModelMetadata {
            model_id: "legacy",
            tokenizer: &tokenizer,
            config: &legacy,
        }));
    }

    #[test]
    fn modality_adverts_follow_checkpoint_capability() {
        // Full capability: image placeholder id + frame-marker tokens.
        let full = tokenizer();
        let config = json!({"model_type":"glm53_flash", "image_token_id":IMAGE_ID});
        let limits = Glm53FlashSpec
            .modality_limits(&ModelMetadata {
                model_id: "capable",
                tokenizer: &full,
                config: &config,
            })
            .unwrap();
        assert_eq!(limits.get(&Modality::Image), Some(&10));
        assert_eq!(limits.get(&Modality::Video), Some(&1));

        // No frame markers in the vocab: image-only, video rejected up
        // front instead of after a full clip fetch + preprocess.
        let no_markers = TestTokenizer::new(&[(IMAGE, IMAGE_ID)]).with_byte_encoder(1000);
        let limits = Glm53FlashSpec
            .modality_limits(&ModelMetadata {
                model_id: "image-only",
                tokenizer: &no_markers,
                config: &config,
            })
            .unwrap();
        assert_eq!(limits.get(&Modality::Image), Some(&10));
        assert!(!limits.contains_key(&Modality::Video));

        // No image token id in the config: nothing advertised.
        let no_id = json!({"model_type":"glm53_flash"});
        let limits = Glm53FlashSpec
            .modality_limits(&ModelMetadata {
                model_id: "text-only",
                tokenizer: &full,
                config: &no_id,
            })
            .unwrap();
        assert!(limits.is_empty());
    }

    #[test]
    fn video_error_branches_fail_loudly() {
        let tokenizer = tokenizer();
        let config = json!({"model_type":"glm53_flash", "image_token_id":IMAGE_ID});
        let metadata = ModelMetadata {
            model_id: "zai-org/GLM-5.3-Flash",
            tokenizer: &tokenizer,
            config: &config,
        };
        let spec = Glm53FlashSpec;

        // Missing video_second_per_grid: the paired processor always emits
        // it, so absence is a broken pipeline, not a 1.0 default.
        let mut input = test_preprocessed_with_tokens(&[], &[4]);
        input.model_specific.insert(
            "video_grid_thw".into(),
            ModelSpecificValue::int_2d(vec![2, 2, 4], 1, 3),
        );
        let err = spec
            .prompt_replacements_for(&metadata, &input, Modality::Video)
            .unwrap_err();
        assert!(matches!(
            err,
            ModelRegistryError::InvalidPreprocessedField { ref field }
                if field == "video_second_per_grid"
        ));

        // Wrong grid shape (row width != 3).
        let mut input = test_preprocessed_with_tokens(&[], &[4]);
        input.model_specific.insert(
            "video_grid_thw".into(),
            ModelSpecificValue::int_2d(vec![2, 2], 1, 2),
        );
        assert!(matches!(
            spec.prompt_replacements_for(&metadata, &input, Modality::Video),
            Err(ModelRegistryError::InvalidPreprocessedField { .. })
        ));

        // Token count not divisible by grid_t.
        let mut input = test_preprocessed_with_tokens(&[], &[5]);
        input.model_specific.insert(
            "video_grid_thw".into(),
            ModelSpecificValue::int_2d(vec![2, 2, 4], 1, 3),
        );
        input.model_specific.insert(
            "video_second_per_grid".into(),
            ModelSpecificValue::Tensor {
                data: vec![1.0],
                shape: vec![1],
            },
        );
        assert!(matches!(
            spec.prompt_replacements_for(&metadata, &input, Modality::Video),
            Err(ModelRegistryError::InvalidPreprocessedField { .. })
        ));
    }

    // Glm5Next frame sampling: HF Glm5NextVideoProcessor.sample_frames parity.
    // The same math is pinned against real transformers output by
    // tests/video_sampling_golden.rs.

    fn glm_source(total_frames: usize, fps: f64, duration: Option<f64>) -> VideoSourceMeta {
        VideoSourceMeta {
            total_frames,
            original_fps: fps,
            duration_seconds: duration,
        }
    }

    fn glm_plan(source: &VideoSourceMeta, max_frames: usize, fps: f32) -> Vec<usize> {
        VideoSamplingStrategy::Glm5Next { max_duration: 0.0 }
            .plan(source, 1, max_frames, fps)
            .indices
    }

    #[test]
    fn glm_threshold_walk_at_2fps() {
        // duration = 98/30 = 3.267s; extract_t = int(3.267 * 2) = 6; the walk
        // keeps the first frame at each 0.5s boundary and stops at 3s.
        let indices = glm_plan(&glm_source(98, 30.0, Some(98.0 / 30.0)), 2048, 2.0);
        assert_eq!(indices, vec![0, 15, 30, 45, 60, 75]);
    }

    #[test]
    fn glm_duration_falls_back_to_frame_count_estimate() {
        // No duration: round(97/30) + 1 = 4s; extract_t = 8; the walk finds 7
        // frames (stops at 90), so linspace pads from the walked span.
        let indices = glm_plan(&glm_source(98, 30.0, None), 2048, 2.0);
        assert_eq!(indices, vec![0, 12, 25, 38, 51, 64, 77, 90]);
    }

    #[test]
    fn glm_odd_count_repeats_last_frame() {
        // extract_t = int(0.5 * 2) = 1; the single walked frame is duplicated
        // to make the count even for temporal patching.
        let indices = glm_plan(&glm_source(2, 2.0, Some(0.5)), 2048, 2.0);
        assert_eq!(indices, vec![0, 0]);
    }

    #[test]
    fn glm_underfull_walk_pads_with_linspace_and_repeats_last() {
        // duration = 2.9s: extract_t = 5 but the walk stops after 4 frames
        // (current_second hits 2.0), so a linspace over the walked span pads
        // to 5, and the odd count repeats the last frame. HF: [0, 3, 7, 11,
        // 15, 15].
        let indices = glm_plan(&glm_source(29, 10.0, Some(2.9)), 2048, 2.0);
        assert_eq!(indices, vec![0, 3, 7, 11, 15, 15]);
    }

    #[test]
    fn glm_overfull_walk_trims_with_linspace() {
        // duration = 2.4s: extract_t = 4 but the walk finds 5 frames
        // (0, 5, 10, 15, 20) before the 2s stop, so a full-span linspace trim
        // applies. HF: [0, 5, 10, 15].
        let indices = glm_plan(&glm_source(24, 10.0, Some(2.4)), 2048, 2.0);
        assert_eq!(indices, vec![0, 5, 10, 15]);
    }

    #[test]
    fn glm_max_frames_caps_extract_count() {
        // int(166.67 * 2) = 333 -> capped to 64; the overfull trim then spans
        // the whole video.
        let indices = glm_plan(&glm_source(5000, 30.0, Some(5000.0 / 30.0)), 64, 2.0);
        assert_eq!(indices.len(), 64);
        assert_eq!(indices[0], 0);
        assert_eq!(*indices.last().unwrap(), 4999);
    }

    #[test]
    fn glm_unknown_fps_falls_back_to_uniform() {
        for bad_fps in [0.0, -1.0, f64::NAN] {
            let glm = VideoSamplingStrategy::Glm5Next { max_duration: 0.0 }.plan(
                &glm_source(100, bad_fps, None),
                4,
                8,
                2.0,
            );
            let uniform =
                VideoSamplingStrategy::Uniform.plan(&glm_source(100, bad_fps, None), 4, 8, 2.0);
            assert_eq!(glm.indices, uniform.indices);
        }
    }

    #[test]
    fn glm_video_fetch_config_reads_video_preprocessor_json() {
        let default_cfg = Glm53FlashSpec.video_fetch_config(None);
        assert_eq!(default_cfg.min_frames, 1);
        assert_eq!(default_cfg.max_frames, 2048);
        assert_eq!(default_cfg.sample_fps, 2.0);
        assert_eq!(
            default_cfg.strategy,
            VideoSamplingStrategy::Glm5Next { max_duration: 0.0 }
        );

        let pp_config =
            PreProcessorConfig::from_json(r#"{"fps": 1.0, "max_frames": 64, "max_duration": 300}"#)
                .expect("video preprocessor config");
        let cfg = Glm53FlashSpec.video_fetch_config(Some(&pp_config));
        assert_eq!(cfg.sample_fps, 1.0);
        assert_eq!(cfg.max_frames, 64);
        assert_eq!(
            cfg.strategy,
            VideoSamplingStrategy::Glm5Next {
                max_duration: 300.0
            }
        );
    }
}
