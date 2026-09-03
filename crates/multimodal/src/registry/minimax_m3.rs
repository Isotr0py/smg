use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    encoder_inputs::{ModelSpecificValue, PreprocessedEncoderInputs},
    media::VideoFetchConfig,
    registry::{ModelMetadata, ModelProcessorSpec, ModelRegistryError, RegistryResult},
    types::{FieldLayout, Modality, PlaceholderRange, PromptReplacement, TokenId},
    video_sampling::{VideoSamplingStrategy, VideoSourceMeta},
    vision::PreProcessorConfig,
};

/// Maximum images accepted in one request (MiniMax-M3 spec 1.3.6).
const MAX_IMAGES_PER_REQUEST: usize = 200;

/// Maximum videos accepted in one request (MiniMax-M3 spec 1.3.6).
const MAX_VIDEOS_PER_REQUEST: usize = 20;

/// Mirrors vLLM's `MiniMaxM3VideoBackend.compute_frames_index_to_sample`
/// (models/minimax_m3/common/mm_preprocess.py): a ceil-interval walk keeping
/// one frame every `1 / sample_fps` seconds of source time. There is no
/// min/max frame clamp; invalid or unknown rates degrade to the first frame,
/// matching the reference's guard (`[0] if total_frames > 0 else []`).
/// `min_frames`/`max_frames` are accepted for signature uniformity but unused.
///
/// Non-uniform by design: consumers that need per-frame timing must read the
/// sampled indices back from `VideoSampleInfo.frame_indices`.
pub(crate) fn minimax_m3_frame_indices(
    source: &VideoSourceMeta,
    _min_frames: usize,
    _max_frames: usize,
    sample_fps: f32,
) -> Vec<usize> {
    let total_frames = source.total_frames;
    let video_fps = source.original_fps;
    let fps = f64::from(sample_fps);
    if total_frames == 0 {
        return Vec::new();
    }
    if !(video_fps.is_finite() && video_fps > 0.0 && fps.is_finite() && fps > 0.0) {
        return vec![0];
    }

    let read_time_interval = 1.0 / fps;
    // The reference subtracts an epsilon so a frame landing exactly on the
    // boundary does not get picked twice.
    let eps = 1e-4;
    let mut indices = Vec::new();
    let mut prev_kept_ts = f64::NEG_INFINITY;
    loop {
        let target_frame = match indices.last() {
            None => 0,
            Some(&last) => {
                let target_ts = prev_kept_ts + read_time_interval - eps;
                ((target_ts * video_fps).ceil() as usize).max(last + 1)
            }
        };
        if target_frame >= total_frames {
            break;
        }
        indices.push(target_frame);
        prev_kept_ts = target_frame as f64 / video_fps;
    }
    if indices.is_empty() {
        indices.push(0);
    }
    indices
}

/// MiniMax-M3 vision spec.
///
/// M3's media tokens carry the same `]<]...[>[` namespace framing as its tool
/// calls. Unlike the Qwen templates, M3's chat template renders a bare
/// `]<]image[>[` (or `]<]video[>[`) with no surrounding markers, so this spec
/// owns the whole wrapper: each placeholder expands to
/// `<start> + N * <pad> + <end>`, with
/// N = `grid_t * grid_h * grid_w / merge_size^2`. That mirrors vLLM's
/// `_get_prompt_updates`, which builds
/// `[start_token_id] + [image_token_id] * N + [end_token_id]`.
///
/// The start/end markers are modality-specific: M3's vocabulary carries a
/// separate `]<]start of video[>[` / `]<]end of video[>[` pair alongside the
/// image one, unlike Qwen's modality-neutral `<|vision_start|>`.
pub(super) struct MiniMaxM3VisionSpec;

impl MiniMaxM3VisionSpec {
    const IMAGE_TOKEN: &'static str = "]<]image[>[";
    const VIDEO_TOKEN: &'static str = "]<]video[>[";
    const IMAGE_START_TOKEN: &'static str = "]<]start of image[>[";
    const IMAGE_END_TOKEN: &'static str = "]<]end of image[>[";
    const VIDEO_START_TOKEN: &'static str = "]<]start of video[>[";
    const VIDEO_END_TOKEN: &'static str = "]<]end of video[>[";

    /// The structural markers wrapping one modality's feature run.
    fn wrapper_tokens(modality: Modality) -> RegistryResult<(&'static str, &'static str)> {
        match modality {
            Modality::Image => Ok((Self::IMAGE_START_TOKEN, Self::IMAGE_END_TOKEN)),
            Modality::Video => Ok((Self::VIDEO_START_TOKEN, Self::VIDEO_END_TOKEN)),
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: "minimax_m3",
                modality,
            }),
        }
    }

    /// The repeated feature token for images.
    ///
    /// `image_token_index` is the checkpoint's own declaration; the tokenizer
    /// lookup is the fallback for checkpoints that omit it.
    fn image_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        match metadata.config_u32(&["image_token_index"]) {
            Some(id) => Ok(id as TokenId),
            None => metadata.token_id(Self::IMAGE_TOKEN),
        }
    }

    /// The repeated feature token for videos.
    fn video_token_id(metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        match metadata.config_u32(&["video_token_index"]) {
            Some(id) => Ok(id as TokenId),
            None => metadata.token_id(Self::VIDEO_TOKEN),
        }
    }

    /// Whether the checkpoint declares video support.
    fn supports_video(metadata: &ModelMetadata) -> bool {
        metadata.config_u32(&["video_token_index"]).is_some()
            || metadata.token_id(Self::VIDEO_TOKEN).is_ok()
    }

    /// Build `[start] + N * pad + [end]` for one media item.
    fn wrapped_replacement(
        metadata: &ModelMetadata,
        modality: Modality,
        placeholder_token: &str,
        pad_token_id: TokenId,
        num_tokens: usize,
    ) -> RegistryResult<PromptReplacement> {
        let (start_token, end_token) = Self::wrapper_tokens(modality)?;
        let start_id = metadata.token_id(start_token)?;
        let end_id = metadata.token_id(end_token)?;

        let mut tokens = Vec::with_capacity(num_tokens + 2);
        tokens.push(start_id);
        tokens.extend(std::iter::repeat_n(pad_token_id, num_tokens));
        tokens.push(end_id);

        Ok(
            PromptReplacement::sequence(modality, placeholder_token, tokens)
                // The encoder features occupy only the padded middle; the two
                // markers around them are structural.
                //
                // `structural_prefix` stays 0: it counts markers the chat
                // template emits *before* the placeholder, which `expand_tokens`
                // folds in by widening the range backwards without re-emitting
                // them. M3's template emits a bare placeholder and both markers
                // are inside `tokens`, so a non-zero prefix would report a range
                // starting one token too early.
                .with_feature_span(1, num_tokens),
        )
    }

    /// Temporal grid depth for one video, from `video_grid_thw[0]`.
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

    /// Build the per-frame video body.
    ///
    /// M3 lays video out as one `]<]start of video[>[` .. `]<]end of video[>[`
    /// block **per temporal frame**, each holding `grid_h * grid_w / merge^2`
    /// pad tokens — not one flat block over the whole clip. vLLM's
    /// `_get_prompt_updates` builds the same shape:
    ///
    /// ```text
    /// for frame in 0..grid_t:
    ///     [start] + [video_token] * M + [end]
    /// ```
    ///
    /// When the decode backend carried the sampled frame indices and source
    /// fps through (`video_frames_indices` / `video_source_fps`, currently the
    /// OpenCV path), each frame is additionally prefixed with a
    /// `]<]X.X seconds[>[` marker — `frames_indices[min(i * tps, len - 1)] /
    /// source_fps`, mirroring vLLM exactly. Without them the frame blocks
    /// alone are emitted, which vLLM documents as the aligned fallback.
    ///
    /// Returns `None` when the layout cannot apply (unknown or single frame, or
    /// a token count that does not divide evenly), leaving the caller on the
    /// single-block path.
    fn per_frame_video_tokens(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        pad_token_id: TokenId,
        num_tokens: usize,
        grid_t: usize,
    ) -> RegistryResult<Option<(Vec<TokenId>, Vec<PlaceholderRange>)>> {
        if grid_t <= 1 || num_tokens == 0 || !num_tokens.is_multiple_of(grid_t) {
            return Ok(None);
        }
        let start_id = metadata.token_id(Self::VIDEO_START_TOKEN)?;
        let end_id = metadata.token_id(Self::VIDEO_END_TOKEN)?;
        let timestamps = Self::video_frame_timestamps(preprocessed);
        // `img_token_compression_config.temporal_patch_size`, default 2.
        let temporal_patch_size = metadata
            .config_u32(&["img_token_compression_config", "temporal_patch_size"])
            .unwrap_or(2) as usize;

        let per_frame = num_tokens / grid_t;
        let mut tokens = Vec::with_capacity(num_tokens + 2 * grid_t);
        let mut ranges = Vec::with_capacity(grid_t);
        for frame_idx in 0..grid_t {
            if let Some((indices, source_fps)) = &timestamps {
                let sampled = indices[(frame_idx * temporal_patch_size).min(indices.len() - 1)];
                let seconds = sampled as f64 / source_fps;
                tokens.extend(Self::encode(
                    metadata,
                    &format!("]<]{seconds:.1} seconds[>["),
                )?);
            }
            tokens.push(start_id);
            ranges.push(PlaceholderRange {
                offset: tokens.len(),
                length: per_frame,
            });
            tokens.extend(std::iter::repeat_n(pad_token_id, per_frame));
            tokens.push(end_id);
        }
        Ok(Some((tokens, ranges)))
    }

    /// The sampled source-frame indices and source fps carried through the
    /// video processor, when available. Both must be present and sane.
    fn video_frame_timestamps(preprocessed: &PreprocessedEncoderInputs) -> Option<(Vec<i64>, f64)> {
        let indices = match preprocessed.model_specific.get("video_frames_indices") {
            Some(ModelSpecificValue::IntTensor { data, .. }) if !data.is_empty() => data,
            _ => return None,
        };
        let source_fps = match preprocessed.model_specific.get("video_source_fps") {
            Some(ModelSpecificValue::Tensor { data, .. }) if !data.is_empty() => f64::from(data[0]),
            _ => return None,
        };
        (source_fps.is_finite() && source_fps > 0.0).then(|| (indices.clone(), source_fps))
    }

    fn encode(metadata: &ModelMetadata, text: &str) -> RegistryResult<Vec<TokenId>> {
        let ids = metadata.tokenizer.encode_text(text).ok_or_else(|| {
            ModelRegistryError::TextEncodingFailed {
                spec: "minimax_m3",
                text: text.to_string(),
            }
        })?;
        Ok(ids.into_iter().map(|id| id as TokenId).collect())
    }

    fn replacements_for(
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
        modality: Modality,
        placeholder_token: &str,
        pad_token_id: TokenId,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        preprocessed
            .feature_token_counts
            .iter()
            .map(|&num_tokens| {
                Self::wrapped_replacement(
                    metadata,
                    modality,
                    placeholder_token,
                    pad_token_id,
                    num_tokens,
                )
            })
            .collect()
    }
}

impl ModelProcessorSpec for MiniMaxM3VisionSpec {
    fn name(&self) -> &'static str {
        "minimax_m3"
    }

    fn matches(&self, metadata: &ModelMetadata) -> bool {
        if metadata
            .config_model_type()
            .is_some_and(|mt| mt == "minimax_m3_vl")
        {
            return true;
        }
        let id = metadata.model_id.to_ascii_lowercase();
        id.contains("minimax") && id.contains("m3")
    }

    fn placeholder_token(&self, _metadata: &ModelMetadata) -> RegistryResult<String> {
        Ok(Self::IMAGE_TOKEN.to_string())
    }

    fn placeholder_token_id(&self, metadata: &ModelMetadata) -> RegistryResult<TokenId> {
        Self::image_token_id(metadata)
    }

    fn placeholder_token_for(
        &self,
        metadata: &ModelMetadata,
        modality: Modality,
    ) -> RegistryResult<String> {
        match modality {
            Modality::Image => self.placeholder_token(metadata),
            Modality::Video => Ok(Self::VIDEO_TOKEN.to_string()),
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
            Modality::Image => Self::image_token_id(metadata),
            Modality::Video => Self::video_token_id(metadata),
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
        // MiniMax-M3 accepts up to 200 images per request (spec 1.3.6), far
        // above the Qwen-family default of 10.
        let mut limits = HashMap::from([(Modality::Image, MAX_IMAGES_PER_REQUEST)]);
        if Self::supports_video(metadata) {
            limits.insert(Modality::Video, MAX_VIDEOS_PER_REQUEST);
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
            // M3's ceil-interval sampling has no frame-count clamps; the
            // min/max fields only feed the connector's config validation and
            // the ffmpeg fallback's generic bounds.
            min_frames: 1,
            // The vendored MiniMaxM3VLVideoProcessor default.
            max_frames: video_preprocessor_config
                .and_then(|config| config.get_extra::<usize>("max_frames"))
                .unwrap_or(768),
            // M3 samples at 1 fps by default (unlike the Qwen family's 2).
            sample_fps: video_preprocessor_config
                .and_then(|config| config.get_extra::<f32>("fps"))
                .unwrap_or(1.0),
            strategy: VideoSamplingStrategy::MiniMaxM3,
            max_long_side_pixel: None,
        }
    }

    fn prompt_replacements(
        &self,
        metadata: &ModelMetadata,
        preprocessed: &PreprocessedEncoderInputs,
    ) -> RegistryResult<Vec<PromptReplacement>> {
        let pad_token_id = Self::image_token_id(metadata)?;
        let placeholder_token = self.placeholder_token(metadata)?;
        Self::replacements_for(
            metadata,
            preprocessed,
            Modality::Image,
            &placeholder_token,
            pad_token_id,
        )
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
                let pad_token_id = Self::video_token_id(metadata)?;
                let placeholder_token = self.placeholder_token_for(metadata, Modality::Video)?;
                let grid_t = Self::video_grid_t(preprocessed);

                preprocessed
                    .feature_token_counts
                    .iter()
                    .map(|&num_tokens| {
                        let per_frame = grid_t.and_then(|grid_t| {
                            Self::per_frame_video_tokens(
                                metadata,
                                preprocessed,
                                pad_token_id,
                                num_tokens,
                                grid_t,
                            )
                            .transpose()
                        });

                        match per_frame {
                            Some(Ok((tokens, ranges))) => Ok(PromptReplacement::sequence(
                                Modality::Video,
                                &placeholder_token,
                                tokens,
                            )
                            .with_feature_ranges(ranges)),
                            Some(Err(err)) => Err(err),
                            None => Self::wrapped_replacement(
                                metadata,
                                Modality::Video,
                                &placeholder_token,
                                pad_token_id,
                                num_tokens,
                            ),
                        }
                    })
                    .collect()
            }
            _ => Err(ModelRegistryError::UnsupportedModality {
                spec: self.name(),
                modality,
            }),
        }
    }

    fn field_layouts(&self) -> HashMap<String, FieldLayout> {
        // Mirrors vLLM's `_get_mm_fields_config` for M3: the pixel tensors are
        // flat over patches and sliced per item by the grid product, while the
        // grid triples are batched one row per item.
        HashMap::from([
            (
                "pixel_values".to_string(),
                FieldLayout::flat("patches_per_image"),
            ),
            ("image_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_image".to_string(), FieldLayout::Batched),
            (
                "pixel_values_videos".to_string(),
                FieldLayout::flat("patches_per_video"),
            ),
            ("video_grid_thw".to_string(), FieldLayout::Batched),
            ("patches_per_video".to_string(), FieldLayout::Batched),
        ])
    }

    fn keep_on_cpu_keys(&self) -> Vec<String> {
        // vLLM marks both grid tensors keep_on_cpu=True.
        vec!["image_grid_thw".to_string(), "video_grid_thw".to_string()]
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::registry::{ModelMetadata, Tokenizer};

    /// Vocabulary ids for M3's media markers, as the checkpoint declares them.
    const IMAGE_ID: TokenId = 200_025;
    const VIDEO_ID: TokenId = 200_026;
    const IMAGE_START_ID: TokenId = 200_029;
    const IMAGE_END_ID: TokenId = 200_030;
    const VIDEO_START_ID: TokenId = 200_031;
    const VIDEO_END_ID: TokenId = 200_032;

    struct M3Tokenizer;

    impl Tokenizer for M3Tokenizer {
        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                MiniMaxM3VisionSpec::IMAGE_TOKEN => Some(IMAGE_ID as u32),
                MiniMaxM3VisionSpec::VIDEO_TOKEN => Some(VIDEO_ID as u32),
                MiniMaxM3VisionSpec::IMAGE_START_TOKEN => Some(IMAGE_START_ID as u32),
                MiniMaxM3VisionSpec::IMAGE_END_TOKEN => Some(IMAGE_END_ID as u32),
                MiniMaxM3VisionSpec::VIDEO_START_TOKEN => Some(VIDEO_START_ID as u32),
                MiniMaxM3VisionSpec::VIDEO_END_TOKEN => Some(VIDEO_END_ID as u32),
                _ => None,
            }
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            match id {
                id if id == IMAGE_ID as u32 => Some(MiniMaxM3VisionSpec::IMAGE_TOKEN.to_string()),
                id if id == VIDEO_ID as u32 => Some(MiniMaxM3VisionSpec::VIDEO_TOKEN.to_string()),
                _ => None,
            }
        }

        fn encode_text(&self, text: &str) -> Option<Vec<u32>> {
            // Known markers map to their declared ids; anything else (the
            // timestamp text) byte-encodes at a high base so tests can
            // reconstruct it.
            self.token_to_id(text)
                .map(|id| vec![id])
                .or_else(|| Some(text.bytes().map(|b| 300_000 + u32::from(b)).collect()))
        }
    }

    /// Decode byte-encoded (base 300_000) text fragments out of a token run.
    fn encoded_text(tokens: &[TokenId]) -> String {
        tokens
            .iter()
            .filter(|&&t| (300_000..300_256).contains(&(t as u32)))
            .map(|&t| char::from((t as u32 - 300_000) as u8))
            .collect()
    }

    fn metadata() -> ModelMetadata<'static> {
        static CONFIG: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
        static TOKENIZER: M3Tokenizer = M3Tokenizer;
        let config = CONFIG.get_or_init(|| {
            json!({
                "model_type": "minimax_m3_vl",
                "image_token_index": IMAGE_ID,
                "video_token_index": VIDEO_ID,
            })
        });
        ModelMetadata {
            model_id: "MiniMaxAI/MiniMax-M3",
            config,
            tokenizer: &TOKENIZER,
        }
    }

    fn preprocessed(counts: Vec<usize>) -> PreprocessedEncoderInputs {
        let item_sizes = vec![(224, 224); counts.len()];
        PreprocessedEncoderInputs::new(ndarray::Array2::<f32>::zeros((1, 1)), counts, item_sizes)
    }

    #[test]
    fn matches_by_model_type_and_id() {
        let spec = MiniMaxM3VisionSpec;
        assert!(spec.matches(&metadata()));
    }

    #[test]
    fn placeholder_tokens_use_the_m3_namespace() {
        let spec = MiniMaxM3VisionSpec;
        let meta = metadata();

        assert_eq!(spec.placeholder_token(&meta).unwrap(), "]<]image[>[");
        assert_eq!(
            spec.placeholder_token_for(&meta, Modality::Video).unwrap(),
            "]<]video[>["
        );
        assert_eq!(spec.placeholder_token_id(&meta).unwrap(), IMAGE_ID);
        assert_eq!(
            spec.placeholder_token_id_for(&meta, Modality::Video)
                .unwrap(),
            VIDEO_ID
        );
    }

    #[test]
    fn image_replacement_is_wrapped_in_start_and_end_markers() {
        let spec = MiniMaxM3VisionSpec;
        let meta = metadata();
        let replacements = spec
            .prompt_replacements(&meta, &preprocessed(vec![4]))
            .unwrap();

        assert_eq!(replacements.len(), 1);
        let replacement = &replacements[0];

        // M3's chat template emits a bare ]<]image[>[, so the spec owns the
        // surrounding markers.
        assert_eq!(
            replacement.tokens,
            vec![
                IMAGE_START_ID,
                IMAGE_ID,
                IMAGE_ID,
                IMAGE_ID,
                IMAGE_ID,
                IMAGE_END_ID
            ]
        );
        assert_eq!(replacement.placeholder_token, "]<]image[>[");
        assert_eq!(replacement.modality, Modality::Image);
    }

    #[test]
    fn feature_span_skips_the_structural_markers() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements(&metadata(), &preprocessed(vec![4]))
            .unwrap();
        let ranges = replacements[0].feature_ranges.as_ref().unwrap();

        // The encoder features are the padded middle only.
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].offset, 1);
        assert_eq!(ranges[0].length, 4);
        // Both markers live inside `tokens`, so nothing is folded in from
        // before the placeholder.
        assert_eq!(replacements[0].structural_prefix, 0);
    }

    #[test]
    fn one_replacement_per_media_item() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements(&metadata(), &preprocessed(vec![2, 3]))
            .unwrap();

        assert_eq!(replacements.len(), 2);
        assert_eq!(replacements[0].tokens.len(), 2 + 2);
        assert_eq!(replacements[1].tokens.len(), 3 + 2);
    }

    #[test]
    fn video_replacement_uses_the_video_pad_token() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements_for(&metadata(), &preprocessed(vec![3]), Modality::Video)
            .unwrap();

        // Video uses M3's own video markers, not the image pair.
        assert_eq!(
            replacements[0].tokens,
            vec![VIDEO_START_ID, VIDEO_ID, VIDEO_ID, VIDEO_ID, VIDEO_END_ID]
        );
        assert_eq!(replacements[0].modality, Modality::Video);
        assert_eq!(replacements[0].placeholder_token, "]<]video[>[");
    }

    /// Preprocessed video carrying a `video_grid_thw` of `[grid_t, h, w]`.
    fn preprocessed_video(counts: Vec<usize>, grid_t: i64) -> PreprocessedEncoderInputs {
        preprocessed(counts).with_extra(
            "video_grid_thw",
            ModelSpecificValue::IntTensor {
                data: vec![grid_t, 4, 4],
                shape: vec![1, 3],
            },
        )
    }

    #[tokio::test]
    async fn multi_frame_video_emits_one_block_per_frame() {
        let spec = MiniMaxM3VisionSpec;
        // 3 temporal frames, 12 tokens total => 4 pad tokens per frame.
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![12], 3),
                Modality::Video,
            )
            .unwrap();

        let tokens = &replacements[0].tokens;
        // Each frame is [start] + 4 pads + [end]; vLLM builds the same shape.
        let frame = |_| {
            let mut v = vec![VIDEO_START_ID];
            v.extend(std::iter::repeat_n(VIDEO_ID, 4));
            v.push(VIDEO_END_ID);
            v
        };
        let expected: Vec<TokenId> = (0..3).flat_map(frame).collect();
        assert_eq!(tokens, &expected);
        assert_eq!(tokens.len(), 3 * (4 + 2));
    }

    #[tokio::test]
    async fn multi_frame_feature_ranges_skip_each_frames_markers() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![12], 3),
                Modality::Video,
            )
            .unwrap();

        let ranges = replacements[0].feature_ranges.as_ref().unwrap();
        assert_eq!(ranges.len(), 3);
        // Frame f starts at f*(4+2), its pads begin one token later.
        assert_eq!((ranges[0].offset, ranges[0].length), (1, 4));
        assert_eq!((ranges[1].offset, ranges[1].length), (7, 4));
        assert_eq!((ranges[2].offset, ranges[2].length), (13, 4));
    }

    #[tokio::test]
    async fn single_frame_video_stays_one_block() {
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![4], 1),
                Modality::Video,
            )
            .unwrap();

        assert_eq!(
            replacements[0].tokens,
            vec![
                VIDEO_START_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_END_ID
            ]
        );
    }

    #[tokio::test]
    async fn ragged_token_count_falls_back_to_one_block() {
        let spec = MiniMaxM3VisionSpec;
        // 10 tokens over 3 frames does not divide evenly.
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![10], 3),
                Modality::Video,
            )
            .unwrap();

        assert_eq!(replacements[0].tokens.len(), 10 + 2);
    }

    #[test]
    fn declares_image_and_video_limits() {
        let spec = MiniMaxM3VisionSpec;
        let limits = spec.modality_limits(&metadata()).unwrap();

        assert_eq!(limits.get(&Modality::Image), Some(&MAX_IMAGES_PER_REQUEST));
        assert_eq!(MAX_IMAGES_PER_REQUEST, 200);
        assert_eq!(limits.get(&Modality::Video), Some(&MAX_VIDEOS_PER_REQUEST));
        assert_eq!(MAX_VIDEOS_PER_REQUEST, 20);
        assert!(!limits.contains_key(&Modality::Audio));
    }

    #[test]
    fn audio_is_rejected() {
        let spec = MiniMaxM3VisionSpec;
        let err = spec
            .prompt_replacements_for(&metadata(), &preprocessed(vec![1]), Modality::Audio)
            .unwrap_err();

        assert!(matches!(
            err,
            ModelRegistryError::UnsupportedModality { .. }
        ));
    }

    #[test]
    fn grid_tensors_stay_on_cpu() {
        // vLLM marks both grid tensors keep_on_cpu=True.
        let spec = MiniMaxM3VisionSpec;
        let keys = spec.keep_on_cpu_keys();

        assert!(keys.contains(&"image_grid_thw".to_string()));
        assert!(keys.contains(&"video_grid_thw".to_string()));
    }

    // MiniMaxM3 frame sampling: vLLM MiniMaxM3VideoBackend
    // (models/minimax_m3/common/mm_preprocess.py) parity. The expected values
    // below were cross-checked against the reference implementation.

    fn m3_source(total_frames: usize, fps: f64) -> VideoSourceMeta {
        VideoSourceMeta {
            total_frames,
            original_fps: fps,
            duration_seconds: (fps.is_finite() && fps > 0.0).then_some(total_frames as f64 / fps),
        }
    }

    #[test]
    fn m3_sampling_ceil_interval_walk() {
        // One frame per second of source time at 30 fps, eps-guarded.
        let plan = VideoSamplingStrategy::MiniMaxM3.plan(&m3_source(98, 30.0), 1, 768, 1.0);
        assert_eq!(plan.indices, vec![0, 30, 60, 90]);

        let plan = VideoSamplingStrategy::MiniMaxM3.plan(&m3_source(98, 30.0), 1, 768, 2.0);
        assert_eq!(plan.indices, vec![0, 15, 30, 45, 60, 75, 90]);

        // Non-integer source fps: ceil keeps the walk on distinct frames.
        let plan = VideoSamplingStrategy::MiniMaxM3.plan(&m3_source(97, 33.0), 1, 768, 1.0);
        assert_eq!(plan.indices, vec![0, 33, 66]);
    }

    #[test]
    fn m3_sampling_degenerates_to_first_frame() {
        // Tiny clips and unknown/invalid rates all yield [0], matching the
        // reference guard (no uniform fallback — the model contract).
        for source in [
            m3_source(2, 2.0),
            m3_source(1, 30.0),
            m3_source(100, 0.0),
            m3_source(100, f64::NAN),
        ] {
            let plan = VideoSamplingStrategy::MiniMaxM3.plan(&source, 1, 768, 1.0);
            assert_eq!(plan.indices, vec![0]);
        }
        let plan = VideoSamplingStrategy::MiniMaxM3.plan(&m3_source(0, 30.0), 1, 768, 1.0);
        assert!(plan.indices.is_empty());
    }

    #[test]
    fn m3_video_fetch_config_defaults_to_1fps() {
        let cfg = MiniMaxM3VisionSpec.video_fetch_config(None);
        assert_eq!(cfg.min_frames, 1);
        assert_eq!(cfg.max_frames, 768);
        assert_eq!(cfg.sample_fps, 1.0);
        assert_eq!(cfg.strategy, VideoSamplingStrategy::MiniMaxM3);

        let pp_config = PreProcessorConfig::from_json(r#"{"fps": 2.0, "max_frames": 256}"#)
            .expect("video preprocessor config");
        let cfg = MiniMaxM3VisionSpec.video_fetch_config(Some(&pp_config));
        assert_eq!(cfg.sample_fps, 2.0);
        assert_eq!(cfg.max_frames, 256);
    }

    #[test]
    fn video_frames_carry_timestamp_markers_when_metadata_present() {
        let spec = MiniMaxM3VisionSpec;
        // grid_t = 2, 2 pad tokens per frame; sampled source frames 0 and 30
        // at 30 fps -> timestamps 0.0s and 1.0s (temporal_patch_size = 2
        // clamps the second lookup to the last index).
        let input = preprocessed_video(vec![4], 2)
            .with_extra(
                "video_frames_indices",
                ModelSpecificValue::IntTensor {
                    data: vec![0, 30],
                    shape: vec![2],
                },
            )
            .with_extra(
                "video_source_fps",
                ModelSpecificValue::Tensor {
                    data: vec![30.0],
                    shape: vec![1],
                },
            );
        let replacements = spec
            .prompt_replacements_for(&metadata(), &input, Modality::Video)
            .unwrap();

        let tokens = &replacements[0].tokens;
        // Each frame: "]<]X.X seconds[>[" (17 byte-encoded tokens) + start +
        // 2 pads + end.
        let ts_len = 17;
        let frame_len = ts_len + 1 + 2 + 1;
        assert_eq!(tokens.len(), 2 * frame_len);
        assert_eq!(tokens[ts_len], VIDEO_START_ID);
        assert_eq!(tokens[frame_len + ts_len], VIDEO_START_ID);
        assert_eq!(encoded_text(&tokens[..ts_len]), "]<]0.0 seconds[>[");
        assert_eq!(
            encoded_text(&tokens[frame_len..frame_len + ts_len]),
            "]<]1.0 seconds[>["
        );

        let ranges = replacements[0].feature_ranges.as_ref().unwrap();
        assert_eq!(
            ranges
                .iter()
                .map(|range| (range.offset, range.length))
                .collect::<Vec<_>>(),
            vec![(ts_len + 1, 2), (frame_len + ts_len + 1, 2)]
        );
    }

    #[test]
    fn video_frames_omit_timestamps_without_metadata() {
        // The no-metadata path (e.g. ffmpeg decode) stays vLLM's documented
        // aligned fallback: frame blocks only.
        let spec = MiniMaxM3VisionSpec;
        let replacements = spec
            .prompt_replacements_for(
                &metadata(),
                &preprocessed_video(vec![4], 2),
                Modality::Video,
            )
            .unwrap();

        assert_eq!(
            replacements[0].tokens,
            vec![
                VIDEO_START_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_END_ID,
                VIDEO_START_ID,
                VIDEO_ID,
                VIDEO_ID,
                VIDEO_END_ID,
            ]
        );
        let ranges = replacements[0].feature_ranges.as_ref().unwrap();
        assert_eq!(
            ranges
                .iter()
                .map(|range| (range.offset, range.length))
                .collect::<Vec<_>>(),
            vec![(1, 2), (5, 2)]
        );
    }
}
