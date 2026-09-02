//! HuggingFace golden checks for IO-time video frame sampling.
//!
//! The fixture is produced by `scripts/generate_video_sampling_golden.py`
//! using the real `Qwen3VLVideoProcessor.sample_frames` and
//! `Glm5NextVideoProcessor.sample_frames`; it pins SMG's sampling strategies
//! to HF's exact frame indices (which source frames get decoded),
//! complementing `qwen_preprocess_golden.rs` (what happens to the selected
//! frames).
#![allow(clippy::expect_used, clippy::panic)]

use llm_multimodal::video_sampling::{VideoSamplingStrategy, VideoSourceMeta};
use serde::Deserialize;

#[derive(Deserialize)]
struct GoldenDocument {
    generator: String,
    transformers: String,
    cases: Vec<SamplingCase>,
}

#[derive(Deserialize)]
struct SamplingCase {
    model: String,
    total_frames: usize,
    source_fps: f64,
    fps: f32,
    min_frames: usize,
    max_frames: usize,
    max_duration: f64,
    has_duration: bool,
    num_sampled: usize,
    indices_head: Vec<u32>,
    indices_tail: Vec<u32>,
    fnv1a_indices_u32le: String,
}

fn fnv1a_indices_u32le(indices: &[usize]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for index in indices {
        for byte in u32::try_from(*index)
            .expect("frame index fits u32")
            .to_le_bytes()
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[test]
fn video_sampling_matches_huggingface_golden() {
    let golden: GoldenDocument = serde_json::from_str(include_str!(
        "fixtures/golden/video_sampling_fingerprints.json"
    ))
    .expect("invalid checked-in video sampling golden fixture");
    assert_eq!(golden.generator, "generate_video_sampling_golden.py");
    assert!(!golden.transformers.is_empty());
    assert_eq!(
        golden.cases.len(),
        164,
        "video sampling golden coverage changed"
    );

    for case in &golden.cases {
        let strategy = match case.model.as_str() {
            "qwen3_vl" => VideoSamplingStrategy::Qwen3Vl,
            "glm5_next" => VideoSamplingStrategy::Glm5Next {
                max_duration: case.max_duration,
            },
            model => panic!("unknown video sampling golden model {model}"),
        };
        let source = VideoSourceMeta {
            total_frames: case.total_frames,
            original_fps: case.source_fps,
            duration_seconds: case
                .has_duration
                .then_some(case.total_frames as f64 / case.source_fps),
        };
        let indices = strategy
            .plan(&source, case.min_frames, case.max_frames, case.fps)
            .indices;

        let label = format!(
            "{} total={} src_fps={} fps={} min={} max={} max_duration={} has_duration={}",
            case.model,
            case.total_frames,
            case.source_fps,
            case.fps,
            case.min_frames,
            case.max_frames,
            case.max_duration,
            case.has_duration
        );
        assert_eq!(
            indices.len(),
            case.num_sampled,
            "sampled count differs: {label}"
        );
        let head: Vec<u32> = indices.iter().take(8).map(|&i| i as u32).collect();
        let tail: Vec<u32> = indices
            .iter()
            .rev()
            .take(8)
            .rev()
            .map(|&i| i as u32)
            .collect();
        assert_eq!(head, case.indices_head, "head indices differ: {label}");
        assert_eq!(tail, case.indices_tail, "tail indices differ: {label}");

        let expected = u64::from_str_radix(&case.fnv1a_indices_u32le, 16)
            .expect("invalid golden FNV-1a fingerprint");
        assert_eq!(
            fnv1a_indices_u32le(&indices),
            expected,
            "sampled frame indices differ from HuggingFace: {label}; got {indices:?}"
        );
    }
}
