//! HuggingFace golden checks for IO-time video frame sampling.
//!
//! The fixture is produced by `scripts/generate_video_sampling_golden.py`
//! using the real `Qwen3VLVideoProcessor.sample_frames`; it pins SMG's
//! sampling strategy to HF's exact frame indices (which source frames get
//! decoded), complementing `qwen_preprocess_golden.rs` (what happens to the
//! selected frames).
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
fn qwen3_vl_sampling_matches_huggingface_golden() {
    let golden: GoldenDocument = serde_json::from_str(include_str!(
        "fixtures/golden/video_sampling_fingerprints.json"
    ))
    .expect("invalid checked-in video sampling golden fixture");
    assert_eq!(golden.generator, "generate_video_sampling_golden.py");
    assert!(!golden.transformers.is_empty());
    assert_eq!(
        golden.cases.len(),
        81,
        "video sampling golden coverage changed"
    );

    for case in &golden.cases {
        assert_eq!(case.model, "qwen3_vl");
        let source = VideoSourceMeta {
            total_frames: case.total_frames,
            original_fps: case.source_fps,
            duration_seconds: Some(case.total_frames as f64 / case.source_fps),
        };
        let indices = VideoSamplingStrategy::Qwen3Vl
            .plan(&source, case.min_frames, case.max_frames, case.fps)
            .indices;

        let label = format!(
            "total={} src_fps={} fps={} min={} max={}",
            case.total_frames, case.source_fps, case.fps, case.min_frames, case.max_frames
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
