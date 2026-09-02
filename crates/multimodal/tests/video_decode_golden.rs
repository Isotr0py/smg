//! Real-video decode golden: verifies the full IO path (ffprobe metadata →
//! sampling strategy → ffmpeg decode) on a checked-in fixture, complementing
//! `video_sampling_golden.rs` which pins the index math against transformers
//! with synthetic metadata.
//!
//! Fixture: `fixtures/videos/indexed_98f_30fps.mp4` — 98 frames at 30 fps,
//! 64x48, each frame a solid color with R = frame_index * 2, encoded with
//! lossless libx264rgb so the decoded R channel identifies the source frame
//! exactly. Regenerate with `scripts/generate_indexed_video_fixture.sh`.
#![allow(clippy::expect_used, clippy::panic, clippy::print_stderr)]

use std::sync::Arc;

use llm_multimodal::{
    MediaConnector, MediaConnectorConfig, MediaSource, VideoFetchConfig, VideoSamplingStrategy,
};

const TOTAL_FRAMES: u32 = 98;
const SOURCE_FPS: f32 = 30.0;
const DURATION_SECONDS: f64 = 98.0 / 30.0;

fn fixture_bytes() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/videos/indexed_98f_30fps.mp4"
    ))
    .expect("read indexed video fixture")
}

fn ffmpeg_available() -> bool {
    std::process::Command::new("ffprobe")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Recover the source frame index from a decoded frame's mean R channel.
fn recovered_index(frame: &image::DynamicImage) -> u32 {
    let rgb = frame.to_rgb8();
    let sum: u64 = rgb.pixels().map(|pixel| u64::from(pixel[0])).sum();
    let mean = sum as f64 / (rgb.width() as u64 * rgb.height() as u64) as f64;
    (mean / 2.0).round() as u32
}

#[tokio::test]
async fn decode_samples_match_strategy_plans_on_real_video() {
    if !ffmpeg_available() {
        eprintln!("ffprobe not available; skipping real-video decode golden");
        return;
    }
    // Pin the ffmpeg backend: the assertions below (e.g. `frame_indices` is
    // None) describe the ffmpeg path, and the backend override is a
    // process-wide OnceLock — in `opencv-video` feature builds `auto` would
    // otherwise pick OpenCV. Safe to mutate here: this is the only test in
    // the binary and it runs before any decode.
    std::env::set_var("SMG_VIDEO_DECODE_BACKEND", "ffmpeg");
    let connector = Arc::new(
        MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
            .expect("media connector"),
    );

    // Qwen3Vl: int(98/30 * 2) = 6 frames (the legacy uniform path would
    // round to 7, so this count proves the strategy actually drove ffmpeg).
    let clip = connector
        .fetch_video(
            MediaSource::InlineBytes(fixture_bytes()),
            VideoFetchConfig {
                strategy: VideoSamplingStrategy::Qwen3Vl,
                ..VideoFetchConfig::default()
            },
        )
        .await
        .expect("decode indexed video with Qwen3Vl strategy");

    let frames = clip.materialized_frames().expect("materialize frames");
    assert_eq!(frames.len(), 6, "Qwen3Vl must sample exactly 6 frames");
    let info = &clip.sample_info;
    assert_eq!(info.total_source_frames, Some(TOTAL_FRAMES));
    assert!(
        (info.source_fps.expect("source fps") - SOURCE_FPS).abs() < 0.01,
        "source fps must come from the container"
    );
    assert!(
        (info.duration_seconds.expect("duration") - DURATION_SECONDS).abs() < 0.01,
        "duration must come from the container"
    );
    assert!(
        (info.sample_fps - 6.0 / DURATION_SECONDS as f32).abs() < 1e-3,
        "effective fps is planned count over duration, got {}",
        info.sample_fps
    );
    // The ffmpeg fps filter guarantees an exact count, not exact indices.
    assert!(info.frame_indices.is_none());

    let recovered: Vec<u32> = frames.iter().map(recovered_index).collect();
    // ffmpeg's fps filter (round=near) samples interval MIDPOINTS, so the
    // selected frames are neither the plan indices nor frame 0 first — e.g.
    // [8, 24, 40, 57, 73, 89] here. Exact index parity is only provided by
    // the opencv decode path; what the ffmpeg path must guarantee is the
    // count, the order, and coverage of the full span.
    let gap = TOTAL_FRAMES / 6; // ~16 frames between samples
    assert!(
        recovered.first().copied().expect("first frame") < gap,
        "sampling must start inside the first interval: {recovered:?}"
    );
    assert!(
        recovered.windows(2).all(|pair| pair[0] < pair[1]),
        "sampled frames must be strictly increasing source frames: {recovered:?}"
    );
    assert!(
        recovered.last().copied().expect("last frame") > TOTAL_FRAMES - gap,
        "sampling must span the video tail: {recovered:?}"
    );

    // Uniform (legacy): round(98/30 * 2) = 7 frames.
    let clip = connector
        .fetch_video(
            MediaSource::InlineBytes(fixture_bytes()),
            VideoFetchConfig::default(),
        )
        .await
        .expect("decode indexed video with Uniform strategy");
    assert_eq!(
        clip.materialized_frames()
            .expect("materialize frames")
            .len(),
        7,
        "Uniform must sample exactly 7 frames"
    );
    assert!(
        (clip.sample_info.sample_fps - 7.0 / DURATION_SECONDS as f32).abs() < 1e-3,
        "Uniform effective fps is planned count over duration"
    );

    // Glm5Next: extract_t = int(98/30 * 2) = 6; the threshold walk stops at
    // 3s and yields exactly 6 frames, no pad/trim. Exercises the strategy
    // end to end through VideoFetchConfig and the ffmpeg plan path.
    let clip = connector
        .fetch_video(
            MediaSource::InlineBytes(fixture_bytes()),
            VideoFetchConfig {
                min_frames: 1,
                max_frames: 2048,
                sample_fps: 2.0,
                strategy: VideoSamplingStrategy::Glm5Next { max_duration: 0.0 },
            },
        )
        .await
        .expect("decode indexed video with Glm5Next strategy");
    assert_eq!(
        clip.materialized_frames()
            .expect("materialize frames")
            .len(),
        6,
        "Glm5Next must sample exactly 6 frames"
    );
    assert!(
        (clip.sample_info.sample_fps - 6.0 / DURATION_SECONDS as f32).abs() < 1e-3,
        "Glm5Next effective fps is planned count over duration"
    );
}
