//! Real-video decode golden for the OpenCV backend: unlike the ffmpeg path
//! (exact frame count, approximate indices), the OpenCV grab/read loop decodes
//! exactly the planned frame indices — pin that end to end on the indexed
//! fixture (see `video_decode_golden.rs` for the fixture format).
#![cfg(feature = "opencv-video")]
#![allow(clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use llm_multimodal::{
    video_sampling::VideoSourceMeta, MediaConnector, MediaConnectorConfig, MediaSource,
    VideoFetchConfig, VideoSamplingStrategy,
};

const TOTAL_FRAMES: usize = 98;
const SOURCE_FPS: f64 = 30.0;

fn fixture_bytes() -> Vec<u8> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/videos/indexed_98f_30fps.mp4"
    ))
    .expect("read indexed video fixture")
}

/// Recover the source frame index from a decoded frame's mean R channel.
fn recovered_index(frame: &image::DynamicImage) -> u32 {
    let rgb = frame.to_rgb8();
    let sum: u64 = rgb.pixels().map(|pixel| u64::from(pixel[0])).sum();
    let mean = sum as f64 / (rgb.width() as u64 * rgb.height() as u64) as f64;
    (mean / 2.0).round() as u32
}

fn plan_indices(cfg: &VideoFetchConfig) -> Vec<usize> {
    cfg.strategy
        .plan(
            &VideoSourceMeta {
                total_frames: TOTAL_FRAMES,
                original_fps: SOURCE_FPS,
                duration_seconds: Some(TOTAL_FRAMES as f64 / SOURCE_FPS),
            },
            cfg.min_frames,
            cfg.max_frames,
            cfg.sample_fps,
        )
        .indices
}

#[tokio::test]
async fn opencv_decodes_exactly_the_planned_frame_indices() {
    // The backend override is a process-wide OnceLock; this binary contains
    // only this test, and the env var is set before any decode runs.
    std::env::set_var("SMG_VIDEO_DECODE_BACKEND", "opencv");
    let connector = Arc::new(
        MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
            .expect("media connector"),
    );

    for strategy in [
        VideoSamplingStrategy::Qwen3Vl,
        VideoSamplingStrategy::Uniform,
    ] {
        let cfg = VideoFetchConfig {
            strategy,
            ..VideoFetchConfig::default()
        };
        let planned = plan_indices(&cfg);
        let clip = connector
            .fetch_video(MediaSource::InlineBytes(fixture_bytes()), cfg)
            .await
            .expect("decode indexed video with OpenCV");

        let frames = clip.materialized_frames().expect("materialize frames");
        let recovered: Vec<u32> = frames.iter().map(recovered_index).collect();
        let planned_u32: Vec<u32> = planned.iter().map(|&i| i as u32).collect();
        // The connector reports the exact indices it decoded...
        assert_eq!(
            clip.sample_info.frame_indices.as_deref(),
            Some(planned_u32.as_slice()),
            "{strategy:?} must record the planned indices"
        );
        // ...and the decoded pixels prove those are the frames we got (the
        // fixture is lossless, so the match is exact).
        assert_eq!(
            recovered, planned_u32,
            "{strategy:?} decoded frames must be the planned source frames"
        );
    }
}
