use std::{collections::HashMap, sync::Arc};

use tokio::task::JoinHandle;

use super::{
    error::{MultiModalError, MultiModalResult},
    media::{ImageFetchConfig, MediaConnector, MediaSource, VideoFetchConfig},
    types::{
        ImageDetail, MediaContentPart, Modality, MultiModalData, MultiModalUUIDs, TrackedMedia,
    },
};

type PendingTask = JoinHandle<MultiModalResult<TrackedMedia>>;

#[derive(Debug)]
pub struct TrackerOutput {
    pub data: MultiModalData,
    pub uuids: MultiModalUUIDs,
}

pub struct AsyncMultiModalTracker {
    media_connector: Arc<MediaConnector>,
    video_fetch_config: VideoFetchConfig,
    pending: HashMap<Modality, Vec<PendingTask>>,
    uuids: MultiModalUUIDs,
}

impl AsyncMultiModalTracker {
    pub fn new(media_connector: Arc<MediaConnector>) -> Self {
        Self {
            media_connector,
            video_fetch_config: VideoFetchConfig::default(),
            pending: HashMap::new(),
            uuids: HashMap::new(),
        }
    }

    /// Override the video fetch/sampling configuration (e.g. from the model's
    /// `ModelProcessorSpec::video_fetch_config`). Defaults to
    /// [`VideoFetchConfig::default`], the legacy uniform sampling.
    pub fn with_video_fetch_config(mut self, video_fetch_config: VideoFetchConfig) -> Self {
        self.video_fetch_config = video_fetch_config;
        self
    }

    pub fn push_part(&mut self, part: MediaContentPart) -> MultiModalResult<()> {
        match part {
            MediaContentPart::Text { .. } => {}
            MediaContentPart::ImageUrl { url, detail, uuid } => {
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
                self.enqueue_image(source, detail.unwrap_or_default(), uuid);
            }
            MediaContentPart::ImageData {
                data,
                mime_type: _,
                uuid,
                detail,
            } => {
                self.enqueue_image(
                    MediaSource::InlineBytes(data),
                    detail.unwrap_or_default(),
                    uuid,
                );
            }
            MediaContentPart::ImageEmbeds { .. } => {
                return Err(MultiModalError::UnsupportedContent("image_embeds"));
            }
            MediaContentPart::AudioUrl { url, uuid } => {
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
                self.enqueue_audio(source, uuid);
            }
            MediaContentPart::AudioData {
                data,
                mime_type: _,
                uuid,
            } => {
                self.enqueue_audio(MediaSource::InlineBytes(data), uuid);
            }
            MediaContentPart::VideoUrl { url, uuid } => {
                let source = match url::Url::parse(&url) {
                    Ok(parsed) if parsed.scheme() == "data" => MediaSource::DataUrl(url),
                    _ => MediaSource::Url(url),
                };
                self.enqueue_video(source, uuid);
            }
            MediaContentPart::VideoData {
                data,
                mime_type: _,
                uuid,
            } => {
                self.enqueue_video(MediaSource::InlineBytes(data), uuid);
            }
        }
        Ok(())
    }

    pub async fn finalize(mut self) -> MultiModalResult<TrackerOutput> {
        let mut data = MultiModalData::new();
        for (modality, tasks) in self.pending.drain() {
            let mut items = Vec::with_capacity(tasks.len());
            for task in tasks {
                let media = task.await??;
                items.push(media);
            }
            data.insert(modality, items);
        }

        Ok(TrackerOutput {
            data,
            uuids: self.uuids,
        })
    }

    fn enqueue_image(&mut self, source: MediaSource, detail: ImageDetail, uuid: Option<String>) {
        let modality = Modality::Image;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let frame = connector
                .fetch_image(source, ImageFetchConfig { detail })
                .await?;
            Ok(TrackedMedia::Image(frame))
        });

        self.pending.entry(modality).or_default().push(handle);
    }

    fn enqueue_video(&mut self, source: MediaSource, uuid: Option<String>) {
        let modality = Modality::Video;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        let video_fetch_config = self.video_fetch_config;
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let clip = connector.fetch_video(source, video_fetch_config).await?;
            Ok(TrackedMedia::Video(clip))
        });

        self.pending.entry(modality).or_default().push(handle);
    }

    fn enqueue_audio(&mut self, source: MediaSource, uuid: Option<String>) {
        let modality = Modality::Audio;
        self.uuids.entry(modality).or_default().push(uuid);

        let connector = Arc::clone(&self.media_connector);
        #[expect(
            clippy::disallowed_methods,
            reason = "spawn handle is stored in self.pending and awaited in finalize(); fire-and-forget is intentional for concurrent media fetching"
        )]
        let handle = tokio::spawn(async move {
            let clip = connector.fetch_audio(source).await?;
            Ok(TrackedMedia::Audio(clip))
        });

        self.pending.entry(modality).or_default().push(handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{media::MediaConnectorConfig, video_sampling::VideoSamplingStrategy};

    fn test_connector() -> Arc<MediaConnector> {
        Arc::new(
            MediaConnector::new(reqwest::Client::new(), MediaConnectorConfig::default())
                .expect("test media connector"),
        )
    }

    #[test]
    fn default_video_fetch_config_is_legacy_uniform_sampling() {
        let tracker = AsyncMultiModalTracker::new(test_connector());
        let cfg = tracker.video_fetch_config;
        assert_eq!(cfg.min_frames, 4);
        assert_eq!(cfg.max_frames, 768);
        assert_eq!(cfg.sample_fps, 2.0);
        assert_eq!(cfg.strategy, VideoSamplingStrategy::Uniform);
    }

    #[test]
    fn with_video_fetch_config_overrides_sampling() {
        let tracker = AsyncMultiModalTracker::new(test_connector()).with_video_fetch_config(
            VideoFetchConfig {
                min_frames: 8,
                max_frames: 32,
                sample_fps: 1.0,
                strategy: VideoSamplingStrategy::Qwen3Vl,
            },
        );
        let cfg = tracker.video_fetch_config;
        assert_eq!(cfg.min_frames, 8);
        assert_eq!(cfg.max_frames, 32);
        assert_eq!(cfg.sample_fps, 1.0);
        assert_eq!(cfg.strategy, VideoSamplingStrategy::Qwen3Vl);
    }
}
