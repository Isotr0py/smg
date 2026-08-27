#!/usr/bin/env python3
"""Generate HuggingFace reference fingerprints for video frame sampling.

The video processor class is constructed locally, so this script does not
download model weights or configuration. Its output is checked into the Rust
integration test as an external correctness oracle for IO-time frame
selection (which source frames get decoded), complementing the preprocessing
golden in qwen_preprocess_fingerprints.json (what happens to the selected
frames).
"""

import json
import struct
from pathlib import Path

from transformers import __version__ as transformers_version
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)
from transformers.video_utils import VideoMetadata

OUTPUT = (
    Path(__file__).parent.parent / "tests" / "fixtures" / "golden" / "video_sampling_fingerprints.json"
)

TOTAL_FRAMES = (1, 2, 3, 7, 98, 493, 1000, 5000, 100000)
SOURCE_FPS = (24.0, 30.0, 60.0)
# (fps, min_frames, max_frames): HF defaults, a slow-fps checkpoint override,
# and a small-budget override that exercises both clamps.
CONFIGS = ((2.0, 4, 768), (1.0, 4, 768), (2.0, 8, 64))


def fingerprint_indices(indices) -> str:
    contiguous = b"".join(struct.pack("<I", int(index)) for index in indices)
    value = 0xCBF29CE484222325
    for byte in contiguous:
        value ^= byte
        value = value * 0x100000001B3 & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def sampling_cases() -> list[dict]:
    results = []
    for fps, min_frames, max_frames in CONFIGS:
        processor = Qwen3VLVideoProcessor(
            fps=fps, min_frames=min_frames, max_frames=max_frames
        )
        for source_fps in SOURCE_FPS:
            for total_frames in TOTAL_FRAMES:
                metadata = VideoMetadata(
                    total_num_frames=total_frames,
                    fps=source_fps,
                    duration=total_frames / source_fps,
                )
                indices = processor.sample_frames(metadata)
                results.append(
                    {
                        "model": "qwen3_vl",
                        "total_frames": total_frames,
                        "source_fps": source_fps,
                        "fps": fps,
                        "min_frames": min_frames,
                        "max_frames": max_frames,
                        "num_sampled": len(indices),
                        "indices_head": indices[:8].tolist(),
                        "indices_tail": indices[-8:].tolist(),
                        "fnv1a_indices_u32le": fingerprint_indices(indices),
                    }
                )
    return results


def main() -> None:
    document = {
        "generator": "generate_video_sampling_golden.py",
        "transformers": transformers_version,
        "cases": sampling_cases(),
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {len(document['cases'])} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
