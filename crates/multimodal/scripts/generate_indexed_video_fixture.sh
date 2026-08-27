#!/usr/bin/env bash
# Regenerate tests/fixtures/videos/indexed_98f_30fps.mp4: 98 frames at 30 fps,
# 64x48, each frame a solid color with R = frame_index * 2 (G=32, B=200),
# encoded with lossless libx264rgb so the decoded R channel identifies the
# source frame exactly. Requires ffmpeg with libx264rgb.
set -euo pipefail

out_dir="$(cd "$(dirname "$0")/.." && pwd)/tests/fixtures/videos"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

python3 - "$work_dir" <<'EOF'
import sys

work_dir = sys.argv[1]
width, height, frames = 64, 48, 98
for index in range(frames):
    row = bytes([index * 2, 32, 200]) * width
    with open(f"{work_dir}/{index:03d}.ppm", "wb") as frame_file:
        frame_file.write(f"P6\n{width} {height}\n255\n".encode())
        frame_file.write(row * height)
EOF

ffmpeg -hide_banner -loglevel error -y \
    -framerate 30 -i "$work_dir/%03d.ppm" \
    -c:v libx264rgb -qp 0 -pix_fmt rgb24 \
    "$out_dir/indexed_98f_30fps.mp4"
