#!/usr/bin/env bash
set -euo pipefail

root=/home/quyet/k210_lab
archive_root="$root/datasets/7scenes_archives"
output_root="$root/datasets/7scenes"
scenes=(chess fire heads office pumpkin redkitchen stairs)

mkdir -p "$output_root"
for scene in "${scenes[@]}"; do
  outer="$archive_root/$scene.zip"
  scene_root="$output_root/$scene"
  mkdir -p "$scene_root"
  # Materialize only one scene's nested archives at a time.
  unzip -q -o "$outer" "$scene/seq-*.zip" "$scene/TrainSplit.txt" \
    "$scene/TestSplit.txt" -d "$output_root"
  for nested in "$scene_root"/seq-*.zip; do
    status=0
    unzip -q -n "$nested" '*.color.png' '*.pose.txt' -d "$scene_root" || status=$?
    # Official nested ZIPs return 1 for a malformed directory metadata field;
    # their file entries and CRCs remain extractable. Status >=2 is fatal.
    if [[ "$status" -ge 2 ]]; then exit "$status"; fi
  done
  color_count=$(find "$scene_root" -type f -name '*.color.png' | wc -l)
  pose_count=$(find "$scene_root" -type f -name '*.pose.txt' | wc -l)
  if [[ "$color_count" -eq 0 || "$color_count" -ne "$pose_count" ]]; then
    echo "RGB/pose validation failed for $scene: $color_count/$pose_count" >&2
    exit 2
  fi
  find "$scene_root" -maxdepth 1 -type f -name 'seq-*.zip' -delete
  echo "$scene RGB_POSE_PASS count=$color_count"
done
