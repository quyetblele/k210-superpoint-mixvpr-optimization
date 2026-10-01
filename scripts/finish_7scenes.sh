#!/usr/bin/env bash
set -euo pipefail

root=/home/quyet/k210_lab
archive_dir="$root/datasets/7scenes_archives"
extract_dir="$root/datasets/7scenes"
base=https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8
scenes=(chess fire heads office pumpkin redkitchen stairs)
sizes=(3079608937 2301204154 956332240 4707873861 2890874911 6141181406 1496412431)

# Do not race the already-running resumable transfers from the previous gate.
while pgrep -f 'curl -L --continue-at.*7scenes_archives' >/dev/null; do
  sleep 30
done

# Recover any transfer that ended early. Files already at exact size are never
# touched. Sequential recovery avoids two writers targeting one archive.
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  expected=${sizes[$index]}
  archive="$archive_dir/$scene.zip"
  actual=0
  if [[ -f "$archive" ]]; then
    actual=$(stat -c %s "$archive")
  fi
  if [[ "$actual" -ne "$expected" ]]; then
    curl -L --fail --retry 12 --retry-all-errors --continue-at - \
      --output "$archive" "$base/$scene.zip"
  fi
done

mkdir -p "$extract_dir"
for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}
  archive="$archive_dir/$scene.zip"
  unzip -t "$archive" >/dev/null
  unzip -q -n "$archive" -d "$extract_dir"
  for sequence_archive in "$extract_dir/$scene"/seq-*.zip; do
    # Microsoft's official nested archives contain a malformed directory-entry
    # size field. unzip returns warning status 1 although all file CRCs are OK.
    # The immutable outer archive is the artifact validated with unzip -t.
    nested_status=0
    # Training needs RGB and pose supervision only; avoid extracting depth maps.
    unzip -q -n "$sequence_archive" '*.color.png' '*.pose.txt' \
      -d "$extract_dir/$scene" || nested_status=$?
    if [[ "$nested_status" -gt 1 ]]; then
      exit "$nested_status"
    fi
  done
  color_count=$(find "$extract_dir/$scene" -type f -name '*.color.png' | wc -l)
  pose_count=$(find "$extract_dir/$scene" -type f -name '*.pose.txt' | wc -l)
  if [[ "$color_count" -eq 0 || "$color_count" -ne "$pose_count" ]]; then
    echo "invalid extracted RGB/pose set for $scene: colors=$color_count poses=$pose_count" >&2
    exit 2
  fi
done

python3 "$root/experiments/edge_vpr_students/build_success_first_manifests.py"
