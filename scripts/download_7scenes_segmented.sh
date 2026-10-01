#!/usr/bin/env bash
set -euo pipefail

root=/home/quyet/k210_lab
archive_dir="$root/datasets/7scenes_archives"
segment_dir="$archive_dir/.segments"
base=https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8
scenes=(fire heads office pumpkin redkitchen stairs)
sizes=(2301204154 956332240 4707873861 2890874911 6141181406 1496412431)
mkdir -p "$segment_dir"

for index in "${!scenes[@]}"; do
  scene=${scenes[$index]}; expected=${sizes[$index]}
  archive="$archive_dir/$scene.zip"
  actual=0; [[ -f "$archive" ]] && actual=$(stat -c %s "$archive")
  if (( actual > expected )); then
    echo "$scene: existing archive is larger than expected ($actual > $expected)" >&2
    exit 2
  fi
  if (( actual == expected )); then
    echo "$scene: already complete ($actual bytes)"; continue
  fi
  work="$segment_dir/$scene"
  rm -rf "$work"
  mkdir -p "$work"
  remaining=$((expected - actual)); parts=4
  chunk=$(((remaining + parts - 1) / parts))
  pids=(); starts=(); ends=(); files=()
  for ((part=0; part<parts; part++)); do
    start=$((actual + part * chunk)); end=$((start + chunk - 1))
    (( start >= expected )) && continue
    (( end >= expected )) && end=$((expected - 1))
    out="$work/part_$part"
    starts+=("$start"); ends+=("$end"); files+=("$out")
    curl -L --fail --retry 12 --retry-all-errors --range "$start-$end" \
      --output "$out" "$base/$scene.zip" >"$work/part_$part.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "$scene: one or more range transfers failed" >&2; exit 3; }
  for ((part=0; part<${#files[@]}; part++)); do
    want=$((ends[part] - starts[part] + 1)); got=$(stat -c %s "${files[part]}")
    if (( got != want )); then
      echo "$scene: range $part size mismatch got=$got want=$want" >&2; exit 4
    fi
  done
  assembled="$work/assembled.zip"
  cp "$archive" "$assembled"
  for file in "${files[@]}"; do cat "$file" >> "$assembled"; done
  total=$(stat -c %s "$assembled")
  (( total == expected )) || { echo "$scene: assembled size mismatch $total != $expected" >&2; exit 5; }
  mv "$assembled" "$archive"
  echo "$scene: complete ($total bytes)"
done
echo "ALL_7SCENES_ARCHIVES_COMPLETE"
