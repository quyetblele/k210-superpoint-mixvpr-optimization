#!/usr/bin/env bash
set -euo pipefail

root=/home/quyet/k210_lab
archive_dir="$root/datasets/7scenes_archives"
segment_dir="$archive_dir/.segments_parallel"
base=https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8
scenes=(fire heads office pumpkin redkitchen stairs)
sizes=(2301204154 956332240 4707873861 2890874911 6141181406 1496412431)
mkdir -p "$segment_dir"

download_scene() {
  local scene=$1 expected=$2 archive="$archive_dir/$1.zip"
  local actual=0
  [[ -f "$archive" ]] && actual=$(stat -c %s "$archive")
  if (( actual > expected )); then echo "$scene: oversized archive" >&2; return 2; fi
  if (( actual == expected )); then echo "$scene: already complete"; return 0; fi
  local work="$segment_dir/$scene"; rm -rf "$work"; mkdir -p "$work"
  local remaining=$((expected - actual)) parts=8
  local chunk=$(((remaining + parts - 1) / parts))
  local -a pids=() starts=() ends=() files=()
  local part start end out
  for ((part=0; part<parts; part++)); do
    start=$((actual + part * chunk)); end=$((start + chunk - 1))
    (( start >= expected )) && continue
    (( end >= expected )) && end=$((expected - 1))
    out="$work/part_$part"; starts+=("$start"); ends+=("$end"); files+=("$out")
    curl -L --fail --retry 12 --retry-all-errors --range "$start-$end" \
      --output "$out" "$base/$scene.zip" >"$work/part_$part.log" 2>&1 & pids+=("$!")
  done
  local failed=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 )) || { echo "$scene: transfer failed" >&2; return 3; }
  for ((part=0; part<${#files[@]}; part++)); do
    local want=$((ends[part] - starts[part] + 1)); local got=$(stat -c %s "${files[part]}")
    (( got == want )) || { echo "$scene: part size $got != $want" >&2; return 4; }
  done
  local assembled="$work/assembled.zip"; cp "$archive" "$assembled"
  for out in "${files[@]}"; do cat "$out" >> "$assembled"; done
  local total=$(stat -c %s "$assembled")
  (( total == expected )) || { echo "$scene: assembled size $total != $expected" >&2; return 5; }
  mv "$assembled" "$archive"; echo "$scene: complete ($total bytes)"
}

pids=()
for index in "${!scenes[@]}"; do download_scene "${scenes[$index]}" "${sizes[$index]}" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 6
echo ALL_7SCENES_ARCHIVES_COMPLETE
