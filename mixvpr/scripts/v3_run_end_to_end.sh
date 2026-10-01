#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/quyet/k210_lab
CFG=$ROOT/mixvpr/configs/v3_random_large_pruning.json
RUN=$ROOT/mixvpr/artifacts/v3_random_large_pruning_final
LOG=$RUN/end_to_end.log
mkdir -p "$RUN"
export MIXVPR_CONFIG="$CFG"
export MIXVPR_RUN_ROOT="$RUN"
exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date -Is) ==="
"$ROOT/mixvpr/run" check
"$ROOT/mixvpr/run" kd-diagnostic --batches 8
"$ROOT/mixvpr/run" train 00_large --stop-after-local-step 6000
for s in 01_w5_208 02_w5_192 03_w5_176 04_w5_160 05_w4_144 06_w4_128 07_w4_120 08_w4_112 09_w3_72 10_w3_64 11_h_144 12_h_128 13_h_112 14_h_96; do
  echo "=== TRAIN $s $(date -Is) ==="
  "$ROOT/mixvpr/run" train "$s" --stop-after-local-step 1000
done
echo "=== TRAINING COMPLETE $(date -Is) ==="
"$ROOT/mixvpr/run" status
echo "=== END $(date -Is) ==="
