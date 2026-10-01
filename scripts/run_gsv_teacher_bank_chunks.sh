#!/usr/bin/env bash
set -euo pipefail

root=/home/quyet/k210_lab
py="$root/.venv-mixvpr-cuda/bin/python"
script="$root/experiments/edge_vpr_students/build_strong_teacher_bank.py"
teacher="$root/artifacts/success_first/strong_teacher/gsv/best.pt"
database="$root/datasets/success_first/gsv_cities.sqlite"
output="$root/artifacts/success_first/strong_teacher_bank/gsv_base"
log="$root/reports/success_first/gsv_base_bank.log"

while true; do
  if [[ -f "$output/manifest.json" ]]; then
    printf 'BANK_COMPLETE\n' | tee -a "$log"
    exit 0
  fi
  free_kib=$(df --output=avail /mnt/c | tail -1)
  if (( free_kib < 4194304 )); then
    printf 'STOP_LOW_C_SPACE avail_kib=%s\n' "$free_kib" | tee -a "$log"
    exit 2
  fi
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nice -n 10 ionice -c 2 -n 7 "$py" "$script" \
    --teacher "$teacher" --database "$database" --output "$output" \
    --batch-size 64 --loader-workers 8 --checkpoint-every-images 1024 --images-per-place 1 \
    --max-images-this-run 1024 >> "$log" 2>&1
done
