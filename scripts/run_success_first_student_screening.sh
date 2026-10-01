#!/usr/bin/env bash
# Resume-safe handoff: frozen teacher bank -> two controlled student screens.
set -euo pipefail

root=/home/quyet/k210_lab
py="$root/.venv-mixvpr-cuda/bin/python"
bank="$root/artifacts/success_first/strong_teacher_bank/gsv_base"
script="$root/experiments/edge_vpr_students/screen_success_first_students.py"
log="$root/reports/success_first/student_screening_per_image_bf16_v2.log"

while [[ ! -f "$bank/manifest.json" ]]; do
  sleep 30
done

# A real, hard-negative smoke comes first.  This uses a separate namespace and
# cannot contaminate the 1,500-step controlled comparison.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nice -n 10 ionice -c 2 -n 7 "$py" "$script" \
  --steps 1 --bank "$bank" \
  --output "$root/artifacts/success_first/student_screening_per_image_bf16_v2_smoke" \
  --physical-places 4 --accumulation 8 >> "$log" 2>&1

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nice -n 10 ionice -c 2 -n 7 "$py" "$script" \
  --steps 1500 --bank "$bank" \
  --output "$root/artifacts/success_first/student_screening_per_image_bf16_v2" \
  --physical-places 4 --accumulation 8 >> "$log" 2>&1
