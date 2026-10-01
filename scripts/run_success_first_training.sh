#!/usr/bin/env bash
set -euo pipefail

root=/home/quyet/k210_lab
py="$root/.venv-mixvpr-cuda/bin/python"
exp="$root/experiments/edge_vpr_students"
data="$root/datasets/success_first"
art="$root/artifacts/success_first"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_gpu() {
  nice -n 10 ionice -c 2 -n 7 "$py" "$@"
}

train_teacher_stage() {
  local stage=$1 database=$2 steps=$3 init=$4 output=$5
  if [[ -f "$output/report.json" ]]; then return; fi
  local resume_args=()
  local init_args=()
  if [[ -f "$output/latest.pt" ]]; then
    resume_args=(--resume "$output/latest.pt")
  elif [[ -n "$init" ]]; then
    init_args=(--init "$init")
  fi
  run_gpu "$exp/train_strong_teacher_gsv.py" --stage "$stage" --database "$database" \
    --steps "$steps" --output "$output" --physical-places 4 --accumulation 8 --views-per-place 4 \
    "${init_args[@]}" "${resume_args[@]}"
}

train_student_stage() {
  local identity=$1 stage=$2 database=$3 bank=$4 steps=$5 init=$6 output=$7
  if [[ -f "$output/report.json" ]]; then return; fi
  local resume_args=()
  local init_args=()
  if [[ -f "$output/latest.pt" ]]; then
    resume_args=(--resume "$output/latest.pt")
  else
    init_args=(--init "$init")
  fi
  run_gpu "$exp/train_success_first_student.py" --identity "$identity" --stage "$stage" \
    --database "$database" --bank "$bank" --steps "$steps" --output "$output" \
    "${init_args[@]}" "${resume_args[@]}"
}

"$py" "$exp/build_gsv_index.py"
"$py" "$exp/build_success_first_manifests.py"
"$py" "$exp/build_indoor_indexes.py"

teacher="$art/strong_teacher"
train_teacher_stage teacher_gsv "$data/gsv_cities.sqlite" 10000 "" "$teacher/gsv"
train_teacher_stage teacher_indoor "$data/7scenes.sqlite" 3000 "$teacher/gsv/best.pt" "$teacher/indoor"
train_teacher_stage teacher_target "$data/project_target.sqlite" 1500 "$teacher/indoor/best.pt" "$teacher/target"

mkdir -p "$teacher/frozen"
"$py" "$exp/freeze_strong_teacher.py" --source "$teacher/target/best.pt" --output "$teacher/frozen" \
  --stages "$teacher/gsv" "$teacher/indoor" "$teacher/target"
frozen_teacher="$teacher/frozen/strong_mixvpr_teacher.pt"

for stage in gsv indoor target; do
  case "$stage" in
    gsv) database="$data/gsv_cities.sqlite" ;;
    indoor) database="$data/7scenes.sqlite" ;;
    target) database="$data/project_target.sqlite" ;;
  esac
  bank="$art/strong_teacher_bank/$stage"
  if [[ ! -f "$bank/manifest.json" ]]; then
    run_gpu "$exp/build_strong_teacher_bank.py" --teacher "$frozen_teacher" \
      --database "$database" --output "$bank"
  fi
done

screen="$art/student_screening"
if [[ ! -f "$screen/screening_report.json" ]]; then
  run_gpu "$exp/screen_success_first_students.py" --steps 1500 \
    --bank "$art/strong_teacher_bank/gsv" --output "$screen" \
    --physical-places 4 --accumulation 8
fi
winner=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "$screen/screening_report.json")

student="$art/student_full/$winner"
train_student_stage "$winner" student_gsv_1 "$data/gsv_cities.sqlite" \
  "$art/strong_teacher_bank/gsv" 5000 "$screen/$winner/best.pt" "$student/gsv_1"

refreshed="$art/student_hard_bank/gsv_1"
if [[ ! -f "$refreshed/manifest.json" ]]; then
  run_gpu "$exp/refresh_student_hard_negatives.py" --identity "$winner" \
    --student "$student/gsv_1/best.pt" --teacher-bank "$art/strong_teacher_bank/gsv" --output "$refreshed"
fi
train_student_stage "$winner" student_gsv_2 "$data/gsv_cities.sqlite" \
  "$refreshed" 5000 "$student/gsv_1/best.pt" "$student/gsv_2"
train_student_stage "$winner" student_indoor "$data/7scenes.sqlite" \
  "$art/strong_teacher_bank/indoor" 3000 "$student/gsv_2/best.pt" "$student/indoor"
train_student_stage "$winner" student_target "$data/project_target.sqlite" \
  "$art/strong_teacher_bank/target" 1500 "$student/indoor/best.pt" "$student/target"

run_gpu "$exp/eval_success_first_gt1.py" --teacher "$frozen_teacher" \
  --student "$student/target/best.pt" --identity "$winner" --output "$art/final_gt1"
