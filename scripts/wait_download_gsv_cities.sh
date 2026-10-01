#!/usr/bin/env bash
set -euo pipefail

root=/home/quyet/k210_lab
credential=/home/quyet/.kaggle/kaggle.json
venv="$root/.venv-mixvpr-cuda"
download_dir="$root/datasets/gsv_cities_download"
extract_dir="$root/datasets/gsv_cities"
log_dir="$root/reports/success_first"

while [[ ! -s "$credential" ]]; do
  sleep 60
done

mode=$(stat -c %a "$credential")
if [[ "$mode" != "600" && "$mode" != "400" ]]; then
  echo "BLOCKED: $credential must have mode 600 or 400; actual=$mode"
  exit 2
fi

mkdir -p "$download_dir" "$extract_dir" "$log_dir/gsv_metadata"
# Metadata access is a compact authentication/license check and does not dump
# the dataset's file list into the execution context.
"$venv/bin/kaggle" datasets metadata amaralibey/gsv-cities \
  -p "$log_dir/gsv_metadata"
"$venv/bin/kaggle" datasets download amaralibey/gsv-cities \
  -p "$download_dir" -q

archive="$download_dir/gsv-cities.zip"
if [[ ! -f "$archive" ]]; then
  echo "BLOCKED: Kaggle did not create expected archive $archive"
  exit 3
fi
unzip -t "$archive" >"$log_dir/gsv_zip_test.log"
unzip -q -n "$archive" -d "$extract_dir"
python3 "$root/experiments/edge_vpr_students/build_success_first_manifests.py"
