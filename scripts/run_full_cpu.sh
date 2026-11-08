#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python_bin="$PWD/baseline/.venv/bin/python"
run_dir="runs/full-cpu-control-seed42-01"
if [[ -e "$run_dir" ]]; then
  echo "Output already exists: $run_dir; refusing to overwrite" >&2
  exit 1
fi
echo '[1/4] Full train, one epoch, CPU; validation after the epoch'
"$python_bin" scripts/train_b02.py --config configs/full_cpu_control.json --data data/competition --variant control --allow-large --output "$run_dir"
echo '[2/4] Reload selected weights and evaluate full val'
"$python_bin" scripts/evaluate_b02.py --data data/competition --model "$run_dir/model.pt" --max-feeds 0 --output "$run_dir/validation"
echo '[3/4] Predict full test'
"$python_bin" scripts/predict_b02.py --data data/competition --model "$run_dir/model.pt" --output "$run_dir/test.parquet"
echo '[4/4] Validate against submission template'
"$python_bin" -m neuroforge.cli validate --predictions "$run_dir/test.parquet" --reference template/sample_submission.parquet
echo "[finished] Submission file: $run_dir/test.parquet"
