#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
mode="${1:-all}"
log_dir="$project_dir/reproduction_logs/table1"

cd "$project_dir"
mkdir -p "$log_dir" results/table1/runs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"$python_bin" -c 'import sys, torch; assert sys.version_info >= (3, 9); assert torch.cuda.is_available(), "CUDA GPU is required"; print("GPU:", torch.cuda.get_device_name(0))'

run_one() {
  local dataset="$1"
  "$python_bin" -u table1/CoGrn_improve.py "$dataset" --config_dir table1/config \
    2>&1 | tee "$log_dir/${dataset}.log"
}

case "$mode" in
  all)
    run_one brain
    run_one kidney
    run_one PBMC
    ;;
  brain|kidney|PBMC)
    run_one "$mode"
    ;;
  pbmc)
    run_one PBMC
    ;;
  *)
    echo "Usage: $0 [all|brain|kidney|PBMC]" >&2
    exit 2
    ;;
esac
