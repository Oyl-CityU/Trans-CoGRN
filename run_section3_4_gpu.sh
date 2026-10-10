#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
mode="${1:-all}"
log_dir="$project_dir/reproduction_logs/section3_4"

cd "$project_dir"
mkdir -p "$log_dir" result/Transfer_MultiSource
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"$python_bin" -c 'import sys, torch; assert sys.version_info >= (3, 9); assert torch.cuda.is_available(), "需要可用的 CUDA GPU"; print("GPU:", torch.cuda.get_device_name(0))'

run_one() {
  local target="$1"
  local nsrc="$2"
  local sources="$3"
  local lambda_tgt="$4"
  local epochs="$5"
  local tag="FIG2_GPU_${target}${nsrc}"
  "$python_bin" -u run_pipeline.py \
    --target "$target" --num_sources "$nsrc" --joint_full \
    --source_list "$sources" \
    --lambda1 1.0 --lambda2 2.0 --lambda3 0.5 \
    --lambda_grn_src 1.0 --lambda_grn_tgt "$lambda_tgt" \
    --epochs_pretrain "$epochs" --tag "$tag" \
    2>&1 | tee "$log_dir/${tag}.log"
}

run_brain() {
  run_one brain 1 "PBMC" 1.0 200
  run_one brain 2 "PBMC,brain3k" 1.0 200
  run_one brain 3 "PBMC,brain3k,Jejunum" 1.0 200
  run_one brain 4 "PBMC,brain3k,Jejunum,lymph" 1.0 200
  run_one brain 5 "PBMC,brain3k,Jejunum,lymph,kidney" 1.0 200
}

run_pbmc() {
  run_one PBMC 1 "brain" 2.0 200
  run_one PBMC 2 "brain,brain3k" 2.0 200
  run_one PBMC 3 "brain,brain3k,Jejunum" 2.0 200
  run_one PBMC 4 "brain,brain3k,Jejunum,lymph" 2.0 200
  run_one PBMC 5 "brain,brain3k,Jejunum,lymph,kidney" 2.0 200
}

run_kidney() {
  run_one kidney 1 "PBMC" 2.0 400
  run_one kidney 2 "PBMC,brain" 2.0 400
  run_one kidney 3 "PBMC,brain,brain3k" 2.0 400
  run_one kidney 4 "PBMC,brain,brain3k,Jejunum" 2.0 400
  run_one kidney 5 "PBMC,brain,brain3k,Jejunum,lymph" 2.0 400
}

case "$mode" in
  all)
    run_brain
    run_pbmc
    run_kidney
    ;;
  brain)
    run_brain
    ;;
  PBMC|pbmc)
    run_pbmc
    ;;
  kidney)
    run_kidney
    ;;
  *)
    echo "用法：$0 [all|brain|PBMC|kidney]" >&2
    exit 2
    ;;
esac

echo "第 3.4 节复现完成，评估指标已在上方输出并保存至相应结果目录。"
