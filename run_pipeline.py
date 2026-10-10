import os
import shutil
import sys
import torch
import argparse
import subprocess

# This reproduction repository exposes only the end-to-end joint-training path.
from step_full_joint import UniversalJointGRN

# --- Path roots (set once, derived from this file's location) ---
PROJECT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(PROJECT_DIR, 'data')
DATASETS_DIR   = os.path.join(DATA_DIR, 'datasets')
PRETRAINED_DIR = os.path.join(DATA_DIR, 'pretrained')
CONFIGS_DIR    = os.path.join(DATA_DIR, 'configs')
COMMON_GENES   = os.path.join(PROJECT_DIR, 'common_genes_7sets.txt')

def get_dataset_config(name):
    """
    根据数据集名称返回对应的配置路径字典
    """
    base_dir = DATASETS_DIR

    if name == "PBMC":
        return {
            "name": "PBMC",
            "atac": f"{base_dir}/PBMC/graph/x_atac_rp_score_feature.txt",
            "rna": f"{base_dir}/PBMC/graph/x_scRNA_grnboost2_feature.txt",
            "node": f"{base_dir}/PBMC/graph/x_node.txt",
            "adj": f"{base_dir}/PBMC/graph/x_graph.csv",
            "orig_model": os.path.join(PRETRAINED_DIR, "PBMC_aligned_model.pth"),
            "config": os.path.join(CONFIGS_DIR, f"{name}_config.yaml")
        }
    elif name == "brain":
        return {
            "name": "brain",
            "atac": f"{base_dir}/H_brain/graph/x_atac_rp_score_feature.txt",
            "rna": f"{base_dir}/H_brain/graph/x_scRNA_grnboost2_feature.txt",
            "node": f"{base_dir}/H_brain/graph/x_node.txt",
            "adj": f"{base_dir}/H_brain/graph/x_graph.csv", # 特殊指定的 prior adj
            "orig_model": os.path.join(PRETRAINED_DIR, "brain_aligned_model.pth"),
            "config": os.path.join(CONFIGS_DIR, f"{name}_config.yaml")
        }
    elif name == "brain3k":
        return {
            "name": "brain3k",
            "atac": f"{base_dir}/H_brain3k/graph/x_atac_rp_score_feature.txt",
            "rna": f"{base_dir}/H_brain3k/graph/x_scRNA_grnboost2_feature.txt",
            "node": f"{base_dir}/H_brain3k/graph/x_node.txt",
            "adj": f"{base_dir}/H_brain3k/graph/x_graph.csv",
            "orig_model": os.path.join(PRETRAINED_DIR, "brain3k_aligned_model.pth"),
            "config": os.path.join(CONFIGS_DIR, f"{name}_config.yaml")
        }
    elif name == "Jejunum":
        return {
            "name": "Jejunum",
            "atac": f"{base_dir}/H_jejunum/graph/x_atac_rp_score_feature.txt",
            "rna": f"{base_dir}/H_jejunum/graph/x_scRNA_grnboost2_feature.txt",
            "node": f"{base_dir}/H_jejunum/graph/x_node.txt",
            "adj": f"{base_dir}/H_jejunum/graph/x_graph.csv",
            "orig_model": os.path.join(PRETRAINED_DIR, "Jejunum_aligned_model.pth"),
            "config": os.path.join(CONFIGS_DIR, f"{name}_config.yaml")
        }
    elif name == "lymph":
        return {
            "name": "lymph",
            "atac": f"{base_dir}/H_lymph/graph/x_atac_rp_score_feature.txt",
            "rna": f"{base_dir}/H_lymph/graph/x_scRNA_grnboost2_feature.txt",
            "node": f"{base_dir}/H_lymph/graph/x_node.txt",
            "adj": f"{base_dir}/H_lymph/graph/x_graph.csv",
            "orig_model": os.path.join(PRETRAINED_DIR, "lymph_aligned_model.pth"),
            "config": os.path.join(CONFIGS_DIR, f"{name}_config.yaml")
        }
    elif name == "kidney":
        return {
            "name": "kidney",
            "atac": f"{base_dir}/H_kidney/graph/x_atac_rp_score_feature.txt",
            "rna": f"{base_dir}/H_kidney/graph/x_scRNA_grnboost2_feature.txt",
            "node": f"{base_dir}/H_kidney/graph/x_node.txt",
            "adj": f"{base_dir}/H_kidney/graph/x_graph.csv",
            "orig_model": os.path.join(PRETRAINED_DIR, "Kidney_aligned_model.pth"),
            "config": os.path.join(CONFIGS_DIR, f"{name}_config.yaml")
        }
    else:
        raise ValueError(f"Unknown dataset: {name}")


def main():
    parser = argparse.ArgumentParser(description="TransCoGRN Pipeline Runner")
    parser.add_argument('--target', type=str, required=True, help="目标域数据集名称 (例如 brain, kidney, PBMC)")
    parser.add_argument('--num_sources', type=int, required=True, help="使用的源域数据集数量 (例如 5, 3, 1)")
    parser.add_argument('--joint_full', action='store_true',
                        help="联合优化 encoder、mapper、fusion 和 GNN")
    parser.add_argument('--lambda1', type=float, default=1.0, help="Mapper Loss weight")
    parser.add_argument('--lambda2', type=float, default=1.0, help="Ortho Loss weight")
    parser.add_argument('--lambda3', type=float, default=1.0, help="MMD Loss weight")
    parser.add_argument('--lambda_grn_src', type=float, default=1.0,
                        help="[--joint_full 专用] 源域 GRN 链接预测损失权重")
    parser.add_argument('--lambda_grn_tgt', type=float, default=1.0,
                        help="[--joint_full 专用] 目标域 GRN 链接预测损失权重 (mapper 输出 imputed ATAC)")
    parser.add_argument('--source_list', type=str, default=None,
                        help="可选: 显式指定源域逗号分隔列表 (例如: 'PBMC,brain,lymph')。若提供则覆盖默认顺序选择。")
    parser.add_argument('--epochs_pretrain', type=int, default=120, help="联合训练的 epoch 数")
    parser.add_argument('--lr_pretrain', type=float, default=1e-4, help="联合训练学习率")
    parser.add_argument('--tag', type=str, default='', help="额外标签，用于区分不同试验保存的中间模型权重")
    args = parser.parse_args()

    # 系统支持的所有数据集列表
    all_datasets = ["PBMC", "brain", "brain3k", "Jejunum", "lymph", "kidney"]

    if args.target not in all_datasets:
        raise ValueError(f"Target {args.target} must be one of {all_datasets}")

    # 移除目标域，剩余的作为候选源域
    available_sources = [d for d in all_datasets if d != args.target]

    # ---------------------------------------------------------
    # 源域选择策略
    # ---------------------------------------------------------
    if args.source_list is not None and len(args.source_list.strip()) > 0:
        # 显式指定: 严格按照用户给定列表 (不区分大小写匹配)
        requested = [s.strip() for s in args.source_list.split(',') if s.strip()]
        # 大小写归一映射
        name_lower_to_canonical = {d.lower(): d for d in all_datasets}
        selected_sources = []
        for r in requested:
            key = r.lower()
            if key not in name_lower_to_canonical:
                raise ValueError(f"未知源域名称: {r}。可用项: {all_datasets}")
            canonical = name_lower_to_canonical[key]
            if canonical == args.target:
                print(f"Warning: 源域 {canonical} 与目标域相同，已自动跳过。")
                continue
            if canonical in selected_sources:
                continue
            selected_sources.append(canonical)
        if len(selected_sources) != args.num_sources:
            print(f"Warning: --source_list 指定了 {len(selected_sources)} 个唯一源域 (目标已剔除)，"
                  f"覆盖了 --num_sources={args.num_sources} 的设定。")
            args.num_sources = len(selected_sources)
    else:
        if args.num_sources > len(available_sources):
            print(f"Warning: Requested {args.num_sources} sources but only {len(available_sources)} available. Using {len(available_sources)}.")
            args.num_sources = len(available_sources)
        # 根据用户指定数量选取源域数据（此处默认顺序截取）
        selected_sources = available_sources[:args.num_sources]

    print("=========================================")
    print(f"[Target] Target Domain: {args.target}")
    print(f"[Source] Source Domains ({args.num_sources}): {selected_sources}")
    print("=========================================\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_config = get_dataset_config(args.target)
    source_configs = [get_dataset_config(s) for s in selected_sources]

    # 模型和结果存放路径
    output_dir = os.path.join(PROJECT_DIR, 'result', 'Transfer_MultiSource')
    os.makedirs(output_dir, exist_ok=True)

    # 为了区分不同消融实验的权重，在名字中加入源域数量 + 可选 tag 标记
    suffix = f"{args.num_sources}src" + (f"_{args.tag}" if args.tag else "")
    da_enc_path = f"{output_dir}/{args.target}_{suffix}_Universal_Adapted_Encoder.pth"
    da_mapper_path = f"{output_dir}/{args.target}_{suffix}_Universal_Mapper.pth"

    # The evaluator reads the generic target-specific prediction filename.
    out_file = f"{output_dir}/{args.target}_TransCoGRN_Finetuned.csv"

    if not args.joint_full:
        raise SystemExit(
            "This reproduction requires --joint_full."
        )

    print("\n\n" + "="*40)
    print(" [Step 2-3-4 Full Joint] encoder+mapper+fusion+GNN End-to-End")
    print("="*40)
    save_fusion_path = f"{output_dir}/{args.target}_{suffix}_Joint_Fusion.pth"
    save_gnn_path    = f"{output_dir}/{args.target}_{suffix}_Joint_GNN.pth"
    joint_runner = UniversalJointGRN(target_config['config'], COMMON_GENES, device)
    joint_runner.run(
        source_configs, target_config,
        save_enc_path=da_enc_path,
        save_map_path=da_mapper_path,
        save_fusion_path=save_fusion_path,
        save_gnn_path=save_gnn_path,
        output_pred_csv=out_file,
        lambda_1=args.lambda1, lambda_2=args.lambda2, lambda_3=args.lambda3,
        lambda_grn_src=args.lambda_grn_src, lambda_grn_tgt=args.lambda_grn_tgt,
        epochs=args.epochs_pretrain, lr=args.lr_pretrain,
    )

    # 额外: 同时保存一份带 num_sources/tag 后缀的 "存档版" CSV,
    # 这样多组实验依次跑也不会互相覆盖, 便于后续做 9 组对比。
    archive_out = f"{output_dir}/{args.target}_{suffix}_TransCoGRN_Finetuned.csv"
    try:
        shutil.copyfile(out_file, archive_out)
        print(f"[Archive] 同时保存: {archive_out}")
    except Exception as _e:
        print(f"[Archive] 复制存档失败 (忽略): {_e}")

    # ---------------------------------------------------------
    # [Final Step] Evaluation
    # ---------------------------------------------------------
    print("\n\n" + "="*40)
    print(" [Final Step] Running Final Evaluation")
    print("="*40)
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_DIR, "evaluate_transfer.py"), args.target],
        check=True,
    )

if __name__ == "__main__":
    main()
