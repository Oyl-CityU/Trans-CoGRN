import os
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import subprocess

# 导入现有的模块和函数，不修改源代码
# Joint_Pipeline ships only the joint-training mode (--joint_full).
# Staged-training modules (step2_source_pretrain_mmd, step4_target_gnn_adaptation)
# are intentionally omitted; their imports are dropped here.
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


def custom_train_mapper(src_configs, da_model_path, mapper_save_path, device):
    """
    由于原始的 step3 是硬编码数据源，这里将其逻辑封装为函数以支持动态源域列表。
    """
    print("=== 第三阶段: 基于 Universal DA Encoder 训练 Mapper ===")

    z_rna_list = []
    z_atac_list = []

    for d in src_configs:
        if not os.path.exists(d['orig_model']):
            print(f"Warning: Model not found {d['orig_model']}, skipping {d['name']}")
            continue

        z_r, z_a = extract_latent_features(d, da_model_path, d['orig_model'], device)
        z_rna_list.append(z_r)
        z_atac_list.append(z_a)

    if len(z_rna_list) == 0:
        raise RuntimeError("No source models found. Ensure the original pre-trained models exist and paths are correct.")

    z_rna_all = torch.cat(z_rna_list, dim=0)
    z_atac_all = torch.cat(z_atac_list, dim=0)
    print(f"\nTraining Data Shape: {z_rna_all.shape}")

    embed_dim = z_rna_all.shape[1]
    mapper = CrossModalMapper(input_dim=embed_dim, output_dim=embed_dim).to(device)
    optimizer = optim.Adam(mapper.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(z_rna_all, z_atac_all)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print("Start Training Mapper...")
    mapper.train()
    for epoch in range(140):
        total_loss = 0
        for b_rna, b_atac in loader:
            optimizer.zero_grad()
            pred = mapper(b_rna)
            loss = loss_fn(pred, b_atac)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: MSE={total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(mapper_save_path), exist_ok=True)
    torch.save(mapper.state_dict(), mapper_save_path)
    print(f"Mapper 已保存: {mapper_save_path}")


def main():
    parser = argparse.ArgumentParser(description="TransCoGRN Pipeline Runner")
    parser.add_argument('--target', type=str, required=True, help="目标域数据集名称 (例如 brain, kidney, PBMC)")
    parser.add_argument('--num_sources', type=int, required=True, help="使用的源域数据集数量 (例如 5, 3, 1)")
    parser.add_argument('--joint_optim', action='store_true', help="是否使用论文中提到的 Step 2 和 Step 3 联合优化 (端到端训练)")
    parser.add_argument('--joint_full', action='store_true',
                        help="把 Step 2/3/4 全部合成一个端到端联合训练 (encoder+mapper+fusion+GNN 同步更新). "
                             "本开关启用时会跳过原 Step 4 单独微调.")
    parser.add_argument('--lambda1', type=float, default=1.0, help="Mapper Loss weight")
    parser.add_argument('--lambda2', type=float, default=1.0, help="Ortho Loss weight")
    parser.add_argument('--lambda3', type=float, default=1.0, help="MMD Loss weight")
    parser.add_argument('--lambda_grn_src', type=float, default=1.0,
                        help="[--joint_full 专用] 源域 GRN 链接预测损失权重")
    parser.add_argument('--lambda_grn_tgt', type=float, default=1.0,
                        help="[--joint_full 专用] 目标域 GRN 链接预测损失权重 (mapper 输出 imputed ATAC)")
    parser.add_argument('--source_list', type=str, default=None,
                        help="可选: 显式指定源域逗号分隔列表 (例如: 'PBMC,brain,lymph')。若提供则覆盖默认顺序选择。")
    parser.add_argument('--epochs_pretrain', type=int, default=120, help="Step 2/3 预训练的 epoch 数")
    parser.add_argument('--lr_pretrain', type=float, default=1e-4, help="预训练学习率")
    parser.add_argument('--epochs_finetune', type=int, default=250, help="Step 4 微调的 epoch 数")
    parser.add_argument('--lr_finetune', type=float, default=1e-4, help="微调学习率")
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

    # 最终输出的预测图路径，名称保持和 CoGrn_improve.py 中预期的一致
    # 注意: CoGrn_improve.py 期望 {target}_TransCoGRN_Finetuned.csv 格式，因此此处不附加 tag。
    out_file = f"{output_dir}/{args.target}_TransCoGRN_Finetuned.csv"

    # ---------------------------------------------------------
    # 主训练流程: 三种互斥模式
    #   --joint_full  : Step 2/3/4 全部联合 (encoder+mapper+fusion+GNN 同步反传, 推断也在内)
    #   --joint_optim : Step 2/3 联合, Step 4 仍单独微调 (老行为)
    #   (default)     : Step 2 / Step 3 / Step 4 顺序训练 (最老行为)
    # ---------------------------------------------------------
    if not args.joint_full:
        raise SystemExit(
            "Joint_Pipeline ships only the --joint_full mode. "
            "Re-run with --joint_full, or use the original Unified_Pipeline "
            "for staged-training modes."
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
    # 通过 subprocess 直接调用测试脚本
    # Reuse the exact interpreter that launched this pipeline. Falling back to
    # the PATH command "python" can select Python 2 on Linux clusters.
    subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "CoGrn_improve.py"), args.target], check=True)

if __name__ == "__main__":
    main()
