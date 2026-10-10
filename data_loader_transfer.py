# --- START OF FILE src/data_loader_transfer.py ---
import pandas as pd
import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_aligned_data(atac_path, rna_path, node_path, common_genes_file, device):
    """
    严格按照 common_genes_file 对齐数据，并进行标准化。
    """
    print(f"--- Loading Data from: {os.path.dirname(rna_path)} ---")

    # 1. 读取节点列表 (Rows)
    with open(node_path, 'r') as f:
        nodes = [line.strip().upper() for line in f]

    # 2. 读取共有基因列表 (Columns)
    if not os.path.exists(common_genes_file):
        raise FileNotFoundError(f"Common genes file not found: {common_genes_file}")

    with open(common_genes_file, 'r') as f:
        common_genes = [line.strip() for line in f]
    print(f"  - Common Genes Target: {len(common_genes)}")

    # 3. 读取 RNA 矩阵
    # 假设输入是 (Nodes x Genes) 或 (Nodes x Features)
    rna_df = pd.read_csv(rna_path, sep='\t', index_col=0)
    rna_df.index = rna_df.index.str.upper()

    # [关键步骤] 强制对齐列 (Features)
    # 不存在的基因填 0，多余的基因丢弃
    rna_df = rna_df.reindex(columns=common_genes, fill_value=0.0)

    # [关键步骤] 强制对齐行 (Nodes)
    rna_df = rna_df.reindex(index=nodes).fillna(0.0)

    print(f"  - RNA Aligned Shape: {rna_df.shape}")

    # 4. 读取 ATAC 矩阵 (如果路径存在)
    # 对于迁移学习的目标域，ATAC 可能不存在或不使用，但为了代码兼容性，我们尝试读取
    if atac_path and os.path.exists(atac_path):
        atac_df = pd.read_csv(atac_path, sep='\t', index_col=0)
        atac_df.index = atac_df.index.str.upper()
        atac_df = atac_df.reindex(index=nodes).fillna(0.0)

        # ATAC 特征如果是 RP Score (Genes)，也建议对齐到 common_genes
        # 如果 ATAC 特征是 Cells，则无法对齐，保持原样即可
        # 这里假设 ATAC 特征是 Cells (根据你之前的日志: 10970, 22772)
        # 所以我们只对齐行 (Nodes)
        print(f"  - ATAC Aligned Shape: {atac_df.shape} (Features retained as original)")

        # 5. 标准化 (Z-Score)
        # 这一点非常重要！迁移学习对数据的 Scale 非常敏感
        scaler_atac = StandardScaler()
        atac_feat = torch.tensor(scaler_atac.fit_transform(atac_df), dtype=torch.float32).to(device)
    else:
        print("  - ATAC path not provided or not found. Returning dummy.")
        atac_feat = None

    scaler_rna = StandardScaler()
    rna_feat = torch.tensor(scaler_rna.fit_transform(rna_df), dtype=torch.float32).to(device)

    return atac_feat, rna_feat, nodes