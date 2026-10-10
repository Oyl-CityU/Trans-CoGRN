# --- START OF FILE CoGrn.py (Disentangled Version) ---

import torch
import argparse
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import precision_recall_curve, auc
import scipy.io as sio
import random
import datetime
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling
from arboreto.algo import genie3, grnboost2
from collections import defaultdict

# --- 全局随机种子 ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 数据加载函数 ---
def load_mtx(data_dir, use_variable_feature=False, additional_genes=None):
    sparse_matrix = sio.mmread(os.path.join(data_dir, "matrix.mtx"))
    cell_names_full = pd.read_csv(os.path.join(data_dir, 'barcodes.tsv'), sep='\t', names=['barcode'])['barcode'].values
    try:
        gene_names_full = pd.read_csv(os.path.join(data_dir, 'genes.tsv'), sep='\t', names=['id', 'gene'])['gene'].values
    except (FileNotFoundError, IndexError):
        gene_names_full = pd.read_csv(os.path.join(data_dir, 'peaks.tsv'), sep='\t', names=['gene'])['gene'].values
    n_features, n_cells = sparse_matrix.shape
    matched_gene_names = gene_names_full[:n_features]
    matched_cell_names = cell_names_full[:n_cells]
    ans = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix, index=matched_gene_names, columns=matched_cell_names
    )
    if use_variable_feature and os.path.exists(os.path.join(data_dir, 'var_features.tsv')):
        index = set(pd.read_csv(os.path.join(data_dir, 'var_features.tsv'), names=['gene'])['gene'].values)
        if additional_genes is not None:
            index = index.union(additional_genes)
        index = list(index.intersection(ans.index))
        ans = ans.loc[index]
    return ans

def load_features(atac_path: str, rna_path: str, node_path: str, common_genes_file: str = None):
    with open(node_path, 'r') as f:
        nodes = [line.strip().upper() for line in f]

    atac_df = pd.read_csv(atac_path, sep='\t', index_col=0)
    atac_df.index = atac_df.index.str.upper()
    atac_df = atac_df.reindex(nodes).fillna(0.0)

    rna_df = pd.read_csv(rna_path, sep='\t', index_col=0)
    rna_df.index = rna_df.index.str.upper()

    # 共有基因过滤逻辑 (用于迁移学习)
    if common_genes_file and os.path.exists(common_genes_file):
        print(f"正在对齐 RNA 特征到共有基因集: {common_genes_file}")
        with open(common_genes_file, 'r') as f:
            common_genes = [line.strip() for line in f]
        rna_df = rna_df.reindex(columns=common_genes, fill_value=0.0)
        print(f"  -> RNA 特征对齐完成，维度: {rna_df.shape}")

    rna_df = rna_df.reindex(nodes).fillna(0.0)

    atac_scaler = StandardScaler()
    atac_features = torch.tensor(atac_scaler.fit_transform(atac_df), dtype=torch.float32)
    print("ATAC 特征形状:", atac_features.shape)

    rna_scaler = StandardScaler()
    rna_features = torch.tensor(rna_scaler.fit_transform(rna_df), dtype=torch.float32)
    print("RNA 特征形状:", rna_features.shape)

    return atac_features, rna_features, nodes

# --- 模型定义 ---

class ATAC_Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(ATAC_Encoder, self).__init__()
        self.feature_engine = nn.Sequential(nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4))
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.2, batch_first=True)
        self.projector = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
                                       nn.Linear(256, embed_dim))
        self.residual = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        x = self.feature_engine(x).unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.projector(attn_out.squeeze(1))
        return x + identity

class RNA_Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super(RNA_Encoder, self).__init__()
        # 公共特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4)
        )
        self.res_blocks = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.LayerNorm(512)
        )

        # [修改] 双头输出: Shared & Private
        self.proj_shared = nn.Linear(512, embed_dim)
        self.proj_private = nn.Linear(512, embed_dim)

        # Shortcut 连接
        if input_dim != embed_dim:
            self.shortcut_shared = nn.Linear(input_dim, embed_dim)
            self.shortcut_private = nn.Linear(input_dim, embed_dim)
        else:
            self.shortcut_shared = nn.Identity()
            self.shortcut_private = nn.Identity()

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = self.res_blocks(feat)

        # 分解特征
        z_shared = self.proj_shared(feat) + self.shortcut_shared(x)
        z_private = self.proj_private(feat) + self.shortcut_private(x)

        return z_shared, z_private

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.head(x)

class ContrastiveEncoders(nn.Module):
    def __init__(self, atac_input_dim, rna_input_dim, embed_dim=128):
        super().__init__()
        self.atac_encoder = ATAC_Encoder(input_dim=atac_input_dim, embed_dim=embed_dim)
        self.rna_encoder = RNA_Encoder(input_dim=rna_input_dim, embed_dim=embed_dim)

        # 投影头
        self.atac_projection = ProjectionHead(input_dim=embed_dim, output_dim=embed_dim)
        self.rna_projection = ProjectionHead(input_dim=embed_dim, output_dim=embed_dim)

    def forward(self, atac_x, rna_x):
        atac_emb = self.atac_encoder(atac_x)

        # RNA 解耦
        z_shared, z_private = self.rna_encoder(rna_x)

        # 对比学习只针对 Shared 部分
        p_atac = self.atac_projection(atac_emb)
        p_rna_shared = self.rna_projection(z_shared)

        return (
            F.normalize(p_atac, dim=-1),
            F.normalize(p_rna_shared, dim=-1),
            z_shared,
            z_private,
            atac_emb
        )

def info_nce_loss(features_1, features_2, temperature=0.1):
    similarity_matrix = torch.matmul(features_1, features_2.T) / temperature
    labels = torch.arange(similarity_matrix.shape[0]).long().to(similarity_matrix.device)
    loss_1_vs_2 = F.cross_entropy(similarity_matrix, labels)
    loss_2_vs_1 = F.cross_entropy(similarity_matrix.T, labels)
    return (loss_1_vs_2 + loss_2_vs_1) / 2

# [新增] 正交损失
def orthogonality_loss(x, y):
    cosine = F.cosine_similarity(x, y, dim=-1)
    return torch.mean(torch.abs(cosine))

class CoGRNFusion(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, dropout_rate=0.3, fused_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_atac_to_rna = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn_rna_to_atac = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, fused_dim)
        )
        self.norm3 = nn.LayerNorm(fused_dim)

    def forward(self, atac_emb, rna_emb):
        atac_q = atac_emb.unsqueeze(1)
        rna_kv = rna_emb.unsqueeze(1)
        attn_output_a, _ = self.attn_atac_to_rna(query=atac_q, key=rna_kv, value=rna_kv)
        fused_atac = self.norm1(atac_emb + attn_output_a.squeeze(1))
        rna_q = rna_emb.unsqueeze(1)
        atac_kv = atac_emb.unsqueeze(1)
        attn_output_r, _ = self.attn_rna_to_atac(query=rna_q, key=atac_kv, value=atac_kv)
        fused_rna = self.norm2(rna_emb + attn_output_r.squeeze(1))
        diff_emb = fused_atac - fused_rna
        prod_emb = fused_atac * fused_rna
        combined_features = torch.cat([fused_atac, fused_rna, diff_emb, prod_emb], dim=1)
        output = self.ffn(combined_features)
        output = self.norm3(output)
        return output

class ConcatFusion(nn.Module):
    def __init__(self, atac_dim, rna_dim, fused_dim=256):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(atac_dim + rna_dim, fused_dim * 2), nn.LayerNorm(fused_dim * 2),
                                     nn.GELU(), nn.Dropout(0.4), nn.Linear(fused_dim * 2, fused_dim))

    def forward(self, atac_emb, rna_emb):
        concatenated_emb = torch.cat([atac_emb, rna_emb], dim=1)
        return self.project(concatenated_emb)

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gnn_layer_type='GAT', num_heads=4, dropout_rate=0.3):
        super().__init__()
        self.gnn_layer_type = gnn_layer_type.upper()
        self.gnn_encoder = nn.ModuleList()
        if self.gnn_layer_type == 'GAT':
            self.gnn_encoder.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout_rate))
            self.gnn_encoder.append(GATv2Conv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout_rate))
        else:
            ConvLayer = globals().get(f"{self.gnn_layer_type}Conv")
            if ConvLayer is None: raise ValueError(f"不支持的 GNN 层类型: {gnn_layer_type}")
            self.gnn_encoder.append(ConvLayer(in_channels, hidden_channels))
            self.gnn_encoder.append(ConvLayer(hidden_channels, out_channels))
        self.decoder = nn.Sequential(nn.Linear(out_channels * 2, hidden_channels), nn.GELU(), nn.Dropout(dropout_rate),
                                     nn.Linear(hidden_channels, 1))

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.gnn_encoder):
            x = conv(x, edge_index)
            if i < len(self.gnn_encoder) - 1: x = F.gelu(F.dropout(x, p=0.4, training=self.training))
        return x

    def decode(self, z, edge_label_index):
        return self.decoder(torch.cat([z[edge_label_index[0]], z[edge_label_index[1]]], dim=-1)).squeeze()

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return self.decode(z, data.edge_label_index)

def load_adjacency_matrix(adj_path, nodes):
    adj_df = pd.read_csv(adj_path, index_col=0)
    adj_df.index = adj_df.index.str.upper()
    adj_df.columns = adj_df.columns.str.upper()
    adj_df = adj_df.reindex(index=nodes, columns=nodes).fillna(0.0)
    rows, cols = np.where(adj_df.values != 0)
    return torch.tensor(np.stack([rows, cols]), dtype=torch.long)

def create_pyg_data(features, edge_index):
    return Data(x=features, edge_index=edge_index)

def train_link_predictor(model, data, optimizer, neg_to_pos_ratio=5):
    model.train()
    optimizer.zero_grad()
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1) * neg_to_pos_ratio,
        method='sparse'
    )
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    temp_data = Data(x=data.x, edge_index=data.edge_index, edge_label_index=edge_label_index)
    edge_label = torch.cat([torch.ones(pos_edge_index.size(1)),
                            torch.zeros(neg_edge_index.size(1))]).to(data.x.device)

    neg_count = neg_edge_index.size(1)
    pos_count = pos_edge_index.size(1)
    pos_weight = torch.tensor([neg_count / pos_count if pos_count > 0 else 1.0]).to(data.x.device)
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    out = model(temp_data)
    loss = loss_func(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def infer_and_save_grn(model, data, nodes, output_path, batch_size=65536):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    num_nodes = data.num_nodes
    adj_matrix = pd.DataFrame(0.0, index=nodes, columns=nodes)
    all_pairs_rows, all_pairs_cols = torch.triu_indices(num_nodes, num_nodes, 1)
    all_pairs = torch.stack([all_pairs_rows, all_pairs_cols], dim=0)
    for perm in torch.utils.data.DataLoader(range(all_pairs.size(1)), batch_size=batch_size):
        edge_batch = all_pairs[:, perm].to(data.x.device)
        scores = torch.sigmoid(model.decode(z, edge_batch)).cpu().numpy()
        edge_indices_cpu = edge_batch.cpu().numpy()
        src_nodes = [nodes[i] for i in edge_indices_cpu[0]]
        dst_nodes = [nodes[i] for i in edge_indices_cpu[1]]
        for i in range(len(scores)):
            adj_matrix.loc[src_nodes[i], dst_nodes[i]] = scores[i]
            adj_matrix.loc[dst_nodes[i], src_nodes[i]] = scores[i]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    adj_matrix.to_csv(output_path)

# --- 主训练流程 (Updated for Disentanglement) ---
def run_experiment_CoGRN(config, device, atac_feat, rna_feat, edge_index, nodes, output_filename):

    print("\n" + "=" * 20 + " 运行: CoGRN+ (Disentangled) " + "=" * 20)

    dataset = TensorDataset(atac_feat, rna_feat)
    dataloader = DataLoader(dataset, batch_size=config['pretrain']['batch_size'], shuffle=True)

    contrastive_model = ContrastiveEncoders(
        atac_input_dim=atac_feat.shape[1],
        rna_input_dim=rna_feat.shape[1],
        embed_dim=config['model']['embed_dim']
    ).to(device)
    optimizer_pre = torch.optim.Adam(contrastive_model.parameters(), lr=config['pretrain']['lr'])

    # --- Pre-training Loop ---
    for epoch in range(1, config['pretrain']['epochs'] + 1):
        contrastive_model.train()
        total_loss = 0
        total_ortho = 0

        for batch_atac, batch_rna in dataloader:
            batch_atac, batch_rna = batch_atac.to(device), batch_rna.to(device)
            optimizer_pre.zero_grad()

            # Forward: 接收5个返回值
            p_atac, p_rna_shared, z_shared, z_private, _ = contrastive_model(batch_atac, batch_rna)

            # 1. InfoNCE Loss (只针对 Shared 和 ATAC)
            loss_nce = info_nce_loss(p_atac, p_rna_shared, temperature=config['pretrain']['temperature'])

            # 2. Orthogonal Loss (推开 Shared 和 Private)
            loss_ortho = orthogonality_loss(z_shared, z_private)

            # 总 Loss
            loss = loss_nce + 0.1 * loss_ortho

            loss.backward()
            optimizer_pre.step()
            total_loss += loss.item()
            total_ortho += loss_ortho.item()

        if epoch % 50 == 0:
            print(f"预训练 Epoch: {epoch:03d}, Loss: {total_loss/len(dataloader):.4f}, Ortho: {total_ortho/len(dataloader):.4f}")

    # --- Freeze & Fusion ---
    with torch.no_grad():
        # 获取最终特征，只取 Shared
        _, _, z_shared, _, atac_emb = contrastive_model(atac_feat, rna_feat)

    # Fusion (使用 z_shared 和 atac_emb)
    fusion_module = CoGRNFusion(embed_dim=config['model']['embed_dim'], fused_dim=config['model']['fused_dim'],
                                **config['fusion']).to(device)

    # 关键：传入 z_shared 作为 RNA 特征
    fused_embeddings = fusion_module(atac_emb, z_shared).detach()

    graph_data = create_pyg_data(fused_embeddings, edge_index).to(device)
    link_model = LinkPredictor(in_channels=graph_data.x.shape[1], **config['gnn']).to(device)
    optimizer_train = torch.optim.Adam(link_model.parameters(), lr=config['train']['lr'],
                                       weight_decay=config['train']['weight_decay'])

    # --- GNN Training Loop ---
    best_loss = float('inf')
    patience_counter = 0
    patience = config['train'].get('patience', 50)
    min_delta = config['train'].get('min_delta', 0.001)

    for epoch in range(1, config['train']['epochs'] + 1):
        loss = train_link_predictor(link_model, graph_data, optimizer_train,
                                    neg_to_pos_ratio=config['train'].get('neg_to_pos_ratio', 5))
        if epoch % 50 == 0:
            print(f"链接预测 Epoch: {epoch:03d}, 损失: {loss:.4f}")
        if loss < best_loss - min_delta:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"提前停止于 Epoch {epoch}")
            break

    # 保存模型 (支持迁移学习)
    model_save_path = output_filename.replace(".csv", "_model.pth")
    input_dims = {'atac_dim': atac_feat.shape[1], 'rna_dim': rna_feat.shape[1]}

    torch.save({
        'contrastive_state_dict': contrastive_model.state_dict(),
        'fusion_state_dict': fusion_module.state_dict(),
        'link_predictor_state_dict': link_model.state_dict(),
        'config': config,
        'input_dims': input_dims,
        'node_list': nodes
    }, model_save_path)
    print(f"=== [Disentangled] 模型已保存: {model_save_path} ===")

    infer_and_save_grn(link_model, graph_data, nodes, output_filename,
                       batch_size=config['train'].get('infer_batch_size', 65536))

# --- 数据预处理辅助函数 (用于 Classical Baseline) ---
def get_aligned_scrna_data(raw_scrna_dir: str, node_file: str) -> pd.DataFrame:
    raw_expr_data = load_mtx(raw_scrna_dir)
    with open(node_file, 'r') as f:
        nodes_to_evaluate = [line.strip().upper() for line in f]
    raw_expr_data.index = raw_expr_data.index.str.upper()
    common_genes = sorted(list(set(nodes_to_evaluate).intersection(raw_expr_data.index)))
    aligned_expr_data = raw_expr_data.loc[common_genes]
    return aligned_expr_data

def run_classical_baseline(algorithm_func, aligned_expr_data: pd.DataFrame, node_file: str, output_path: str, model_name: str):
    print(f"Running {model_name}...")
    with open(node_file, 'r') as f:
        nodes_to_evaluate = [line.strip().upper() for line in f]
    genes_to_run = list(aligned_expr_data.index)
    association = algorithm_func(expression_data=aligned_expr_data.T, tf_names=genes_to_run, verbose=True)
    network_sparse = association.pivot_table(index='TF', columns='target', values='importance', fill_value=0.0)
    final_network = network_sparse.reindex(index=nodes_to_evaluate, columns=nodes_to_evaluate).fillna(0.0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_network.to_csv(output_path)

def load_gold_standard(filepath: str):
    filename = os.path.basename(filepath)
    if filename.endswith('.csv'):
        adj_df = pd.read_csv(filepath, index_col=0)
        adj_df.index = adj_df.index.str.upper()
        adj_df.columns = adj_df.columns.str.upper()
        rows, cols = np.where(adj_df.values != 0)
        tfs = adj_df.index[rows]
        targets = adj_df.columns[cols]
        return set(zip(tfs, targets))
    elif filename.endswith(('.txt', '.tsv')):
        df = pd.read_csv(filepath, sep='\t', header=None, usecols=[0, 1])
        df.columns = ['TF', 'Target']
        return set(zip(df['TF'].str.upper(), df['Target'].str.upper()))

def evaluate_model(prediction_file: str, gold_standard_file: str, node_file: str, top_k: int = 1000):
    # (保持原有评估逻辑不变)
    with open(node_file, 'r') as f:
        nodes = [line.strip().upper() for line in f]
    full_gold_standard_edges = load_gold_standard(gold_standard_file)
    known_tfs = {tf for tf, target in full_gold_standard_edges}
    adj_matrix = pd.read_csv(prediction_file, index_col=0)
    adj_matrix.index = adj_matrix.index.str.upper()
    adj_matrix.columns = adj_matrix.columns.str.upper()
    adj_matrix = adj_matrix.reindex(index=nodes, columns=nodes).fillna(0.0)
    tfs_in_our_network = known_tfs.intersection(nodes)
    relevant_gold_standard_edges = {(tf, target) for tf, target in full_gold_standard_edges if tf in nodes and target in nodes}
    edge_scores = []
    for tf in tfs_in_our_network:
        for target in nodes:
            if tf == target: continue
            score = adj_matrix.loc[tf, target]
            label = 1 if (tf, target) in relevant_gold_standard_edges else 0
            edge_scores.append(((tf, target), score, label))
    edge_scores.sort(key=lambda x: x[1], reverse=True)
    edges, scores, true_labels = zip(*edge_scores)
    scores, true_labels = np.array(scores), np.array(true_labels)
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    aupr = auc(recall, precision)

    k = min(top_k, len(true_labels)) if top_k > 0 else 0
    precision_at_k = np.sum(true_labels[:k]) / k if k > 0 else 0
    total_possible_edges = len(tfs_in_our_network) * (len(nodes) - 1)
    random_precision = len(relevant_gold_standard_edges) / total_possible_edges if total_possible_edges > 0 else 0
    enrichment_fold = precision_at_k / random_precision if random_precision > 0 else 0

    return {
        "Precision@K": precision_at_k,
        "AUPR": aupr,
        "Enrichment Fold @K": enrichment_fold
    }

# --- Main Entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--config_dir', type=str, default='data/configs')
    args = parser.parse_args()

    config_filename = f"{args.dataset_name}_config.yaml"
    config_path = os.path.join(args.config_dir, config_filename)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    N_REPEATS = 1 # 迁移学习调试阶段跑一次即可
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = os.path.join(os.getcwd(), f"run_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)

    print("--- 加载数据 ---")
    node_path = os.path.join(config['paths']['aligned_data_dir'], "x_node.txt")
    atac_path = os.path.join(config['paths']['aligned_data_dir'], "x_atac_rp_score_feature.txt")
    rna_path = os.path.join(config['paths']['aligned_data_dir'], "x_scRNA_grnboost2_feature.txt")
    adj_path = os.path.join(config['paths']['aligned_data_dir'], "x_graph.csv")

    # 尝试加载 common_genes.txt
    common_genes_file = "common_genes_3sets.txt"
    if not os.path.exists(common_genes_file):
        common_genes_file = None
        print("Warning: common_genes_3sets.txt 未找到，将使用全量特征")

    atac_feat, rna_feat, nodes = load_features(atac_path, rna_path, node_path, common_genes_file)
    edge_index = load_adjacency_matrix(adj_path, nodes)

    atac_feat = atac_feat.to(device)
    rna_feat = rna_feat.to(device)
    edge_index = edge_index.to(device)

    # 运行
    run_output_dir = os.path.join(base_output_dir, "run_1")
    os.makedirs(run_output_dir, exist_ok=True)

    output_file = os.path.join(run_output_dir, "CoGRN_predicted_grn_GAT.csv")

    run_experiment_CoGRN(config, device, atac_feat, rna_feat, edge_index, nodes, output_file)

    print("Done.")
