import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import yaml
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将上一级目录添加到系统路径中
sys.path.append(parent_dir)
# 导入 CoGRN
sys.path.append(os.getcwd())
from trans_model import ContrastiveEncoders, info_nce_loss
from data_loader_transfer import load_aligned_data
from step3_train_da_mapper import CrossModalMapper

# MMD Loss
def mmd_loss(source_features, target_features):
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    gaussian_kernel = lambda x, y: sum([torch.exp(-(torch.cdist(x, y) ** 2) / (2 * s ** 2)) for s in sigmas])
    return torch.mean(gaussian_kernel(source_features, source_features)) + \
           torch.mean(gaussian_kernel(target_features, target_features)) - \
           2 * torch.mean(gaussian_kernel(source_features, target_features))

# Orthogonality Loss
def ortho_loss(z_shared, z_private):
    # L_ortho = 1/N \sum |z_shared * z_private| / (||z_shared|| * ||z_private||)
    cos_sim = torch.abs(torch.sum(z_shared * z_private, dim=-1)) / (torch.norm(z_shared, dim=-1) * torch.norm(z_private, dim=-1) + 1e-8)
    return torch.mean(cos_sim)

class UniversalDomainAdapterJoint:
    def __init__(self, config_path, common_genes_file, device):
        self.device = device
        self.common_genes_file = common_genes_file
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def run(self, source_configs, target_config, save_enc_path, save_map_path,
            lambda_1=1.0, lambda_2=1.0, lambda_3=1.0, epochs=120, lr=1e-4):
        print(f"\n=== 启动通用域适应与跨模态映射联合优化 (Joint Optimization) ===")
        print(f"Target Domain: {target_config['name']}")
        print(f"Source Domains: {[s['name'] for s in source_configs]}")

        # 1. 加载 Target RNA
        print(f"Loading Target: {target_config['name']} ...")
        _, tgt_rna, _ = load_aligned_data(target_config['atac'], target_config['rna'], target_config['node'], self.common_genes_file, self.device)
        tgt_loader = DataLoader(TensorDataset(tgt_rna), batch_size=64, shuffle=True, drop_last=True)

        # 2. 加载所有 Source 数据
        src_loaders = []
        input_dims = [] # (atac_dim, rna_dim)

        for src in source_configs:
            print(f"Loading Source: {src['name']} ...")
            a, r, _ = load_aligned_data(src['atac'], src['rna'], src['node'], self.common_genes_file, self.device)
            # 创建 Loader
            loader = DataLoader(TensorDataset(a, r), batch_size=64, shuffle=True, drop_last=True)
            src_loaders.append(loader)
            input_dims.append((a.shape[1], r.shape[1]))

        # 3. 构建动态模型架构
        print("构建动态多源模型架构 (Encoder + Mapper)...")
        rna_dim = input_dims[0][1] # 所有源域 RNA 维度相同
        embed_dim = self.config['model']['embed_dim']

        # 实例化 Base Model (作为容器)
        base_model = ContrastiveEncoders(input_dims[0][0], rna_dim, embed_dim).to(self.device)

        # 实例化特定于源域的 ATAC Encoders
        atac_encoders = nn.ModuleList()
        for i, (a_dim, _) in enumerate(input_dims):
            tmp = ContrastiveEncoders(a_dim, rna_dim, embed_dim).to(self.device)

            # [重要修改] 初始化时，如果有预训练权重，则加载预训练模型的 ATAC Encoder 权重
            # 这有助于 Mapper 在联合训练中有一个相对稳定的 ground-truth ATAC 特征
            orig_model_path = source_configs[i].get('orig_model')
            if orig_model_path and os.path.exists(orig_model_path):
                print(f"  -> 加载 {source_configs[i]['name']} 的预训练 ATAC Encoder 权重...")
                checkpoint = torch.load(orig_model_path, map_location=self.device)
                atac_state_dict = {k.replace('atac_encoder.', ''): v for k, v in checkpoint['contrastive_state_dict'].items() if k.startswith('atac_encoder.')}
                tmp.atac_encoder.load_state_dict(atac_state_dict)

            atac_encoders.append(tmp.atac_encoder)

        # 实例化 Cross-Modal Mapper
        mapper = CrossModalMapper(input_dim=embed_dim, output_dim=embed_dim).to(self.device)

        # 优化器: 优化 RNA Encoder + 所有 ATAC Encoders + Projection Heads + Mapper
        params = list(base_model.rna_encoder.parameters()) + \
                 list(base_model.rna_projection.parameters()) + \
                 list(base_model.atac_projection.parameters()) + \
                 list(atac_encoders.parameters()) + \
                 list(mapper.parameters())

        optimizer = optim.Adam(params, lr=lr)
        mse_loss_fn = nn.MSELoss()

        # 4. 训练循环
        print(f">>> Start Joint Adaptation & Mapping Training... (epochs={epochs}, lr={lr}, λ1={lambda_1}, λ2={lambda_2}, λ3={lambda_3})")
        base_model.train()
        atac_encoders.train()
        mapper.train()

        min_steps = min([len(l) for l in src_loaders] + [len(tgt_loader)])

        for epoch in range(epochs):
            total_nce = 0
            total_mmd = 0
            total_ortho = 0
            total_map = 0

            src_iters = [iter(l) for l in src_loaders]
            tgt_iter = iter(tgt_loader)

            for _ in range(min_steps):
                # Target Batch
                try:
                    t_batch = next(tgt_iter)[0]
                except StopIteration:
                    break

                optimizer.zero_grad()
                loss_step = 0

                # B. Forward Target (Shared RNA)
                z_t_shared, z_t_private = base_model.rna_encoder(t_batch)

                # 遍历每个源域
                for i, s_iter in enumerate(src_iters):
                    try:
                        s_atac, s_rna = next(s_iter)
                    except StopIteration:
                        continue

                    # A. Forward Source (Specific ATAC + Shared RNA)
                    emb_a = atac_encoders[i](s_atac)
                    z_s_shared, z_s_private = base_model.rna_encoder(s_rna)

                    # 1. NCE Loss (语义保持)
                    p_a = base_model.atac_projection(emb_a)
                    p_r = base_model.rna_projection(z_s_shared)
                    loss_nce = info_nce_loss(p_a, p_r)

                    # 2. MMD Loss (域对齐)
                    loss_mmd_val = mmd_loss(z_s_shared, z_t_shared)

                    # 3. Orthogonality Loss (特征解耦) - 针对 Source
                    loss_ortho_val = ortho_loss(z_s_shared, z_s_private)

                    # 4. Mapper Loss (跨模态映射)
                    mapped_atac = mapper(z_s_shared)
                    loss_map = mse_loss_fn(mapped_atac, emb_a.detach()) # detached emb_a to avoid collapsing ATAC encoder towards mapper

                    # 累加 Loss
                    loss_step += (loss_nce + lambda_1 * loss_map + lambda_2 * loss_ortho_val + lambda_3 * loss_mmd_val)

                    total_nce += loss_nce.item()
                    total_mmd += loss_mmd_val.item()
                    total_ortho += loss_ortho_val.item()
                    total_map += loss_map.item()

                # Target Ortho Loss
                loss_ortho_t = ortho_loss(z_t_shared, z_t_private)
                loss_step += lambda_2 * loss_ortho_t
                total_ortho += loss_ortho_t.item()

                loss_step.backward()
                optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}: NCE={total_nce/min_steps:.4f}, MMD={total_mmd/min_steps:.4f}, ORTHO={total_ortho/min_steps:.4f}, MAP={total_map/min_steps:.4f}")

        # 5. 保存模型
        os.makedirs(os.path.dirname(save_enc_path), exist_ok=True)
        # 保存 Universal Encoder
        torch.save({
            'contrastive_state_dict': base_model.state_dict(),
            'config': self.config,
            'input_dims': {'rna': rna_dim},
            'target_name': target_config['name']
        }, save_enc_path)
        print(f"Adapted Universal Encoder Saved: {save_enc_path}")

        # 保存 Mapper
        os.makedirs(os.path.dirname(save_map_path), exist_ok=True)
        torch.save(mapper.state_dict(), save_map_path)
        print(f"Universal Mapper Saved: {save_map_path}")
