"""
step_full_joint.py
==========================================================
导师要求: 把 step2_3 (Encoder+Mapper) 与 step4 (Fusion+GNN) 合成
一个真正的端到端联合训练, 让 encoder/mapper 直接以 GRN 链接预测损失
为目标进行参数更新, 而不仅仅是生成/对齐任务的目标.

总损失 (端到端反传):
  L = Σ_s [ NCE(emb_a_s, z_s_shared)
            + λ1 · MSE(mapper(z_s_shared), emb_a_s.detach())
            + λ2 · Ortho(z_s_shared, z_s_private)
            + λ3 · MMD(z_s_shared, z_t_shared)
            + λ_grn_src · BCE_src(fusion(emb_a_s, z_s_shared) → GAT, A_s) ]
    + λ2 · Ortho(z_t_shared, z_t_private)
    + λ_grn_tgt · BCE_tgt(fusion(mapper(z_t_shared), z_t_shared) → GAT, A_t)

源域 GRN 损失用 source 真实 ATAC + 真实 prior graph A_s;
目标域 GRN 损失用 mapper-imputed ATAC + DeepSEM-derived prior graph A_t.

不修改 step2_3_joint_optimization.py / step4_target_gnn_adaptation.py;
本文件由 run_pipeline.py 通过 --joint_full 开关触发.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

sys.path.append(os.getcwd())
from trans_model import (
    ContrastiveEncoders, CoGRNFusion, LinkPredictor,
    info_nce_loss, load_adjacency_matrix, infer_and_save_grn,
    create_pyg_data,
)
from data_loader_transfer import load_aligned_data
from step3_train_da_mapper import CrossModalMapper
from step2_3_joint_optimization import mmd_loss, ortho_loss


def link_loss_full_graph(link_model, fused, edge_index, neg_to_pos_ratio, device):
    """
    全图 BCE 链接预测损失 (从 step4 抽出来, 适配 detached/non-detached 输入).
    """
    pos = edge_index
    if pos.numel() == 0:
        return torch.tensor(0.0, device=device)
    neg = negative_sampling(
        edge_index=pos,
        num_nodes=fused.size(0),
        num_neg_samples=pos.size(1) * neg_to_pos_ratio,
        method='sparse',
    )
    edge_label_index = torch.cat([pos, neg], dim=1)
    edge_label = torch.cat([
        torch.ones(pos.size(1), device=device),
        torch.zeros(neg.size(1), device=device),
    ])
    z = link_model.encode(fused, pos)
    out = link_model.decode(z, edge_label_index)
    pos_weight = torch.tensor([float(neg_to_pos_ratio)], device=device)
    return F.binary_cross_entropy_with_logits(out, edge_label, pos_weight=pos_weight)


class UniversalJointGRN:
    """端到端联合训练 encoder + mapper + fusion + GNN."""

    def __init__(self, config_path, common_genes_file, device):
        self.device = device
        self.common_genes_file = common_genes_file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def run(self, source_configs, target_config,
            save_enc_path, save_map_path, save_fusion_path, save_gnn_path,
            output_pred_csv,
            lambda_1=1.0, lambda_2=2.0, lambda_3=0.5,
            lambda_grn_src=1.0, lambda_grn_tgt=1.0,
            epochs=300, lr=1e-4, neg_to_pos_ratio=5,
            grad_clip=1.0):
        device = self.device

        print(f"\n=== [Step 2-3-4 Full Joint] encoder + mapper + fusion + GNN ===")
        print(f"Target: {target_config['name']}, Sources: {[s['name'] for s in source_configs]}")
        print(f"λ1={lambda_1} (Map), λ2={lambda_2} (Ortho), λ3={lambda_3} (MMD), "
              f"λ_grn_src={lambda_grn_src}, λ_grn_tgt={lambda_grn_tgt}")
        print(f"epochs={epochs}, lr={lr}, neg/pos={neg_to_pos_ratio}, grad_clip={grad_clip}\n")

        # -------------------- 1. 数据 (全图加载, GRN 需要全部节点) --------------------
        src_data = []
        for s in source_configs:
            print(f"Loading Source: {s['name']} ...")
            a, r, nodes_s = load_aligned_data(
                s['atac'], s['rna'], s['node'],
                self.common_genes_file, device,
            )
            ei = load_adjacency_matrix(s['adj'], nodes_s).to(device)
            src_data.append({
                'name': s['name'], 'atac': a, 'rna': r,
                'edge_index': ei, 'orig_model': s.get('orig_model'),
            })

        print(f"\nLoading Target: {target_config['name']} ...")
        _, tgt_rna, tgt_nodes = load_aligned_data(
            target_config['atac'], target_config['rna'], target_config['node'],
            self.common_genes_file, device,
        )
        tgt_edge_index = load_adjacency_matrix(target_config['adj'], tgt_nodes).to(device)
        print(f"Target nodes={len(tgt_nodes)}, prior edges={tgt_edge_index.shape[1]}")

        # -------------------- 2. 构建模型 --------------------
        rna_dim = src_data[0]['rna'].shape[1]
        embed_dim = self.config['model']['embed_dim']
        fused_dim = self.config['model']['fused_dim']

        # Base 容器 (取第一个源域的 ATAC 维度仅用于占位; 真正 ATAC encoder 在 atac_encoders 里)
        base_model = ContrastiveEncoders(
            src_data[0]['atac'].shape[1], rna_dim, embed_dim,
        ).to(device)

        # 每个源域一个独立的 ATAC encoder
        atac_encoders = nn.ModuleList()
        for i, s in enumerate(src_data):
            tmp = ContrastiveEncoders(s['atac'].shape[1], rna_dim, embed_dim).to(device)
            orig = s.get('orig_model')
            if orig and os.path.exists(orig):
                ckpt = torch.load(orig, map_location=device)
                state = {
                    k.replace('atac_encoder.', ''): v
                    for k, v in ckpt['contrastive_state_dict'].items()
                    if k.startswith('atac_encoder.')
                }
                tmp.atac_encoder.load_state_dict(state)
                print(f"  -> 加载 {s['name']} 预训练 ATAC Encoder 权重")
            atac_encoders.append(tmp.atac_encoder)

        mapper = CrossModalMapper(input_dim=embed_dim, output_dim=embed_dim).to(device)

        fusion = CoGRNFusion(
            embed_dim=embed_dim, fused_dim=fused_dim,
            **self.config['fusion'],
        ).to(device)

        link_pred = LinkPredictor(in_channels=fused_dim, **self.config['gnn']).to(device)

        # 用第一个源域的 fusion+GNN 权重做 warm-start (如果有)
        first_orig = src_data[0].get('orig_model')
        if first_orig and os.path.exists(first_orig):
            ckpt = torch.load(first_orig, map_location=device)
            try:
                fusion.load_state_dict(ckpt['fusion_state_dict'])
                link_pred.load_state_dict(ckpt['link_predictor_state_dict'])
                print(f"  -> Warm-start fusion + link_pred from {os.path.basename(first_orig)}")
            except Exception as e:
                print(f"  -> Warm-start 跳过: {e}")

        # -------------------- 3. 优化器 --------------------
        params = (list(base_model.parameters())
                  + list(atac_encoders.parameters())
                  + list(mapper.parameters())
                  + list(fusion.parameters())
                  + list(link_pred.parameters()))
        optimizer = optim.Adam(params, lr=lr)
        mse = nn.MSELoss()

        # -------------------- 4. 训练循环 (全图, 一步反传) --------------------
        print(f">>> Joint Training start ...\n")
        for epoch in range(1, epochs + 1):
            base_model.train()
            atac_encoders.train()
            mapper.train()
            fusion.train()
            link_pred.train()

            optimizer.zero_grad()

            log = {'NCE': 0.0, 'Map': 0.0, 'Ortho': 0.0,
                   'MMD': 0.0, 'GRN_s': 0.0, 'GRN_t': 0.0}
            total = torch.tensor(0.0, device=device)

            # Target 共享/私有特征 (供 MMD + Target GRN)
            z_t_shared, z_t_private = base_model.rna_encoder(tgt_rna)

            # Per-source 损失
            for i, s in enumerate(src_data):
                emb_a = atac_encoders[i](s['atac'])
                z_s_shared, z_s_private = base_model.rna_encoder(s['rna'])

                # 1. NCE
                p_a = base_model.atac_projection(emb_a)
                p_r = base_model.rna_projection(z_s_shared)
                l_nce = info_nce_loss(p_a, p_r)

                # 2. Mapper MSE (detach ATAC 防止 ATAC encoder 向 mapper 坍缩)
                l_map = mse(mapper(z_s_shared), emb_a.detach())

                # 3. Ortho on source
                l_ortho_s = ortho_loss(z_s_shared, z_s_private)

                # 4. MMD (源 vs 目标)
                l_mmd = mmd_loss(z_s_shared, z_t_shared)

                # 5. Source GRN: fusion(real_atac, shared) -> GNN -> BCE on A_s
                fused_s = fusion(emb_a, z_s_shared)
                l_grn_s = link_loss_full_graph(link_pred, fused_s, s['edge_index'],
                                               neg_to_pos_ratio, device)

                total = (total + l_nce + lambda_1 * l_map
                         + lambda_2 * l_ortho_s + lambda_3 * l_mmd
                         + lambda_grn_src * l_grn_s)

                log['NCE']   += l_nce.item()
                log['Map']   += l_map.item()
                log['Ortho'] += l_ortho_s.item()
                log['MMD']   += l_mmd.item()
                log['GRN_s'] += l_grn_s.item()

            # Target Ortho
            l_ortho_t = ortho_loss(z_t_shared, z_t_private)
            total = total + lambda_2 * l_ortho_t
            log['Ortho'] += l_ortho_t.item()

            # Target GRN: fusion(imputed_atac, shared) -> GNN -> BCE on A_t
            z_t_atac_hat = mapper(z_t_shared)
            fused_t = fusion(z_t_atac_hat, z_t_shared)
            l_grn_t = link_loss_full_graph(link_pred, fused_t, tgt_edge_index,
                                           neg_to_pos_ratio, device)
            total = total + lambda_grn_tgt * l_grn_t
            log['GRN_t'] = l_grn_t.item()

            total.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()

            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                ns = max(len(src_data), 1)
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"NCE={log['NCE']/ns:.4f} Map={log['Map']/ns:.4f} "
                      f"Ortho={log['Ortho']/(ns+1):.4f} MMD={log['MMD']/ns:.4f} | "
                      f"GRN_src={log['GRN_s']/ns:.4f} GRN_tgt={log['GRN_t']:.4f} | "
                      f"Total={total.item():.4f}")

        # -------------------- 5. 保存权重 --------------------
        os.makedirs(os.path.dirname(save_enc_path), exist_ok=True)
        torch.save({
            'contrastive_state_dict': base_model.state_dict(),
            'config': self.config,
            'input_dims': {'rna': rna_dim},
            'target_name': target_config['name'],
        }, save_enc_path)
        torch.save(mapper.state_dict(), save_map_path)
        torch.save(fusion.state_dict(), save_fusion_path)
        torch.save(link_pred.state_dict(), save_gnn_path)
        print(f"\n模型权重已保存 -> {os.path.dirname(save_enc_path)}")

        # -------------------- 6. 推理 --------------------
        print(f">>> Inference on target {target_config['name']} ...")
        base_model.eval(); mapper.eval(); fusion.eval(); link_pred.eval()
        with torch.no_grad():
            z_t_shared_inf, _ = base_model.rna_encoder(tgt_rna)
            z_t_atac_inf = mapper(z_t_shared_inf)
            z_t_fused_inf = fusion(z_t_atac_inf, z_t_shared_inf)
            graph_data = create_pyg_data(z_t_fused_inf, tgt_edge_index).to(device)
            infer_and_save_grn(
                link_pred, graph_data, tgt_nodes, output_pred_csv,
                batch_size=self.config['train'].get('infer_batch_size', 65536),
            )
        print(f"Done. Prediction saved -> {output_pred_csv}")
