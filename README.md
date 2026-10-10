# Reproduction of CoGRN and Trans-CoGRN Experiments

本仓库提供论文 *Trans-CoGRN: Multi-Omics Integration and Cross-Modal Transfer for Gene Regulatory Network Inference*（Chong Chen, Xiling Luo, Ying Li and Le Ou-Yang）的实验代码、处理后输入、模型权重、运行配置和复现结果。所有实验均通过仓库内的相对路径执行，覆盖以下内容：

1. Table 1 的 CoGRN-GAT、Concat-GNN、RNA-only-GNN、ATAC-only-GNN；
2. 第 3.4 节每个目标域使用 1–5 个源域的 Trans-CoGRN 推断；
3. 主文 Figure 2（Precision@500）和 Supplementary Figure S2（NDCG@500、EF@500、AUPR）。

为避免重复存储未被复现实验直接读取的文件，仓库不包含原始 10x 矩阵、MAESTRO/FIMO 临时文件和训练中间 checkpoint。`data/datasets/` 中的六个处理后数据集及 `data/pretrained/` 中的权重用于三目标域多源迁移实验。

## 复现结果

### Table 1：CoGRN-GAT

| Dataset | Precision@500（Reproduced） | Reported | AUPR（Reproduced） | Reported | EF@500（Reproduced） | Reported |
|---|---:|---:|---:|---:|---:|---:|
| Brain | 0.840 | 0.838 | 0.671466 | 0.672 | 1.394628 | 1.391 |
| Kidney | 0.868 | 0.868 | 0.735477 | 0.736 | 1.218568 | 1.219 |
| PBMC | 0.870 | 0.872 | 0.730528 | 0.731 | 1.265227 | 1.268 |

四种模型的完整机器可读结果见 `results/table1/all_datasets_metrics.csv`。

### 第 3.4 节：Trans-CoGRN Precision@500

| Target | 1 source | 2 sources | 3 sources | 4 sources | 5 sources | DeepSEM baseline |
|---|---:|---:|---:|---:|---:|---:|
| Brain | 0.724 | 0.802 | 0.792 | 0.816 | 0.746 | 0.620 |
| PBMC | 0.756 | 0.766 | 0.726 | 0.754 | 0.758 | 0.692 |
| Kidney | 0.734 | 0.728 | 0.650 | 0.716 | 0.734 | 0.702 |

15 个设置均已使用 CUDA 完成。14/15 个设置的 Precision@500 高于 DeepSEM；Kidney 3-source 本次运行低于基线，显示训练链路存在运行间波动。Precision、NDCG、EF、AUPR、论文值及差值均在 `results/section3_4/all_datasets_metrics.csv` 中。

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── run_table1_gpu.sh
├── run_section3_4_gpu.sh
├── table1/
│   ├── CoGrn_improve.py
│   └── config/{brain,kidney,PBMC}_config.yaml
├── run_pipeline.py
├── step_full_joint.py
├── trans_model.py
├── data_loader_transfer.py
├── CoGrn_improve.py
├── plot_section3_4.py
├── data/
│   ├── table1/{brain,kidney,PBMC}/
│   ├── datasets/
│   ├── pretrained/
│   ├── configs/{brain,kidney,PBMC}_config.yaml
│   └── gold_standards/
└── results/
    ├── table1/
    └── section3_4/
```

## 环境与依赖

验证环境：Linux、Python 3.9、PyTorch 2.4.0 + CUDA 12.4、PyTorch Geometric 2.6.1。主要 Python 依赖记录在 `requirements.txt`。

```bash
conda create -n CoGRN python=3.9 -y
conda activate CoGRN
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python verify_setup.py
```

## Git LFS 大文件

预训练权重和大型特征矩阵由 `.gitattributes` 标记为 Git LFS 文件。首次发布或克隆时执行：

```bash
git lfs install
git lfs pull
git lfs ls-files
```

若 `git lfs` 尚未安装，必须先安装它，再执行 `git add`，否则不应直接推送本目录的大文件。

## 复现 Table 1

激活 `CoGRN` 环境后，在项目根目录执行全部三个数据集：

```bash
conda activate CoGRN
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh all
```

只运行单个数据集：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh brain
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh kidney
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh PBMC
```

每次均运行四个模型，结果写入 `results/table1/runs/`，日志写入 `reproduction_logs/table1/`。

## 复现第 3.4 节和 Figure 2 / Figure S2

三个目标域共 15 个设置耗时较长，并严格串行使用 GPU 0：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_section3_4_gpu.sh all
```

也可只运行一个目标域，将 `all` 改为 `brain`、`PBMC` 或 `kidney`。训练预测写入 `result/Transfer_MultiSource/`，日志写入 `reproduction_logs/section3_4/`，最终指标及 PDF/SVG/PNG 图片写入 `results/section3_4/figures/`。

源域组合如下：

- Brain：`PBMC` → `+brain3k` → `+Jejunum` → `+lymph` → `+kidney`；
- PBMC：`brain` → `+brain3k` → `+Jejunum` → `+lymph` → `+kidney`；
- Kidney：`PBMC` → `+brain` → `+brain3k` → `+Jejunum` → `+lymph`。

## 已附结果

- `results/table1/all_datasets_metrics.csv`：Table 1 三数据集、四模型指标；
- `results/table1/<dataset>/CoGRN_predicted_grn_GAT.csv`：三个 CoGRN 预测矩阵；
- `results/section3_4/all_datasets_metrics.csv`：第 3.4 节全部 15 个设置的指标；
- `results/section3_4/*TransCoGRN_Finetuned.csv`：15 个预测网络；
- `results/section3_4/figures/`：完整 Figure 2 与 Figure S2；
- `results/section3_4/logs/`：15 次已完成 GPU 运行的审计日志。

## 可复现性说明

- 固定了 Python、NumPy、PyTorch 和 CUDA 相关随机设置，但 PyTorch Geometric 的负采样和 GPU 浮点归约仍可能带来小幅波动。
- 2-source 和 4-source 设置按上面的既有源域顺序取前 2/4 个域，这是当前项目可审计的组合定义。
- NDCG@500 按预测分数降序排列，使用 `1/log2(rank+1)` 折损并以理想 Top-500 排序归一化。
- 当前结果是单次固定种子运行，不代表跨种子均值或置信区间。

## Code and data availability

论文对应的项目主页为 <https://github.com/Oyl-CityU/Trans-CoGRN>。本仓库中的处理后数据和预训练权重仅用于验证文中报告的实验结果；其使用和再分发应遵循原始数据来源及项目仓库的许可条款。
