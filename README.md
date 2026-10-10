# CoGRN 与 Trans-CoGRN 实验复现

本仓库提供复现论文 *Trans-CoGRN: Multi-Omics Integration and Cross-Modal Transfer for Gene Regulatory Network Inference* 中 CoGRN 与 Trans-CoGRN 实验所需的代码、处理后输入、预训练权重和运行脚本。

仓库覆盖以下实验：

1. Table 1 中 Brain、Kidney 和 PBMC 三个数据集上的 CoGRN 实验；
2. 论文第 3.4 节中，以三个数据集分别作为目标域、依次使用一至五个源域的 Trans-CoGRN 实验。

运行脚本中的路径均相对于仓库根目录。实验直接使用 `data/` 中已处理的节点、图结构、RNA 特征和 ATAC 特征文件，因此不需要原始 10x 矩阵及预处理中间文件。

## 项目结构

```text
.
├── README.md
├── requirements.txt
├── verify_setup.py
├── run_table1_gpu.sh
├── run_section3_4_gpu.sh
├── table1/
│   ├── CoGrn_improve.py
│   └── config/
│       ├── brain_config.yaml
│       ├── kidney_config.yaml
│       └── PBMC_config.yaml
├── run_pipeline.py
├── step_full_joint.py
├── trans_model.py
├── data_loader_transfer.py
├── evaluate_transfer.py
├── common_genes_7sets.txt
├── data/
│   ├── table1/                  # Table 1 所需的处理后输入
│   ├── datasets/                # 源域与目标域的处理后数据
│   ├── pretrained/              # 源域预训练权重
│   ├── configs/                 # 目标域配置文件
│   └── gold_standards/          # 评估所用参考网络
├── result/Transfer_MultiSource/ # 第 3.4 节运行时生成的预测与检查点
└── results/
    ├── table1/                  # Table 1 输出
    └── section3_4/              # 第 3.4 节输出
```

两个主要运行入口为：

- `run_table1_gpu.sh`：复现 Table 1 中的配对多组学实验；
- `run_section3_4_gpu.sh`：复现第 3.4 节中的多源域迁移实验。

## 环境配置

本项目已在以下环境中完成验证：

- Linux；
- Python 3.9；
- PyTorch 2.4.0（CUDA 12.4）；
- PyTorch Geometric 2.6.1；
- 单张 NVIDIA GPU。

创建并激活环境：

```bash
conda create -n CoGRN python=3.9 -y
conda activate CoGRN

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

检查处理后输入、预训练权重、Python 依赖和 GPU 是否可用：

```bash
python verify_setup.py
```

## 大文件下载

预训练权重和较大的特征矩阵由 Git LFS 管理。克隆仓库后执行：

```bash
git lfs install
git lfs pull
```

如果 Git LFS 对象未正确下载，`verify_setup.py` 会报告缺失或内容为空的文件。

## 复现 Table 1

在 GPU 0 上依次运行三个目标数据集：

```bash
conda activate CoGRN
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh all
```

也可以只运行一个数据集：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh brain
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh kidney
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh PBMC
```

每次运行会训练 CoGRN-GAT、Concat-GNN、RNA-only-GNN 和 ATAC-only-GNN。终端最终仅输出 Table 1 使用的三项指标：`Precision@500`、`AUPR` 和 `EF@500`。预测文件和指标表保存在 `results/table1/runs/`，终端日志保存在 `reproduction_logs/table1/`。

## 复现论文第 3.4 节

在 GPU 0 上依次运行全部 15 组源域设置：

```bash
conda activate CoGRN
CUDA_VISIBLE_DEVICES=0 bash run_section3_4_gpu.sh all
```

如只需复现一个目标数据集，可将 `all` 替换为 `brain`、`PBMC` 或 `kidney`：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_section3_4_gpu.sh brain
CUDA_VISIBLE_DEVICES=0 bash run_section3_4_gpu.sh PBMC
CUDA_VISIBLE_DEVICES=0 bash run_section3_4_gpu.sh kidney
```

源域加入顺序如下：

- Brain：`PBMC`、`brain3k`、`Jejunum`、`lymph`、`kidney`；
- PBMC：`brain`、`brain3k`、`Jejunum`、`lymph`、`kidney`；
- Kidney：`PBMC`、`brain`、`brain3k`、`Jejunum`、`lymph`。

对于每个目标域，脚本按上述顺序分别使用前一至五个源域进行训练和评估。评估指标直接打印到终端。预测文件和中间检查点写入 `result/Transfer_MultiSource/`，终端日志写入 `reproduction_logs/section3_4/`。该复现流程不需要绘图步骤。

## 可复现性说明

- 运行脚本固定 Python、NumPy、PyTorch 和 CUDA 的随机种子；
- PyTorch Geometric 的负采样及 GPU 归约计算仍可能产生少量运行间差异；
- 上述源域顺序同时定义了双源域和四源域等实验设置；
- 当前实验对每项设置运行一次固定随机种子，不计算跨随机种子的置信区间。

## 代码与数据来源

论文相关项目主页为 <https://github.com/Oyl-CityU/Trans-CoGRN>。处理后数据与预训练权重的使用和再分发应遵循原始数据来源及原项目仓库的相关条款。
