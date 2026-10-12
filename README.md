# CoGRN and Trans-CoGRN Reproduction

This repository provides the code, processed inputs, pretrained weights, and execution scripts required to reproduce the CoGRN and Trans-CoGRN experiments described in *Trans-CoGRN: Multi-Omics Integration and Cross-Modal Transfer for Gene Regulatory Network Inference*.

The repository covers:

1. the CoGRN experiments for the Brain, Kidney, and PBMC datasets reported in Table 1;
2. the Trans-CoGRN experiments in Section 3.4, using one to five source domains for each target dataset.

All paths used by the scripts are relative to the repository root. Raw 10x matrices and preprocessing intermediates are not required because the experiments start from the processed node, graph, RNA-feature, and ATAC-feature files included under `data/`.

## Repository structure

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
│   ├── table1/                  # Processed inputs for Table 1
│   ├── datasets/                # Processed source- and target-domain datasets
│   ├── pretrained/              # Pretrained source-domain weights
│   ├── configs/                 # Target-domain configurations
│   └── gold_standards/          # Reference networks used for evaluation
├── result/Transfer_MultiSource/ # Predictions and checkpoints generated for Section 3.4
└── results/
    ├── table1/                  # Table 1 outputs
    └── section3_4/              # Section 3.4 outputs
```

The two main entry points are:

- `run_table1_gpu.sh` for the paired multi-omics experiments in Table 1;
- `run_section3_4_gpu.sh` for the multi-source transfer experiments in Section 3.4.

## Environment setup

The reproduction has been validated with:

- Linux;
- Python 3.9;
- PyTorch 2.4.0 with CUDA 12.4;
- PyTorch Geometric 2.6.1;
- one NVIDIA GPU.

Create and activate the environment:

```bash
conda create -n Trans_Cogrn python=3.9 -y
conda activate Trans_Cogrn

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify that the processed inputs, pretrained weights, Python dependencies, and GPU are available:

```bash
python verify_setup.py
```

## Large files

The pretrained weights and large feature matrices are managed with Git LFS. After cloning the repository, run:

```bash
git lfs install
git lfs pull
```

If the Git LFS objects have not been downloaded correctly, `verify_setup.py` reports missing or empty files.

## Reproducing Table 1

Run all three target datasets sequentially on GPU 0:

```bash
conda activate Trans_Cogrn
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh all
```

To run a single dataset:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh brain
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh kidney
CUDA_VISIBLE_DEVICES=0 bash run_table1_gpu.sh PBMC
```

Each run trains CoGRN-GAT, Concat-GNN, RNA-only-GNN, and ATAC-only-GNN. The final terminal table reports only the three metrics used in Table 1: `Precision@500`, `AUPR`, and `EF@500`. Predictions and metric tables are saved to `results/table1/runs/`, and terminal logs are saved to `reproduction_logs/table1/`.

## Reproducing Section 3.4

Run all 15 source-domain settings sequentially on GPU 0:

```bash
conda activate Trans_Cogrn
CUDA_VISIBLE_DEVICES=0 bash run_section3_4_gpu.sh all
```

To reproduce a single target dataset, replace `all` with `brain`, `PBMC`, or `kidney`:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_section3_4_gpu.sh brain
CUDA_VISIBLE_DEVICES=0 bash run_section3_4_gpu.sh PBMC
CUDA_VISIBLE_DEVICES=0 bash run_section3_4_gpu.sh kidney
```

The source domains are added in the following order:

- Brain: `PBMC`, `brain3k`, `Jejunum`, `lymph`, `kidney`;
- PBMC: `brain`, `brain3k`, `Jejunum`, `lymph`, `kidney`;
- Kidney: `PBMC`, `brain`, `brain3k`, `Jejunum`, `lymph`.

For each target dataset, the script trains and evaluates the model using the first one through five source domains in the corresponding list. Evaluation metrics are printed directly in the terminal. Predictions and intermediate checkpoints are written to `result/Transfer_MultiSource/`, and terminal logs are written to `reproduction_logs/section3_4/`. No plotting step is required.

## Reproducibility notes

- The scripts set the Python, NumPy, PyTorch, and CUDA random seeds.
- PyTorch Geometric negative sampling and GPU reductions may still introduce minor run-to-run variation.
- The source-domain order above also defines the two-source and four-source settings.
- The current experiments use one fixed-seed run per setting and do not estimate confidence intervals across random seeds.

## Code and data availability

The project repository associated with the manuscript is <https://github.com/Oyl-CityU/Trans-CoGRN>. The use and redistribution of the processed data and pretrained weights remain subject to the terms of the original data sources and project repository.
