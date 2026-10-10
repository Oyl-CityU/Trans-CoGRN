#!/usr/bin/env python3
"""Validate the packaged inputs and the active CUDA environment."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
TARGETS = {"brain": "H_brain", "kidney": "H_kidney", "PBMC": "PBMC"}
GRAPH_FILES = (
    "x_node.txt",
    "x_graph.csv",
    "x_scRNA_grnboost2_feature.txt",
    "x_atac_rp_score_feature.txt",
)


def require(path: Path, missing: list[Path]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        missing.append(path)


def main() -> int:
    missing: list[Path] = []

    for target, joint_name in TARGETS.items():
        for filename in GRAPH_FILES:
            require(ROOT / "data" / "table1" / target / "graph" / filename, missing)
            require(ROOT / "data" / "datasets" / joint_name / "graph" / filename, missing)

    for source in ("H_brain3k", "H_jejunum", "H_lymph"):
        for filename in GRAPH_FILES:
            require(ROOT / "data" / "datasets" / source / "graph" / filename, missing)

    for filename in (
        "brain_aligned_model.pth",
        "brain3k_aligned_model.pth",
        "Jejunum_aligned_model.pth",
        "lymph_aligned_model.pth",
        "Kidney_aligned_model.pth",
        "PBMC_aligned_model.pth",
    ):
        require(ROOT / "data" / "pretrained" / filename, missing)

    for filename in (
        "Brain_gold_standard_network.txt",
        "kidney_gold_standard_network.txt",
        "PBMC_gold_standard_network.txt",
    ):
        require(ROOT / "data" / "gold_standards" / filename, missing)

    if missing:
        print("Missing or empty required files:", file=sys.stderr)
        for path in missing:
            print(f"  - {path.relative_to(ROOT)}", file=sys.stderr)
        print("Run `git lfs pull` and retry.", file=sys.stderr)
        return 1

    try:
        import torch
        import torch_geometric
    except ImportError as exc:
        print(f"Missing Python dependency: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("Required files are complete, but PyTorch cannot see a CUDA GPU.", file=sys.stderr)
        return 1

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyG: {torch_geometric.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("Repository setup is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
