#!/usr/bin/env python3
"""评估单个目标数据集上的 DeepSEM 与 Trans-CoGRN 预测结果。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import auc, precision_recall_curve


ROOT = Path(__file__).resolve().parent


def load_gold_standard(path: Path):
    if path.suffix == ".csv":
        adjacency = pd.read_csv(path, index_col=0)
        adjacency.index = adjacency.index.str.upper()
        adjacency.columns = adjacency.columns.str.upper()
        rows, columns = np.where(adjacency.values != 0)
        return set(zip(adjacency.index[rows], adjacency.columns[columns]))

    table = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1])
    table.columns = ["TF", "Target"]
    table["TF"] = table["TF"].str.upper()
    table["Target"] = table["Target"].str.upper()
    return set(zip(table["TF"], table["Target"]))


def evaluate(prediction_path: Path, gold_path: Path, node_path: Path, top_k=500):
    nodes = [line.strip().upper() for line in node_path.read_text().splitlines()]
    gold_edges = load_gold_standard(gold_path)
    known_tfs = {tf for tf, _ in gold_edges}
    relevant_edges = {
        (tf, target) for tf, target in gold_edges if tf in nodes and target in nodes
    }

    adjacency = pd.read_csv(prediction_path, index_col=0)
    adjacency.index = adjacency.index.str.upper()
    adjacency.columns = adjacency.columns.str.upper()
    adjacency = adjacency.reindex(index=nodes, columns=nodes).fillna(0.0)

    ranked = []
    for tf in known_tfs.intersection(nodes):
        for target in nodes:
            if tf != target:
                ranked.append((adjacency.loc[tf, target], int((tf, target) in relevant_edges)))
    ranked.sort(key=lambda item: item[0], reverse=True)

    scores = np.asarray([item[0] for item in ranked])
    labels = np.asarray([item[1] for item in ranked])
    k = min(top_k, len(labels))
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    precision_at_k = float(labels[:k].sum() / k) if k else 0.0
    positives = int(labels.sum())

    possible_edges = len(known_tfs.intersection(nodes)) * (len(nodes) - 1)
    random_precision = len(relevant_edges) / possible_edges if possible_edges else 0.0
    enrichment = precision_at_k / random_precision if random_precision else float("inf")

    discounts = 1.0 / np.log2(np.arange(k) + 2.0) if k else np.array([])
    dcg = float(np.sum(labels[:k] * discounts)) if k else 0.0
    ideal_positives = min(k, positives)
    idcg = float(np.sum(discounts[:ideal_positives])) if ideal_positives else 0.0

    return {
        "Precision@K": precision_at_k,
        "AUPR": float(auc(recall_curve, precision_curve)),
        "Enrichment Fold @K": enrichment,
        "NDCG@K": dcg / idcg if idcg else 0.0,
    }


def resolve_repository_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("brain", "kidney", "PBMC"))
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()

    config_path = ROOT / "data" / "configs" / f"{args.target}_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    node_path = resolve_repository_path(config["paths"]["aligned_data_dir"]) / "x_node.txt"
    gold_path = resolve_repository_path(
        config["paths"]["gold_standards"]["Combined_Network"]
    )
    predictions = {
        "DeepSEM": ROOT / "data" / "datasets" / f"DeepSEM_predicted_grn_{args.target}.csv",
        "Trans-CoGRN": ROOT / "result" / "Transfer_MultiSource" / f"{args.target}_TransCoGRN_Finetuned.csv",
    }

    rows = {}
    for model, prediction_path in predictions.items():
        metrics = evaluate(prediction_path, gold_path, node_path, top_k=args.top_k)
        rows[model] = metrics
        print(
            f"[METRICS_JSON] model={model} gs=Combined_Network "
            + json.dumps(metrics, default=float)
        )

    table = pd.DataFrame.from_dict(rows, orient="index").rename(columns={
        "Precision@K": f"Precision@{args.top_k}",
        "Enrichment Fold @K": f"EF@{args.top_k}",
        "NDCG@K": f"NDCG@{args.top_k}",
    })
    table = table[[f"Precision@{args.top_k}", f"NDCG@{args.top_k}",
                   f"EF@{args.top_k}", "AUPR"]]
    print(f"\n第 3.4 节评估结果：目标数据集={args.target}")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
