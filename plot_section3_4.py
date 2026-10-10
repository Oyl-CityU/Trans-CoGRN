#!/usr/bin/env python3
"""Collect and plot all metrics reported in Section 3.4 and Figure S2."""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Order: DeepSEM, 1 src, 2 src, 3 src, 4 src, 5 src.
PAPER = {
    "brain": {
        "precision": [0.620, 0.724, 0.798, 0.790, 0.806, 0.786],
        "ndcg": [0.634, 0.702, 0.793, 0.786, 0.793, 0.778],
        "ef": [1.029, 1.202, 1.325, 1.312, 1.338, 1.305],
        "aupr": [0.600, 0.639, 0.639, 0.644, 0.643, 0.640],
    },
    "PBMC": {
        "precision": [0.692, 0.746, 0.744, 0.758, 0.750, 0.766],
        "ndcg": [0.682, 0.737, 0.734, 0.746, 0.740, 0.754],
        "ef": [1.006, 1.085, 1.082, 1.102, 1.091, 1.114],
        "aupr": [0.693, 0.717, 0.719, 0.716, 0.707, 0.712],
    },
    "kidney": {
        "precision": [0.702, 0.724, 0.710, 0.704, 0.730, 0.716],
        "ndcg": [0.688, 0.704, 0.695, 0.711, 0.690, 0.701],
        "ef": [0.986, 1.016, 0.997, 1.025, 0.988, 1.005],
        "aupr": [0.708, 0.698, 0.708, 0.705, 0.708, 0.709],
    },
}

METRIC_RE = re.compile(
    r"\[METRICS_JSON\]\s+model=(DeepSEM|Trans-CoGRN)\s+"
    r"gs=Combined_Network\s+(\{.*\})"
)

METRIC_KEYS = {
    "precision": "Precision@K",
    "ndcg": "NDCG@K",
    "ef": "Enrichment Fold @K",
    "aupr": "AUPR",
}


def parse_log(path: Path):
    metrics = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    if "使用设备: cuda" not in text:
        raise RuntimeError(f"CUDA confirmation missing in {path}")
    for model, payload in METRIC_RE.findall(text):
        metrics[model] = json.loads(payload)
    if "DeepSEM" not in metrics or "Trans-CoGRN" not in metrics:
        raise RuntimeError(f"Incomplete metrics in {path}")
    if "NDCG@K" not in metrics["Trans-CoGRN"]:
        raise RuntimeError(f"NDCG@K missing in {path}; rerun with the updated evaluator")
    return metrics


def collect(log_dir: Path):
    rows = []
    for target in ("brain", "PBMC", "kidney"):
        for nsrc in range(1, 6):
            path = log_dir / f"FIG2_GPU_{target}{nsrc}.log"
            if not path.exists():
                continue
            metrics = parse_log(path)
            row = {"target": target, "num_sources": nsrc}
            for short, key in METRIC_KEYS.items():
                baseline = metrics["DeepSEM"][key]
                observed = metrics["Trans-CoGRN"][key]
                row[f"deepsem_{short}"] = baseline
                row[f"trans_cogrn_{short}"] = observed
                row[f"paper_deepsem_{short}"] = PAPER[target][short][0]
                row[f"paper_trans_cogrn_{short}"] = PAPER[target][short][nsrc]
                row[f"delta_vs_paper_{short}"] = observed - PAPER[target][short][nsrc]
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No completed Section 3.4 logs found in {log_dir}")
    return rows


def write_csv(rows, output_dir: Path):
    path = output_dir / "section3_4_gpu_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def values_for(rows, target, metric):
    target_rows = sorted((r for r in rows if r["target"] == target), key=lambda r: r["num_sources"])
    if not target_rows:
        return []
    return [target_rows[0][f"deepsem_{metric}"]] + [r[f"trans_cogrn_{metric}"] for r in target_rows]


def draw_panel(ax, values, title, ylabel=None):
    colors = ["#A7AFB5", "#C6E2E9", "#A9D3DF", "#79B3C8", "#4C91AD", "#28677F"]
    labels = ["DeepSEM", "1 src", "2 src", "3 src", "4 src", "5 src"][: len(values)]
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors[: len(values)], edgecolor="#3AA66B", linewidth=0.9)
    bars[0].set_edgecolor("#A7AFB5")
    best = int(np.argmax(values[1:])) + 1
    bars[best].set_edgecolor("#D99200")
    bars[best].set_linewidth(1.8)
    baseline = values[0]
    for idx, (bar, value) in enumerate(zip(bars, values)):
        if idx > 0 and value < baseline:
            bar.set_edgecolor("#D9674F")
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
    spread = max(values) - min(values)
    margin = max(0.008, spread * 0.32)
    ax.set_ylim(min(values) - margin, max(values) + margin)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_title(title, fontsize=9, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.6, linestyle=":")
    ax.set_axisbelow(True)


def export(fig, base: Path):
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_figure2(rows, output_dir: Path):
    targets = [t for t in ("brain", "PBMC", "kidney") if values_for(rows, t, "precision")]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.15 * len(targets), 3.05), squeeze=False)
    for index, (ax, target) in enumerate(zip(axes[0], targets)):
        draw_panel(ax, values_for(rows, target, "precision"), "Brain" if target == "brain" else target,
                   "Precision@500" if index == 0 else None)
    fig.tight_layout()
    export(fig, output_dir / "figure2_precision_gpu_reproduction")


def draw_figure_s2(rows, output_dir: Path):
    targets = [t for t in ("brain", "PBMC", "kidney") if values_for(rows, t, "ndcg")]
    metric_rows = [("ndcg", "NDCG@500"), ("ef", "Enrichment Fold@500"), ("aupr", "AUPR")]
    fig, axes = plt.subplots(3, len(targets), figsize=(3.15 * len(targets), 7.8), squeeze=False)
    for row_idx, (metric, ylabel) in enumerate(metric_rows):
        for col_idx, target in enumerate(targets):
            title = ("Brain" if target == "brain" else target) if row_idx == 0 else ""
            draw_panel(axes[row_idx, col_idx], values_for(rows, target, metric), title,
                       ylabel if col_idx == 0 else None)
    fig.tight_layout()
    export(fig, output_dir / "figureS2_gpu_reproduction")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.log_dir.parent.parent / "section3_4_reproduction"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    rows = collect(args.log_dir)
    csv_path = write_csv(rows, output_dir)
    draw_figure2(rows, output_dir)
    draw_figure_s2(rows, output_dir)
    expected = 5 * len({row["target"] for row in rows})
    print(f"Collected {len(rows)}/{expected} transfer settings")
    print(f"Metrics: {csv_path}")
    print(f"Main figure: {output_dir / 'figure2_precision_gpu_reproduction.pdf'}")
    print(f"Supplementary figure: {output_dir / 'figureS2_gpu_reproduction.pdf'}")


if __name__ == "__main__":
    main()
