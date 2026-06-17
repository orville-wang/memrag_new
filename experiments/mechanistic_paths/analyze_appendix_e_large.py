#!/usr/bin/env python3
"""Large-sample paired analysis for Appendix-E retrieval-weaver experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ARMS = ["memgen", "expel_trained", "expel_untrained"]
ARM_LABELS = {
    "memgen": "MemGen",
    "expel_trained": "Retrieval + trained",
    "expel_untrained": "Retrieval + untrained",
}
METRICS = {
    "gold_answer_prob": "Gold prob",
    "gold_vs_distractor_margin": "Gold margin",
    "next_token_entropy": "Entropy",
}
COLORS = {
    "memgen": "#4C78A8",
    "expel_trained": "#59A14F",
    "expel_untrained": "#E15759",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.dpi": 140,
            "savefig.dpi": 360,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_input(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"input must be label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    return label, Path(path)


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def values(rows: List[Dict], arm: str, metric: str) -> np.ndarray:
    return np.array([float(r["arms"][arm][metric]) for r in rows], dtype=float)


def mean_sem(x: Iterable[float]) -> Tuple[float, float]:
    arr = np.array(list(x), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(arr.mean()), sem


def bootstrap_ci(x: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    if x.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(rows: List[Dict], left: str, right: str, metric: str, n_boot: int, seed: int) -> Dict:
    delta = values(rows, left, metric) - values(rows, right, metric)
    mean, sem = mean_sem(delta)
    ci_low, ci_high = bootstrap_ci(delta, n_boot, seed)
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "mean_delta": mean,
        "sem": sem,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "positive_rate": float((delta > 0).mean()),
        "negative_rate": float((delta < 0).mean()),
        "zero_rate": float((delta == 0).mean()),
        "n": int(delta.size),
    }


def summarize_model(rows: List[Dict], n_boot: int, seed: int) -> Dict:
    summary = {"n": len(rows), "arms": {}, "paired": {}}
    for arm in ARMS:
        summary["arms"][arm] = {}
        for metric in METRICS:
            arr = values(rows, arm, metric)
            mean, sem = mean_sem(arr)
            ci_low, ci_high = bootstrap_ci(arr, n_boot, seed)
            summary["arms"][arm][metric] = {
                "mean": mean,
                "sem": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
    for pair in [("expel_trained", "memgen"), ("expel_trained", "expel_untrained"), ("expel_untrained", "memgen")]:
        key = f"{pair[0]}_minus_{pair[1]}"
        summary["paired"][key] = {
            metric: paired_stats(rows, pair[0], pair[1], metric, n_boot, seed)
            for metric in METRICS
        }
    return summary


def plot_metric_bars(all_summary: Dict[str, Dict], out_dir: Path) -> None:
    models = list(all_summary)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, metric in zip(axes, METRICS):
        width = 0.24
        x = np.arange(len(models))
        for i, arm in enumerate(ARMS):
            means = [all_summary[m]["arms"][arm][metric]["mean"] for m in models]
            lows = [all_summary[m]["arms"][arm][metric]["ci95_low"] for m in models]
            highs = [all_summary[m]["arms"][arm][metric]["ci95_high"] for m in models]
            yerr = np.array([[means[j] - lows[j] for j in range(len(models))], [highs[j] - means[j] for j in range(len(models))]])
            ax.bar(x + (i - 1) * width, means, width, yerr=yerr, capsize=3, color=COLORS[arm], label=ARM_LABELS[arm])
        ax.set_title(METRICS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "appendix_e_large_metric_bars.png", bbox_inches="tight")
    plt.close(fig)


def plot_paired_forest(all_summary: Dict[str, Dict], out_dir: Path) -> None:
    pairs = [
        ("expel_trained_minus_memgen", "Trained - MemGen"),
        ("expel_trained_minus_expel_untrained", "Trained - Untrained"),
    ]
    metrics = ["gold_answer_prob", "gold_vs_distractor_margin", "next_token_entropy"]
    height = max(4.6, 0.78 * len(all_summary) * len(pairs) + 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, height), sharey=True)
    for ax, metric in zip(axes, metrics):
        y_labels = []
        y = []
        means = []
        lows = []
        highs = []
        colors = []
        pos = 0
        for model, summary in all_summary.items():
            for pair_key, pair_label in pairs:
                item = summary["paired"][pair_key][metric]
                y_labels.append(f"{model}: {pair_label}")
                y.append(pos)
                means.append(item["mean_delta"])
                lows.append(item["ci95_low"])
                highs.append(item["ci95_high"])
                colors.append("#59A14F" if "trained_minus_memgen" in pair_key else "#E15759")
                pos += 1
        means = np.array(means)
        xerr = np.array([means - np.array(lows), np.array(highs) - means])
        ax.errorbar(means, y, xerr=xerr, fmt="o", color="#333333", ecolor="#666666", capsize=3, markersize=4)
        for yi, mi, c in zip(y, means, colors):
            ax.scatter([mi], [yi], color=c, s=34, zorder=3)
        ax.axvline(0, color="#BDBDBD", linewidth=1)
        ax.set_title(f"Delta: {METRICS[metric]}")
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
        if ax is axes[0]:
            ax.set_yticks(y)
            ax.set_yticklabels(y_labels)
        else:
            ax.set_yticks(y)
    fig.tight_layout()
    fig.savefig(out_dir / "appendix_e_large_paired_delta_forest.png", bbox_inches="tight")
    plt.close(fig)


def write_report(all_summary: Dict[str, Dict], out_dir: Path, report_path: Path, inputs: Dict[str, Path]) -> None:
    lines = [
        "# Appendix-E Retrieval-Weaver 大样本统计",
        "",
        "本报告使用 paired 统计检查大样本下的结论是否稳定。每个样本同时跑 MemGen、Retrieval + trained weaver、Retrieval + untrained weaver，因此 paired delta 比单纯均值更能排除样本难度差异。",
        "",
        "## 均值",
        "",
        "| Model | n | Path | Gold prob | Gold margin | Entropy |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for model, summary in all_summary.items():
        for arm in ARMS:
            lines.append(
                f"| {model} | {summary['n']} | {ARM_LABELS[arm]} | "
                f"{summary['arms'][arm]['gold_answer_prob']['mean']:.4f} | "
                f"{summary['arms'][arm]['gold_vs_distractor_margin']['mean']:.4f} | "
                f"{summary['arms'][arm]['next_token_entropy']['mean']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Paired Delta",
            "",
            "| Model | Comparison | Metric | Mean delta | 95% bootstrap CI | Positive rate |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for model, summary in all_summary.items():
        for pair_key, pair_label in [
            ("expel_trained_minus_memgen", "Trained - MemGen"),
            ("expel_trained_minus_expel_untrained", "Trained - Untrained"),
        ]:
            for metric in METRICS:
                item = summary["paired"][pair_key][metric]
                lines.append(
                    f"| {model} | {pair_label} | {METRICS[metric]} | "
                    f"{item['mean_delta']:.4f} | [{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] | "
                    f"{item['positive_rate']:.3f} |"
                )
    lines.extend(
        [
            "",
            "## 图表",
            "",
            f"![metric bars]({(out_dir / 'appendix_e_large_metric_bars.png').resolve()})",
            "",
            f"![paired delta]({(out_dir / 'appendix_e_large_paired_delta_forest.png').resolve()})",
            "",
            "## 结论读法",
            "",
            "- 本轮最稳定的结论是：Retrieval + trained weaver 相对 MemGen 在两个 base model 上都提高 gold prob 和 gold margin，且 bootstrap CI 不跨 0；Pooled-200 也保持同向。",
            "- Trained - Untrained 不是单调胜出：Qwen1.5B 上 trained 明显优于 untrained；SmolLM3-3B 上 untrained 的 gold prob 更高，但 trained 明显降低 entropy。",
            "- 因此不能只看候选答案概率。untrained weaver 有时能通过强扰动偶然提高 gold prob，但 trained weaver 的作用路径更像可控的 latent policy modulation。",
            "",
            "## Raw Inputs",
            "",
        ]
    )
    for model, path in inputs.items():
        lines.append(f"- {model}: `{path.resolve()}`")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="label=path")
    parser.add_argument("--out-dir", default="results/mechanistic_paths/figures_appendix_e_large")
    parser.add_argument("--summary-output", default="results/mechanistic_paths/appendix_e_large_summary.json")
    parser.add_argument("--report-output", default="docs/mechanistic_paths_appendix_e_large_report.md")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_style()
    inputs = dict(parse_input(spec) for spec in args.input)
    loaded_rows = {model: load_rows(path) for model, path in inputs.items()}
    all_summary = {
        model: summarize_model(rows, args.bootstrap, args.seed)
        for model, rows in loaded_rows.items()
    }
    if len(loaded_rows) > 1:
        pooled_rows = [row for rows in loaded_rows.values() for row in rows]
        all_summary[f"Pooled-{len(pooled_rows)}"] = summarize_model(pooled_rows, args.bootstrap, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_bars(all_summary, out_dir)
    plot_paired_forest(all_summary, out_dir)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(all_summary, out_dir, Path(args.report_output), inputs)
    print("SUMMARY_OUTPUT", args.summary_output)
    print("REPORT_OUTPUT", args.report_output)
    print("FIGURE_DIR", out_dir)


if __name__ == "__main__":
    main()
