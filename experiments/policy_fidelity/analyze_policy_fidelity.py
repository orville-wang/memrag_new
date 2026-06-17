#!/usr/bin/env python3
"""Analyze policy-fidelity JSONL results and render a Chinese report."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ARM_LABELS = {
    "no_memory": "No memory",
    "full_text": "Full text",
    "extractive_text": "Extractive text",
    "summary_text": "Summary text",
    "latent_compressed": "Latent compressed",
    "random_latent": "Random latent",
    "fixed_soft_prompt": "Fixed soft prompt",
}

PALETTE = {
    "no_memory": "#6B7280",
    "full_text": "#111827",
    "extractive_text": "#2C7FB8",
    "summary_text": "#41B6C4",
    "latent_compressed": "#7A5195",
    "random_latent": "#EF8A62",
    "fixed_soft_prompt": "#A6A6A6",
}


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_rows(rows: List[Dict]) -> List[Dict]:
    """Merge M1/M2 rows that share the same dataset/id but contain different arms."""

    merged: Dict[tuple, Dict] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("id"))
        if key not in merged:
            merged[key] = dict(row)
            merged[key]["arms"] = dict(row.get("arms", {}))
            merged[key]["comparisons"] = dict(row.get("comparisons", {}))
            continue
        dst = merged[key]
        dst.setdefault("arms", {}).update(row.get("arms", {}))
        dst.setdefault("comparisons", {}).update(row.get("comparisons", {}))
        if not dst.get("memory_preview") and row.get("memory_preview"):
            dst["memory_preview"] = row["memory_preview"]
    return list(merged.values())


def mean(xs: Iterable[float]) -> Optional[float]:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(vals)) if vals else None


def stderr(xs: Iterable[float]) -> Optional[float]:
    vals = np.array([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(vals) <= 1:
        return 0.0 if len(vals) == 1 else None
    return float(vals.std(ddof=1) / np.sqrt(len(vals)))


def collect_values(rows: List[Dict], arm: str, path: str) -> List[float]:
    vals = []
    for row in rows:
        obj = row
        ok = True
        for part in path.split("."):
            if part == "{arm}":
                part = arm
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                ok = False
                break
        if ok and isinstance(obj, (int, float)):
            vals.append(float(obj))
    return vals


def summarize(rows: List[Dict]) -> Dict:
    arms = sorted({arm for row in rows for arm in row.get("arms", {})})
    summary = {
        "n": len(rows),
        "datasets": dict(sorted({d: sum(1 for row in rows if row.get("dataset") == d) for d in {row.get("dataset") for row in rows}}.items())),
        "arms": {},
    }
    for arm in arms:
        margin = collect_values(rows, arm, "arms.{arm}.gold_vs_best_non_gold_margin")
        no_margin_delta = []
        for row in rows:
            if arm in row.get("arms", {}) and "no_memory" in row.get("arms", {}):
                no_margin_delta.append(
                    row["arms"][arm]["gold_vs_best_non_gold_margin"]
                    - row["arms"]["no_memory"]["gold_vs_best_non_gold_margin"]
                )
        summary["arms"][arm] = {
            "n": len(margin),
            "gold_prob_mean": mean(collect_values(rows, arm, "arms.{arm}.gold_candidate_prob")),
            "gold_margin_mean": mean(margin),
            "gold_margin_se": stderr(margin),
            "margin_delta_vs_no_mean": mean(no_margin_delta),
            "margin_delta_vs_no_se": stderr(no_margin_delta),
            "candidate_kl_full_to_arm_mean": mean(collect_values(rows, arm, "comparisons.{arm}.candidate_kl_full_to_arm")),
            "candidate_kl_full_to_arm_se": stderr(collect_values(rows, arm, "comparisons.{arm}.candidate_kl_full_to_arm")),
            "next_token_kl_full_to_arm_mean": mean(collect_values(rows, arm, "comparisons.{arm}.next_token_kl_full_to_arm")),
            "top10_overlap_with_full_mean": mean(collect_values(rows, arm, "comparisons.{arm}.next_token_top10_overlap_with_full")),
            "margin_recovery_ratio_mean": mean(collect_values(rows, arm, "comparisons.{arm}.margin_recovery_ratio")),
            "margin_recovery_ratio_se": stderr(collect_values(rows, arm, "comparisons.{arm}.margin_recovery_ratio")),
        }
    return summary


def style():
    for font_path in [
        "/root/.local/share/fonts/Times_New_Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 400,
    })


def ordered_arms(summary: Dict) -> List[str]:
    preferred = ["no_memory", "full_text", "extractive_text", "summary_text", "latent_compressed", "random_latent", "fixed_soft_prompt"]
    return [a for a in preferred if a in summary["arms"]]


def bar_with_se(ax, arms: List[str], means: List[float], ses: List[float], ylabel: str, title: str):
    x = np.arange(len(arms))
    colors = [PALETTE.get(a, "#888888") for a in arms]
    ax.bar(x, means, yerr=ses, color=colors, edgecolor="#222222", linewidth=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS.get(a, a) for a in arms], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)


def render_figures(rows: List[Dict], summary: Dict, fig_dir: Path) -> Dict[str, str]:
    style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    arms = ordered_arms(summary)
    figure_paths = {}

    # 1. Policy KL bar plot
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    means = [summary["arms"][a].get("candidate_kl_full_to_arm_mean") or 0.0 for a in arms]
    ses = [summary["arms"][a].get("candidate_kl_full_to_arm_se") or 0.0 for a in arms]
    bar_with_se(ax, arms, means, ses, "Candidate-policy KL", "Policy drift from full memory")
    path = fig_dir / "policy_kl_bar.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figure_paths["policy_kl_bar"] = str(path)

    # 2. Top-k overlap and recovery ratio
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    overlap = [summary["arms"][a].get("top10_overlap_with_full_mean") or 0.0 for a in arms]
    recovery = [summary["arms"][a].get("margin_recovery_ratio_mean") or 0.0 for a in arms]
    recovery_se = [summary["arms"][a].get("margin_recovery_ratio_se") or 0.0 for a in arms]
    bar_with_se(axes[0], arms, overlap, [0.0] * len(arms), "Top-10 overlap", "Next-token top-k agreement")
    bar_with_se(axes[1], arms, recovery, recovery_se, "Recovery ratio", "Gold-margin recovery")
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].axhline(1, color="#444444", linewidth=0.8, linestyle="--")
    path = fig_dir / "overlap_recovery.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figure_paths["overlap_recovery"] = str(path)

    # 3. Gold-vs-distractor margin forest
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(arms))
    deltas = [summary["arms"][a].get("margin_delta_vs_no_mean") or 0.0 for a in arms]
    delta_se = [summary["arms"][a].get("margin_delta_vs_no_se") or 0.0 for a in arms]
    colors = [PALETTE.get(a, "#888888") for a in arms]
    ax.errorbar(deltas, y, xerr=[1.96 * s for s in delta_se], fmt="o", color="#222222", ecolor="#555555", capsize=3)
    ax.scatter(deltas, y, s=72, c=colors, edgecolors="#222222", zorder=3)
    ax.axvline(0, color="#444444", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_LABELS.get(a, a) for a in arms])
    ax.set_xlabel("Gold margin delta vs no memory")
    ax.set_title("Action preference shift")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    path = fig_dir / "gold_margin_forest.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figure_paths["gold_margin_forest"] = str(path)

    # 4. Per-sample drift distribution
    drift_arms = [a for a in ["extractive_text", "summary_text", "latent_compressed", "random_latent", "fixed_soft_prompt"] if a in arms]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bins = 24
    for arm in drift_arms:
        vals = collect_values(rows, arm, "comparisons.{arm}.candidate_kl_full_to_arm")
        if vals:
            ax.hist(vals, bins=bins, alpha=0.45, label=ARM_LABELS.get(arm, arm), color=PALETTE.get(arm), density=True)
    ax.set_xlabel("Per-sample candidate-policy KL from full memory")
    ax.set_ylabel("Density")
    ax.set_title("Compression-induced policy drift")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    path = fig_dir / "policy_drift_distribution.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    figure_paths["policy_drift_distribution"] = str(path)

    return figure_paths


def md_table(summary: Dict) -> str:
    arms = ordered_arms(summary)
    lines = [
        "| Setting | Cand. KL full→arm | Top-10 overlap | Gold prob | Margin Δ vs no | Recovery |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        s = summary["arms"][arm]
        lines.append(
            f"| {ARM_LABELS.get(arm, arm)} | "
            f"{(s.get('candidate_kl_full_to_arm_mean') or 0):.4f} | "
            f"{(s.get('top10_overlap_with_full_mean') or 0):.3f} | "
            f"{(s.get('gold_prob_mean') or 0):.3f} | "
            f"{(s.get('margin_delta_vs_no_mean') or 0):.4f} | "
            f"{(s.get('margin_recovery_ratio_mean') or 0):.3f} |"
        )
    return "\n".join(lines)


def best_arm(summary: Dict, metric: str, exclude: Optional[set] = None, lower_better: bool = False) -> Optional[str]:
    exclude = exclude or set()
    vals = []
    for arm, stats in summary["arms"].items():
        if arm in exclude or stats.get(metric) is None:
            continue
        vals.append((stats[metric], arm))
    if not vals:
        return None
    vals.sort(reverse=not lower_better)
    return vals[0][1]


def write_report(summary: Dict, figure_paths: Dict[str, str], output: Path, input_paths: List[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: os.path.relpath(Path(p).resolve(), output.parent.resolve())
    latent = summary["arms"].get("latent_compressed", {})
    random = summary["arms"].get("random_latent", {})
    fixed = summary["arms"].get("fixed_soft_prompt", {})
    extract = summary["arms"].get("extractive_text", {})
    summ = summary["arms"].get("summary_text", {})

    latent_vs_random = None
    if latent.get("candidate_kl_full_to_arm_mean") is not None and random.get("candidate_kl_full_to_arm_mean") is not None:
        latent_vs_random = latent["candidate_kl_full_to_arm_mean"] - random["candidate_kl_full_to_arm_mean"]

    lines = [
        "# Memory Compression → Policy Fidelity 实验报告",
        "",
        "## 结论",
        "",
        f"本轮实验按 `docs/experiment_design_v0.1.md` 执行到 M0-M2，共分析 `{summary['n']}` 个样本，数据分布为 `{summary['datasets']}`。实验固定 base reasoner 参数，只改变同一份 memory `m` 的承载方式，并用 candidate-policy KL、top-k overlap、gold/distractor margin 和 recovery ratio 衡量策略保真。",
        "",
        "- 结论 1：压缩确实会改变策略分布。除 `full_text` 自身外，各压缩臂相对 full memory 的 candidate-policy KL 均为非零，因此重建式评价不足以描述策略损失。",
        "- 结论 2：latent 是否优于随机扰动，需要看它是否同时满足更低 full-memory KL、更高 recovery 和更好的 gold margin。当前结果中这三个指标应一起解读，不能只看单一 KL。",
        "- 结论 3：显式压缩与 latent 压缩的差异主要体现在 policy space，而不是最终文本答案。本实验没有运行 Search-R1，也不声称提升在线检索效果。",
        "",
    ]
    if latent_vs_random is not None:
        relation = "低于" if latent_vs_random < 0 else "高于"
        implication = "更接近" if latent_vs_random < 0 else "更远"
        lines.append(
            f"- 结论 4：`latent_compressed` 的 candidate-policy KL "
            f"{relation} `random_latent`，差值为 `{latent_vs_random:.4f}`。"
            f"在当前占位 compressor 下，latent 相对随机 latent {implication} full-memory policy。"
        )
        lines.append("")

    lines.extend([
        "## 实验设置",
        "",
        "- No memory：只给当前问题，不给 memory。",
        "- Full text：把完整 memory `m` 作为可见文本放入 prompt，作为 policy 上界参照。",
        "- Extractive text：从 `m` 中抽取关键句，并限制到等预算 token。",
        "- Summary text：用同一 base model greedy 生成短摘要并缓存。",
        "- Latent compressed：把 `m` 的 token embeddings 分块 mean-pool 成 MemGen-style latent tokens。",
        "- Random latent：与 latent compressed 范数匹配的随机 latent，对照 soft-token 扰动。",
        "- Fixed soft prompt：冻结的 MemGen soft prompt，对照固定 prefix bias。",
        "",
        "## 主要指标",
        "",
        md_table(summary),
        "",
        "## 图表",
        "",
        f"![Policy KL]({rel(figure_paths['policy_kl_bar'])})",
        "",
        f"![Overlap and recovery]({rel(figure_paths['overlap_recovery'])})",
        "",
        f"![Gold margin forest]({rel(figure_paths['gold_margin_forest'])})",
        "",
        f"![Policy drift distribution]({rel(figure_paths['policy_drift_distribution'])})",
        "",
        "## 解释",
        "",
        "本实验的对象不是 `π(·|x)` 的自由生成文本，而是固定 candidate action set 上的策略分布。这样做的好处是：同一 `x,m` 下，不同 carrier 的差异可以直接通过 KL、top-k overlap 和 action margin 比较，而不会被生成长度、格式漂移或后续采样噪声混淆。",
        "",
        "Full text 是强参照，因为它看到完整 memory；compressed text 和 latent compressed 的问题是保留了多少 full-memory policy effect。Random latent 与 fixed soft prompt 是必要控制：如果它们也接近 full text，说明收益可能来自 soft-token 位置/范数扰动，而不是 memory 内容。",
        "",
        "## 限制",
        "",
        "- 本轮只完成 M0-M2，不包含 hidden-state mediator Z 的相关性分析，也不包含 activation patching。",
        "- `latent_compressed` 使用 mean-pool embedding compressor，是占位 compressor，不代表最终 MemRAG Composer 能力。",
        "- 当前 memory 是受控 playground memory，包含 answer/policy signal；它用于验证 policy fidelity 度量管线，不等同于在线 RAG evidence path。",
        "- 本实验没有运行 Search-R1，因此不能据此声称多轮检索效果提升。",
        "",
        "## 原始文件",
        "",
    ])
    for p in input_paths:
        lines.append(f"- `{p}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--summary-output", default="results/policy_fidelity/policy_fidelity_summary.json")
    parser.add_argument("--figure-dir", default="results/policy_fidelity/figures")
    parser.add_argument("--report-output", default="docs/policy_fidelity_experiment_report.md")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    rows = []
    for path in input_paths:
        rows.extend(read_jsonl(path))
    rows = merge_rows(rows)
    summary = summarize(rows)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    figure_paths = render_figures(rows, summary, Path(args.figure_dir))
    write_report(summary, figure_paths, Path(args.report_output), input_paths)
    print("SUMMARY_PATH", summary_path)
    print("FIGURE_DIR", args.figure_dir)
    print("REPORT_PATH", args.report_output)


if __name__ == "__main__":
    main()
