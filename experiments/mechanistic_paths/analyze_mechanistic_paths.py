#!/usr/bin/env python3
"""Analyze mechanistic path experiment outputs and render paper-style figures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


ARMS = ["memgen", "expel_trained", "expel_untrained"]
ARM_LABELS = {
    "no_latent": "No latent",
    "memgen": "MemGen",
    "expel_trained": "Retrieval + trained weaver",
    "expel_untrained": "Retrieval + untrained weaver",
}
COLORS = {
    "memgen": "#4C78A8",
    "expel_trained": "#59A14F",
    "expel_untrained": "#E15759",
    "no_latent": "#8C8C8C",
}


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def sem(values: Iterable[float]) -> float:
    vals = np.array([float(v) for v in values], dtype=float)
    if vals.size <= 1:
        return 0.0
    return float(vals.std(ddof=1) / np.sqrt(vals.size))


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "figure.dpi": 140,
            "savefig.dpi": 360,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def summarize(rows: List[Dict]) -> Dict:
    summary: Dict = {"n": len(rows), "arms": {}, "comparisons": {}, "layers": {}}
    for arm in ["no_latent"] + ARMS:
        arm_rows = [row["arms"][arm] for row in rows if arm in row["arms"]]
        summary["arms"][arm] = {
            "n": len(arm_rows),
            "gold_answer_prob_mean": mean(r["gold_answer_prob"] for r in arm_rows),
            "gold_answer_prob_sem": sem(r["gold_answer_prob"] for r in arm_rows),
            "search_prob_mean": mean(r["search_prob"] for r in arm_rows),
            "search_prob_sem": sem(r["search_prob"] for r in arm_rows),
            "answer_prob_mean": mean(r["answer_prob"] for r in arm_rows),
            "gold_vs_distractor_margin_mean": mean(r["gold_vs_distractor_margin"] for r in arm_rows),
            "search_vs_answer_margin_mean": mean(r["search_vs_answer_margin"] for r in arm_rows),
            "next_token_entropy_mean": mean(r["next_token_entropy"] for r in arm_rows),
            "latent_norm_mean": mean(r.get("latent_norm_mean", 0.0) for r in arm_rows),
        }

    comparison_keys = sorted({key for row in rows for key in row.get("comparisons", {})})
    for key in comparison_keys:
        vals = [row["comparisons"][key].get("next_token_kl") for row in rows if key in row.get("comparisons", {})]
        vals = [v for v in vals if v is not None]
        if vals:
            summary["comparisons"][key] = {"next_token_kl_mean": mean(vals), "next_token_kl_sem": sem(vals), "n": len(vals)}

    for arm in ARMS:
        layer_hidden = defaultdict(list)
        layer_attn = defaultdict(list)
        layer_lens_action = defaultdict(list)
        layer_lens_answer = defaultdict(list)
        for row in rows:
            arm_row = row["arms"].get(arm)
            if not arm_row:
                continue
            for item in arm_row.get("hidden_drift_to_no_latent", []):
                layer_hidden[int(item["layer"])].append(float(item["l2_to_no_latent"]))
            for item in arm_row.get("latent_attention_mass", []):
                layer_attn[int(item["layer"])].append(float(item["latent_attention_mass"]))
            for item in arm_row.get("layer_action_key_lens", []):
                layer = int(item["layer"])
                layer_lens_action[layer].append(float(item["search_vs_answer_logit_margin"]))
            for item in arm_row.get("layer_answer_content_lens", []):
                layer = int(item["layer"])
                layer_lens_answer[layer].append(float(item["gold_vs_distractor_logit_margin"]))
        summary["layers"][arm] = {
            "hidden_l2_to_no_latent": [{"layer": k, "mean": mean(v), "sem": sem(v)} for k, v in sorted(layer_hidden.items())],
            "latent_attention_mass": [{"layer": k, "mean": mean(v), "sem": sem(v)} for k, v in sorted(layer_attn.items())],
            "search_vs_answer_logit_margin": [{"layer": k, "mean": mean(v), "sem": sem(v)} for k, v in sorted(layer_lens_action.items())],
            "gold_vs_distractor_logit_margin": [{"layer": k, "mean": mean(v), "sem": sem(v)} for k, v in sorted(layer_lens_answer.items())],
        }
    return summary


def plot_candidate_policy(summary: Dict, out_dir: Path):
    arms = ARMS
    x = np.arange(len(arms))
    gold = [summary["arms"][arm]["gold_answer_prob_mean"] for arm in arms]
    search = [summary["arms"][arm]["search_prob_mean"] for arm in arms]
    err = [summary["arms"][arm]["gold_answer_prob_sem"] for arm in arms]
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    width = 0.34
    ax.bar(x - width / 2, search, width, color="#8AB17D", label="Search candidates")
    ax.bar(x + width / 2, gold, width, color="#4C78A8", yerr=err, capsize=3, label="Gold answer")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms], rotation=18, ha="right")
    ax.set_ylabel("Candidate policy probability")
    ax.set_ylim(0, max(0.05, min(1.0, max(search + gold) * 1.25)))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_policy_probability.png", bbox_inches="tight")
    plt.close(fig)


def plot_kl(summary: Dict, out_dir: Path):
    keys = ["no_latent_to_memgen", "no_latent_to_expel_trained", "no_latent_to_expel_untrained", "memgen_to_expel_trained", "expel_trained_to_expel_untrained"]
    labels = ["No latent -> MemGen", "No latent -> RTW", "No latent -> RUW", "MemGen -> RTW", "RTW -> RUW"]
    vals = [summary["comparisons"].get(k, {}).get("next_token_kl_mean", np.nan) for k in keys]
    errs = [summary["comparisons"].get(k, {}).get("next_token_kl_sem", 0.0) for k in keys]
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    ax.bar(np.arange(len(vals)), vals, yerr=errs, capsize=3, color=["#8C8C8C", "#59A14F", "#E15759", "#76B7B2", "#F28E2B"])
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel("Next-token KL")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "next_token_policy_kl.png", bbox_inches="tight")
    plt.close(fig)


def line_from_layer(items: List[Dict]):
    return np.array([i["layer"] for i in items]), np.array([i["mean"] for i in items]), np.array([i["sem"] for i in items])


def plot_layer_metric(summary: Dict, out_dir: Path, metric_key: str, ylabel: str, filename: str, include_no_marker: bool = False):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for arm in ARMS:
        items = summary["layers"].get(arm, {}).get(metric_key, [])
        if not items:
            continue
        layers, vals, errs = line_from_layer(items)
        ax.plot(layers, vals, color=COLORS[arm], linewidth=2.3, label=ARM_LABELS[arm])
        ax.fill_between(layers, vals - errs, vals + errs, color=COLORS[arm], alpha=0.16, linewidth=0)
    if include_no_marker:
        ax.axhline(0, color="#BDBDBD", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_generation_tokens(rows: List[Dict], out_dir: Path):
    token_counts = {arm: defaultdict(int) for arm in ARMS}
    for row in rows:
        for arm in ARMS:
            text = row["arms"][arm].get("generation", {}).get("text", "")
            if "<search>" in text:
                key = "search tag"
            elif "<answer>" in text:
                key = "answer tag"
            elif not text.strip():
                key = "empty"
            else:
                key = "other"
            token_counts[arm][key] += 1
    keys = ["search tag", "answer tag", "other", "empty"]
    data = np.array([[token_counts[arm][key] for key in keys] for arm in ARMS], dtype=float)
    denom = data.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1
    data = data / denom
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bottom = np.zeros(len(ARMS))
    palette = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#76B7B2", "#B07AA1", "#9C755F", "#BAB0AC"]
    for idx, key in enumerate(keys):
        ax.bar(np.arange(len(ARMS)), data[:, idx], bottom=bottom, color=palette[idx % len(palette)], label=key)
        bottom += data[:, idx]
    ax.set_xticks(np.arange(len(ARMS)))
    ax.set_xticklabels([ARM_LABELS[a] for a in ARMS], rotation=18, ha="right")
    ax.set_ylabel("Generated prefix share")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "generation_prefix_mix.png", bbox_inches="tight")
    plt.close(fig)


def write_report(rows: List[Dict], summary: Dict, out_dir: Path, report_path: Path, raw_path: Path, summary_path: Path):
    comp = summary["comparisons"]
    arms = summary["arms"]
    trained_gain = arms["expel_trained"]["gold_answer_prob_mean"] - arms["memgen"]["gold_answer_prob_mean"]
    trained_margin_gain = arms["expel_trained"]["gold_vs_distractor_margin_mean"] - arms["memgen"]["gold_vs_distractor_margin_mean"]
    untrained_gap = arms["expel_trained"]["gold_answer_prob_mean"] - arms["expel_untrained"]["gold_answer_prob_mean"]
    entropy_gap = arms["expel_untrained"]["next_token_entropy_mean"] - arms["expel_trained"]["next_token_entropy_mean"]
    kl_trained_untrained = comp.get("expel_trained_to_expel_untrained", {}).get("next_token_kl_mean", float("nan"))
    if trained_margin_gain >= 0:
        margin_clause = f"gold-vs-distractor margin 同步提高 {trained_margin_gain:+.4f}"
    else:
        margin_clause = f"gold-vs-distractor margin 下降 {trained_margin_gain:+.4f}"
    if untrained_gap >= 0:
        untrained_clause = f"它读到同一批 retrieved experience，但 gold probability 比 trained weaver 低 {untrained_gap:.4f}"
    else:
        untrained_clause = (
            f"它读到同一批 retrieved experience，gold probability 反而比 trained weaver 高 {-untrained_gap:.4f}；"
            "这说明未训练 latent 可以偶然推高候选概率，不能只看单一概率指标"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    def last_mean(arm: str, metric: str) -> float:
        items = summary["layers"].get(arm, {}).get(metric, [])
        return float(items[-1]["mean"]) if items else float("nan")

    figures = [
        "candidate_policy_probability.png",
        "next_token_policy_kl.png",
        "layer_hidden_l2_to_no_latent.png",
        "latent_attention_mass_by_layer.png",
        "logit_lens_search_answer_margin.png",
        "logit_lens_gold_distractor_margin.png",
        "generation_prefix_mix.png",
    ]
    lines = [
        "# MemGen / Retrieval-Conditioned Weaver 作用路径对比实验",
        "",
        f"样本数：{summary['n']}。三条主路径共享同一个可见 reasoner prompt、同一批 retrieved experience、同一组候选动作；差别只在 latent memory 的生成方式。",
        "",
        "## 三条路径",
        "",
        "- **MemGen**：训练后的 MemGen weaver 只读取当前 prompt，生成 latent memory。",
        "- **Retrieval + trained weaver**：先从 ExpeL-like memory bank 检索显式经验文本，再把 `prompt + retrieved experience` 输入训练后的 weaver，reasoner 只看到 latent。",
        "- **Retrieval + untrained weaver**：检索文本完全相同，但 weaver/projection/latent query 是未训练初始化。",
        "",
        "## 主要数字",
        "",
        "| Path | Gold prob | Search prob | Gold margin | Search-vs-answer margin | Entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = arms[arm]
        lines.append(
            f"| {ARM_LABELS[arm]} | {item['gold_answer_prob_mean']:.4f} | {item['search_prob_mean']:.4f} | "
            f"{item['gold_vs_distractor_margin_mean']:.4f} | {item['search_vs_answer_margin_mean']:.4f} | {item['next_token_entropy_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 层级作用路径读数",
            "",
            "| Path | Final latent attention | Final hidden L2 | Final search-answer logit margin | Final gold-distractor logit margin |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        lines.append(
            f"| {ARM_LABELS[arm]} | "
            f"{last_mean(arm, 'latent_attention_mass'):.4f} | "
            f"{last_mean(arm, 'hidden_l2_to_no_latent'):.4f} | "
            f"{last_mean(arm, 'search_vs_answer_logit_margin'):.4f} | "
            f"{last_mean(arm, 'gold_vs_distractor_logit_margin'):.4f} |"
        )
    lines.extend(
        [
            "",
            "| Comparison | Next-token KL |",
            "|---|---:|",
        ]
    )
    for key, item in comp.items():
        lines.append(f"| {key} | {item['next_token_kl_mean']:.4f} |")
    lines.extend(
        [
            "",
            "## 图表",
            "",
        ]
    )
    for fig in figures:
        lines.append(f"![{fig}]({(out_dir / fig).resolve()})")
        lines.append("")
    lines.extend(
        [
            "## 结论",
            "",
            f"1. **retrieval-conditioned trained weaver 确实产生了不同于原 MemGen 的 policy shift。** 相比 MemGen，Retrieval + trained weaver 的 gold candidate probability 平均变化 {trained_gain:+.4f}，{margin_clause}；两者 next-token KL 为 {comp.get('memgen_to_expel_trained', {}).get('next_token_kl_mean', float('nan')):.4f}，说明检索经验经过 trained weaver 后不是等价于原始 MemGen latent。",
            f"2. **untrained weaver 是关键负控，但不能简单用 gold probability 判定。** {untrained_clause}。它的 next-token entropy 比 trained weaver 高 {entropy_gap:.4f}，trained-vs-untrained KL 为 {kl_trained_untrained:.4f}。这更像无序扰动，而不是稳定的经验利用。",
            "3. **层级指标比最终答案更能解释作用路径。** latent attention mass 表示 reasoner 哪些层直接读取 latent slots；hidden L2 表示 latent 对决策状态的扰动强度；logit-lens action-key probe 在强制前缀 `<` 后比较 `search` vs `answer`，answer-content probe 在 `<answer> ` 后比较 gold vs distractor。",
            "4. **当前实验仍是 prompt-level augmentation，不等价于完整在线 Search-R1 多轮 agent。** 它回答的是 frozen reasoner 下 memory carrier 如何改变 token policy 和 hidden trajectory，不能单独证明最终多轮检索收益。",
            "",
            "## Raw Files",
            "",
            f"- Raw JSONL: `{raw_path.resolve()}`",
            f"- Summary JSON: `{summary_path.resolve()}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="results/mechanistic_paths/figures")
    parser.add_argument("--summary-output", default="results/mechanistic_paths/mechanistic_summary.json")
    parser.add_argument("--report-output", default="docs/mechanistic_paths_report.md")
    args = parser.parse_args()

    global args_input_for_report
    args_input_for_report = args.input

    setup_style()
    rows = load_rows(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_candidate_policy(summary, out_dir)
    plot_kl(summary, out_dir)
    plot_layer_metric(summary, out_dir, "hidden_l2_to_no_latent", "Hidden-state L2 to no-latent baseline", "layer_hidden_l2_to_no_latent.png")
    plot_layer_metric(summary, out_dir, "latent_attention_mass", "Attention mass on latent slots", "latent_attention_mass_by_layer.png")
    plot_layer_metric(summary, out_dir, "search_vs_answer_logit_margin", "Search - gold answer logit margin", "logit_lens_search_answer_margin.png", include_no_marker=True)
    plot_layer_metric(summary, out_dir, "gold_vs_distractor_logit_margin", "Gold - distractor logit margin", "logit_lens_gold_distractor_margin.png", include_no_marker=True)
    plot_generation_tokens(rows, out_dir)
    write_report(rows, summary, out_dir, Path(args.report_output), Path(args.input), Path(args.summary_output))
    print("SUMMARY_OUTPUT", args.summary_output)
    print("REPORT_OUTPUT", args.report_output)
    print("FIGURE_DIR", out_dir)


if __name__ == "__main__":
    main()
