#!/usr/bin/env python3
"""Render conclusion-focused figures for the policy-fidelity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


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


def setup_style() -> None:
    for font_path in [
        "/root/.local/share/fonts/Times_New_Roman.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    ]:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 450,
        }
    )


def stat(summary: dict, arm: str, key: str) -> float:
    value = summary["arms"][arm].get(key)
    return float(value) if value is not None else 0.0


def save_policy_fidelity_map(summary: dict, out_dir: Path) -> Path:
    arms = ["no_memory", "extractive_text", "summary_text", "latent_compressed", "random_latent", "fixed_soft_prompt"]
    x = [stat(summary, a, "candidate_kl_full_to_arm_mean") for a in arms]
    y = [stat(summary, a, "margin_delta_vs_no_mean") for a in arms]

    label_offsets = {
        "summary_text": (12, 0),
        "extractive_text": (12, 12),
        "no_memory": (12, -2),
        "latent_compressed": (-116, 16),
        "random_latent": (-24, 22),
        "fixed_soft_prompt": (12, -26),
    }

    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    for arm, xi, yi in zip(arms, x, y):
        ax.scatter(
            xi,
            yi,
            s=160,
            color=PALETTE[arm],
            edgecolor="#222222",
            linewidth=1.0,
            zorder=3,
        )
        dx, dy = label_offsets.get(arm, (8, 5))
        ax.annotate(
            ARM_LABELS[arm],
            (xi, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=13,
        )

    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.axvline(0, color="#444444", linewidth=0.9)
    ax.set_xlim(-0.10, 2.22)
    ax.set_ylim(-2.05, 0.82)
    ax.set_xlabel("Candidate-policy KL from full memory")
    ax.set_ylabel("Gold margin delta vs no memory")
    ax.set_title("Policy fidelity vs. action preference")
    ax.grid(alpha=0.22, linewidth=0.8)
    ax.text(
        0.02,
        0.98,
        "Better: lower KL, higher margin",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="#333333",
    )
    path = out_dir / "conclusion_policy_fidelity_map.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_recovery_ladder(summary: dict, out_dir: Path) -> Path:
    arms = ["no_memory", "extractive_text", "summary_text", "latent_compressed", "random_latent", "fixed_soft_prompt", "full_text"]
    vals = [stat(summary, a, "margin_recovery_ratio_mean") for a in arms]
    se = [stat(summary, a, "margin_recovery_ratio_se") for a in arms]
    order = np.argsort(vals)
    arms = [arms[i] for i in order]
    vals = [vals[i] for i in order]
    se = [se[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    y = np.arange(len(arms))
    ax.barh(
        y,
        vals,
        xerr=[1.96 * s for s in se],
        color=[PALETTE[a] for a in arms],
        edgecolor="#222222",
        linewidth=0.8,
        capsize=3,
    )
    ax.axvline(0, color="#444444", linewidth=0.9)
    ax.axvline(1, color="#444444", linewidth=0.9, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_LABELS[a] for a in arms])
    ax.set_xlabel("Gold-margin recovery ratio")
    ax.set_title("How much full-memory action preference is retained")
    ax.grid(axis="x", alpha=0.22, linewidth=0.8)
    path = out_dir / "conclusion_recovery_ladder.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_latent_control_panel(summary: dict, out_dir: Path) -> Path:
    arms = ["latent_compressed", "random_latent", "fixed_soft_prompt"]
    metrics = [
        ("candidate_kl_full_to_arm_mean", "KL from full", "Lower is better"),
        ("gold_prob_mean", "Gold prob.", "Higher is better"),
        ("margin_recovery_ratio_mean", "Recovery", "Higher is better"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.8))
    for ax, (metric, title, direction) in zip(axes, metrics):
        vals = [stat(summary, a, metric) for a in arms]
        x = np.arange(len(arms))
        ax.bar(
            x,
            vals,
            color=[PALETTE[a] for a in arms],
            edgecolor="#222222",
            linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABELS[a].replace(" ", "\n") for a in arms])
        ax.set_title(title)
        ax.set_ylabel(direction)
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        if metric == "margin_recovery_ratio_mean":
            ax.axhline(0, color="#444444", linewidth=0.9)
    fig.suptitle("Latent carrier controls", y=0.98)
    path = out_dir / "conclusion_latent_controls.png"
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/policy_fidelity/policy_fidelity_summary.json")
    parser.add_argument("--out-dir", default="results/policy_fidelity/figures")
    args = parser.parse_args()

    setup_style()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        save_policy_fidelity_map(summary, out_dir),
        save_recovery_ladder(summary, out_dir),
        save_latent_control_panel(summary, out_dir),
    ]
    for path in paths:
        print("FIGURE", path)


if __name__ == "__main__":
    main()
