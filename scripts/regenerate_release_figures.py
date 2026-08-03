#!/usr/bin/env python3
"""Regenerate figures affected by the Protocol A primary-metric correction."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "summaries"
FIGURES = ROOT / "figures"


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight", metadata={"Creator": "VERA-IDS v2026.08"})
    plt.close(fig)


def tradeoff(core: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    markers = {"strict": "o", "strict_tau": "s"}
    for _, row in core.iterrows():
        label = f"{row.dataset} {row.model_family.upper()} {row.policy_variant}"
        ax.scatter(
            row.system_macro_f1_supported_labels,
            row.system_accuracy,
            marker=markers[row.policy_variant],
            s=70,
            label=label,
        )
    ax.set_xlabel("System macro-F1 (supported labels)")
    ax.set_ylabel("System accuracy")
    ax.set_title("Protocol A accuracy–macro-F1 trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "accuracy_macro_f1_tradeoff.png")


def deltas(comparison: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = range(len(comparison))
    ax.bar([value - 0.18 for value in x], comparison["delta_macro_f1_supported_labels_vs_strict"], 0.36, label="vs strict")
    ax.bar([value + 0.18 for value in x], comparison["delta_macro_f1_supported_labels_vs_operational"], 0.36, label="vs strict_tau")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x), comparison["dataset"])
    ax.set_ylabel("Macro-F1 delta (supported labels)")
    ax.set_title("Competitive flat baseline vs two-stage Protocol A")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, "baseline_vs_competitive_delta.png")


def leaderboard(core: pd.DataFrame, comparison: pd.DataFrame) -> None:
    rows = []
    for _, row in comparison.iterrows():
        rows.append((row.dataset, "flat competitive", row.competitive_macro_f1))
    for _, row in core.loc[core.policy_variant.isin(["strict", "strict_tau"])].iterrows():
        rows.append((row.dataset, f"{row.model_family.upper()} {row.policy_variant}", row.system_macro_f1_supported_labels))
    frame = pd.DataFrame(rows, columns=["dataset", "candidate", "macro_f1"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    for ax, dataset in zip(axes, sorted(frame.dataset.unique())):
        subset = frame.loc[frame.dataset == dataset].sort_values("macro_f1")
        ax.barh(subset.candidate, subset.macro_f1)
        ax.set_title(dataset)
        ax.set_xlabel("Macro-F1 (supported labels)")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Protocol A corrected leaderboard")
    save(fig, "competitive_validation_leaderboard.png")


def main() -> int:
    core = pd.read_csv(SUMMARY / "protocol_a_core_summary.csv")
    comparison = pd.read_csv(SUMMARY / "protocol_a_flat_vs_two_stage.csv")
    tradeoff(core)
    deltas(comparison)
    leaderboard(core, comparison)
    print("Corrected Protocol A figures regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
