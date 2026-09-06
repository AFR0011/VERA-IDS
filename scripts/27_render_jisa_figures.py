#!/usr/bin/env python3
"""Render sparse journal-style figures from frozen JISA figure-source tables.

Outputs are written under outputs/12_jisa_finalization rather than the protected public
figures/ directory. Each figure is saved as editable SVG, vector PDF, and 300-dpi PNG.
No scientific value is recomputed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/12_jisa_finalization/18_figure_sources"
OUT = ROOT / "outputs/12_jisa_finalization/19_jisa_figures"

FONT_SIZE = 8.5
LINE_WIDTH = 0.9
MARKER_SIZE = 5.2


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def figure1_design() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lanes = [
        (0.76, "(a)", ["Support audit", "Supported split", "Stage 1 gate", "Stage 2 family", "Known metrics"]),
        (0.49, "(b)", ["Support audit", "Family exclusion", "Stage 1 gate", "Stage 2 family", "UDR + sinks"]),
        (0.22, "(c)", ["Validation scores", "Rejector select", "Freeze threshold", "Test replay", "UDR + cost"]),
    ]
    x_positions = np.linspace(0.08, 0.88, 5)
    box_w, box_h = 0.145, 0.105

    for y, panel, labels in lanes:
        ax.text(0.01, y, panel, fontsize=9, fontweight="bold", va="center")
        for i, (x, label) in enumerate(zip(x_positions, labels)):
            rect = Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h, fill=False, linewidth=0.8)
            ax.add_patch(rect)
            ax.text(x, y, label, ha="center", va="center", fontsize=8)
            if i < len(labels) - 1:
                x0 = x + box_w / 2 + 0.008
                x1 = x_positions[i + 1] - box_w / 2 - 0.008
                ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops={"arrowstyle": "->", "lw": 0.8})

    ax.text(0.08, 0.91, "Evaluation condition", ha="center", va="bottom", fontsize=8)
    ax.text(0.48, 0.91, "Two-stage decision", ha="center", va="bottom", fontsize=8)
    ax.text(0.88, 0.91, "Evidence", ha="center", va="bottom", fontsize=8)
    save(fig, "figure1_evaluation_design")


def figure2_known_vs_heldout() -> None:
    df = pd.read_csv(SRC / "fig2_known_vs_heldout.csv")
    datasets = ["CICIDS2017", "CICIoT2023"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharex=True)

    for panel_i, (ax, dataset) in enumerate(zip(axes, datasets)):
        sub = df[df["dataset"].astype(str) == dataset].copy()
        sub = sub.sort_values("known_recall", ascending=True).reset_index(drop=True)
        y = np.arange(len(sub))
        for i, row in sub.iterrows():
            ax.plot([row["heldout_udr_mean"], row["known_recall"]], [i, i], linewidth=0.8, alpha=0.7)
        ax.scatter(sub["known_recall"], y, marker="o", s=28, facecolors="none", edgecolors="black", linewidths=0.8, label="Known recall")
        ax.errorbar(
            sub["heldout_udr_mean"],
            y,
            xerr=sub["heldout_udr_errorbar"],
            fmt="s",
            markersize=4.5,
            linewidth=0.8,
            capsize=2,
            color="black",
            label="Held-out UDR",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(sub["family"])
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Recognition rate")
        ax.grid(axis="x", linewidth=0.4, alpha=0.3)
        clean_axes(ax)
        add_panel_label(ax, f"({'ab'[panel_i]}) {dataset}")

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="black", label="Known recall"),
        Line2D([0], [0], marker="s", linestyle="none", color="black", label="Held-out UDR"),
    ]
    axes[1].legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout(w_pad=2.0)
    save(fig, "figure2_known_vs_heldout")


def figure3_surface_sensitivity() -> None:
    df = pd.read_csv(SRC / "fig3_cicids_surface_sensitivity.csv").copy()
    df = df.sort_values("historical_unknown_detection_rate", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(5.3, 3.6))
    for i, row in df.iterrows():
        ax.plot(
            [row["historical_unknown_detection_rate"], row["new_unknown_detection_rate"]],
            [i, i],
            linewidth=1.0,
            alpha=0.75,
        )
    ax.scatter(df["historical_unknown_detection_rate"], y, marker="o", s=28, facecolors="none", edgecolors="black", linewidths=0.8)
    ax.scatter(df["new_unknown_detection_rate"], y, marker="s", s=24, color="black")
    ax.set_yticks(y)
    ax.set_yticklabels(df["holdout_family"])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Unknown detection rate")
    ax.grid(axis="x", linewidth=0.4, alpha=0.3)
    clean_axes(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="black", label="Recovered surface"),
            Line2D([0], [0], marker="s", linestyle="none", color="black", label="Grouped surface"),
        ],
        loc="lower right",
        frameon=False,
    )
    fig.tight_layout()
    save(fig, "figure3_cicids_surface_sensitivity")


def _heatmap_panel(ax: plt.Axes, df: pd.DataFrame, dataset: str, panel: str):
    sub = df[df["dataset"].astype(str) == dataset].copy()
    rows = sorted(sub["holdout_family"].astype(str).unique().tolist())
    cols = sorted(sub["destination"].astype(str).unique().tolist())
    pivot = sub.pivot_table(index="holdout_family", columns="destination", values="residual_destination_share_mean", aggfunc="mean").reindex(index=rows, columns=cols).fillna(0.0)
    arr = pivot.to_numpy(dtype=float)
    im = ax.imshow(arr, vmin=0, vmax=1, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            if arr[r, c] >= 0.10:
                ax.text(c, r, f"{arr[r,c]:.2f}", ha="center", va="center", fontsize=6.8, color="white" if arr[r,c] >= 0.55 else "black")
    ax.set_xlabel("Residual predicted label")
    ax.set_ylabel("Held-out family")
    add_panel_label(ax, f"({panel}) {dataset}")
    return im


def figure4_failure_destinations() -> None:
    df = pd.read_csv(SRC / "fig4_failure_destination_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.1))
    im = _heatmap_panel(axes[0], df, "CICIDS2017", "a")
    _heatmap_panel(axes[1], df, "CICIoT2023", "b")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Share of residual failures")
    fig.subplots_adjust(left=0.10, right=0.92, bottom=0.23, top=0.91, wspace=0.42)
    save(fig, "figure4_failure_destinations")


def figure5_rejector_tradeoff() -> None:
    df = pd.read_csv(SRC / "fig5_rejector_tradeoff_primary_3pct.csv").copy()
    df = df[df["method"].astype(str) != "max_confidence"].copy()
    methods = ["margin", "entropy", "family_conditional"]
    labels = {"margin": "Margin", "entropy": "Entropy", "family_conditional": "Family-conditional"}

    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.7), sharex=True, sharey=True)
    dataset_markers = {"CICIDS2017": "o", "CICIoT2023": "s"}

    for idx, (ax, method) in enumerate(zip(axes, methods)):
        sub = df[df["method"].astype(str) == method].copy()
        for _, row in sub.iterrows():
            x = 100.0 * float(row["delta_overall_reject_rate_vs_max_confidence"])
            y = 100.0 * float(row["delta_unknown_detection_rate_vs_max_confidence"])
            marker = dataset_markers[str(row["dataset"])]
            ax.scatter(x, y, marker=marker, s=28, facecolors="none", edgecolors="black", linewidths=0.8)
            if bool(row.get("test_any_selection_constraint_exceeded", False)):
                ax.scatter(x, y, marker="x", s=24, color="black", linewidths=0.8)
            ax.annotate(str(row["holdout_family"]), (x, y), xytext=(3, 2), textcoords="offset points", fontsize=6.3)
        ax.axhline(0, linewidth=0.6, color="black", alpha=0.5)
        ax.axvline(0, linewidth=0.6, color="black", alpha=0.5)
        ax.set_xlabel("Δ test rejection (pp)")
        ax.grid(linewidth=0.35, alpha=0.25)
        clean_axes(ax)
        add_panel_label(ax, f"({'abc'[idx]}) {labels[method]}")

    axes[0].set_ylabel("Δ held-out UDR (pp)")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="black", label="CICIDS2017"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor="black", label="CICIoT2023"),
        Line2D([0], [0], marker="x", linestyle="none", color="black", label="Test constraint exceeded"),
    ]
    axes[2].legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout(w_pad=1.4)
    save(fig, "figure5_rejector_tradeoff")


def write_caption_notes() -> None:
    text = """# JISA figure caption notes

## Figure 1
Evaluation conditions used to distinguish supported-family performance, held-out-family behavior, and validation-selected rejection. The schematic describes the evaluation logic and inference boundaries; it is not presented as a novel classifier architecture.

## Figure 2
Known-family recall versus held-out-family Unknown detection under the fixed RF class-balanced profile. CICIoT2023 error bars show standard deviation across seeds 123-127; CICIDS2017 uses the frozen seed-123 group-safe sensitivity surface. Family-level comparisons are descriptive.

## Figure 3
CICIDS2017 held-out-family UDR under the recovered contiguous-within-day surface and the exact-representation-grouped sensitivity partition. The split change alters partition composition as well as eliminating exact postprocessed-feature sharing, so the paired shifts are not attributed causally to duplicate removal.

## Figure 4
Distribution of residual non-Unknown destinations for held-out-family observations. Cell values are conditional on a held-out observation not being predicted Unknown. CICIoT2023 cells average the fixed RF profile across five seeds. Destination concentration is a model-behavior diagnostic and does not imply semantic or causal equivalence between families.

## Figure 5
Change in held-out-family UDR versus change in test rejection rate relative to maximum-confidence rejection after validation-only operating-point selection under the primary 3% validation budget. Crosses mark operating points that exceeded at least one corresponding selection constraint on test. The 3% budget is a validation-time selection constraint, not a guaranteed test rejection rate.
"""
    (OUT / "FIGURE_CAPTION_NOTES.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    if not SRC.exists():
        raise FileNotFoundError(f"Missing figure-source directory: {SRC}. Run scripts/26_build_jisa_figure_sources.py first.")
    figure1_design()
    figure2_known_vs_heldout()
    figure3_surface_sensitivity()
    figure4_failure_destinations()
    figure5_rejector_tradeoff()
    write_caption_notes()
    payload = {
        "status": "JISA_VECTOR_FIGURES_RENDERED",
        "figures": [
            "figure1_evaluation_design",
            "figure2_known_vs_heldout",
            "figure3_cicids_surface_sensitivity",
            "figure4_failure_destinations",
            "figure5_rejector_tradeoff",
        ],
        "formats": ["svg", "pdf", "png"],
        "source_dir": str(SRC.relative_to(ROOT)),
        "output_dir": str(OUT.relative_to(ROOT)),
    }
    (OUT / "figure_render_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
