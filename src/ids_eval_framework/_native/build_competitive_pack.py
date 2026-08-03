#!/usr/bin/env python3
"""
21.BuildCompetitiveMetricsPack.py
=================================

Purpose
-------
Assemble the competitive-metrics reporting layer from the separate
`runs_competitive_metrics/` evidence lane.

This script does not train models. It summarizes closed-set paper-style
performance separately from the thesis-safe baseline/open-set claims.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from ids_eval_framework.src.paths import REPO_ROOT

ROOT = REPO_ROOT

CFG: dict[str, Any] = {
    "runs_root": ROOT / "runs_competitive_metrics",
    "baseline_pack": ROOT / "thesis_full_scope_pack",
    "out_root": ROOT / "competitive_metrics_pack",
}

SMOKE_CFG: dict[str, Any] = {
    "runs_root": ROOT / "runs_competitive_metrics_smoke",
    "out_root": ROOT / "competitive_metrics_pack_smoke",
}


TABLES: list[tuple[str, str, str]] = [
    ("C01", "competitive_results_all.csv", "All competitive validation and winner-test rows"),
    ("C02", "competitive_validation_leaderboard.csv", "Validation leaderboard used for model selection"),
    ("C03", "competitive_winner_test_results.csv", "Winner-only test results"),
    ("C04", "baseline_vs_competitive_summary.csv", "Thesis baseline versus competitive winner deltas"),
    ("C05", "metric_surface_comparison.csv", "Strict, operational, and literature-comparable metric surfaces"),
    ("C06", "binary_metrics_summary.csv", "Binary benign/attack metrics for competitive winners"),
    ("C07", "per_family_f1_summary.csv", "Per-family precision, recall, and F1 from selected reports"),
    ("C08", "runtime_resource_costs.csv", "Fit/prediction time and resource-cost summary"),
    ("C09", "literature_comparability_audit.csv", "Cross-paper comparability guardrail audit"),
    ("C10", "competitive_claim_status.csv", "Claim status labels for each competitive surface"),
]

FIGURES: list[tuple[str, str, str]] = [
    ("CF01", "competitive_validation_leaderboard.png", "Validation macro-F1 leaderboard"),
    ("CF02", "per_family_f1_heatmap.png", "Per-family F1 heatmap for selected reports"),
    ("CF03", "baseline_vs_competitive_delta.png", "Baseline versus competitive metric deltas"),
    ("CF04", "accuracy_macro_f1_tradeoff.png", "Accuracy and macro-F1 tradeoff scatter"),
]


def mkdirs() -> None:
    (CFG["out_root"] / "tables").mkdir(parents=True, exist_ok=True)
    (CFG["out_root"] / "figures").mkdir(parents=True, exist_ok=True)


def apply_smoke_overrides() -> None:
    if os.environ.get("IDS_COMPETITIVE_SMOKE", "").strip() not in {"1", "true", "True", "yes"}:
        return
    for key, value in SMOKE_CFG.items():
        CFG[key] = value


def apply_runtime_overrides() -> None:
    if os.environ.get("IDS_COMPETITIVE_RUNS_ROOT", "").strip():
        CFG["runs_root"] = ROOT / os.environ["IDS_COMPETITIVE_RUNS_ROOT"].strip()
    if os.environ.get("IDS_COMPETITIVE_PACK_ROOT", "").strip():
        CFG["out_root"] = ROOT / os.environ["IDS_COMPETITIVE_PACK_ROOT"].strip()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, name: str) -> Path:
    path = CFG["out_root"] / "tables" / name
    df.to_csv(path, index=False)
    return path


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def load_competitive_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = CFG["runs_root"] / "summary"
    all_rows = read_csv(summary / "competitive_results_all.csv")
    val = read_csv(summary / "validation_leaderboard.csv")
    test = read_csv(summary / "winner_test_results.csv")
    return all_rows, val, test


def load_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = read_csv(CFG["baseline_pack"] / "protocol_a_anchor_summary.csv")
    operational = read_csv(CFG["baseline_pack"] / "protocol_a_operational_summary.csv")
    return anchor, operational


def best_rows(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["system_macro_f1_supported_labels"] = numeric(out, "system_macro_f1_supported_labels")
    out["system_accuracy"] = numeric(out, "system_accuracy")
    out = out.sort_values(
        ["dataset", "system_macro_f1_supported_labels", "system_accuracy"],
        ascending=[True, False, False],
    )
    out = out.drop_duplicates("dataset", keep="first")
    return pd.DataFrame(
        {
            "dataset": out["dataset"],
            "surface": source,
            "candidate": out["model_family"].astype(str) + "_" + out["policy_variant"].astype(str),
            "claim_status": "thesis-safe",
            "accuracy": out["system_accuracy"],
            "macro_f1": out["system_macro_f1_supported_labels"],
            "weighted_f1": np.nan,
            "binary_attack_f1": np.nan,
            "benign_family_fp_rate": numeric(out, "benign_family_fp_rate"),
            "overall_reject_rate": numeric(out, "overall_reject_rate"),
        }
    )


def build_baseline_vs_competitive(anchor: pd.DataFrame, operational: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    baseline = pd.concat(
        [
            best_rows(anchor, "thesis_strict_baseline"),
            best_rows(operational, "thesis_operational_overlay"),
        ],
        ignore_index=True,
    )
    if test.empty:
        return baseline
    comp = test.copy()
    comp["macro_f1"] = numeric(comp, "macro_f1")
    comp["accuracy"] = numeric(comp, "accuracy")
    comp = comp.sort_values(["dataset", "macro_f1", "accuracy"], ascending=[True, False, False])
    comp = comp.drop_duplicates("dataset", keep="first")
    comp_small = pd.DataFrame(
        {
            "dataset": comp["dataset"],
            "surface": "competitive_winner",
            "candidate": comp["surface"].astype(str) + ":" + comp["candidate"].astype(str),
            "claim_status": comp.get("claim_status", "literature-comparable"),
            "accuracy": comp["accuracy"],
            "macro_f1": comp["macro_f1"],
            "weighted_f1": numeric(comp, "weighted_f1"),
            "binary_attack_f1": numeric(comp, "binary_attack_f1"),
            "benign_family_fp_rate": numeric(comp, "benign_family_fp_rate"),
            "overall_reject_rate": numeric(comp, "overall_reject_rate"),
        }
    )
    rows: List[dict[str, object]] = []
    for dataset in sorted(set(comp_small["dataset"].astype(str))):
        c = comp_small.loc[comp_small["dataset"].astype(str) == dataset].iloc[0]
        b = baseline.loc[
            (baseline["dataset"].astype(str) == dataset)
            & (baseline["surface"].astype(str) == "thesis_strict_baseline")
        ]
        o = baseline.loc[
            (baseline["dataset"].astype(str) == dataset)
            & (baseline["surface"].astype(str) == "thesis_operational_overlay")
        ]
        b_row = b.iloc[0] if not b.empty else None
        o_row = o.iloc[0] if not o.empty else None
        rows.append(
            {
                "dataset": dataset,
                "competitive_candidate": c["candidate"],
                "competitive_claim_status": c["claim_status"],
                "competitive_accuracy": c["accuracy"],
                "competitive_macro_f1": c["macro_f1"],
                "competitive_weighted_f1": c["weighted_f1"],
                "strict_baseline_accuracy": None if b_row is None else b_row["accuracy"],
                "strict_baseline_macro_f1": None if b_row is None else b_row["macro_f1"],
                "operational_accuracy": None if o_row is None else o_row["accuracy"],
                "operational_macro_f1": None if o_row is None else o_row["macro_f1"],
                "delta_macro_f1_vs_strict": None if b_row is None else float(c["macro_f1"]) - float(b_row["macro_f1"]),
                "delta_accuracy_vs_strict": None if b_row is None else float(c["accuracy"]) - float(b_row["accuracy"]),
                "delta_macro_f1_vs_operational": None if o_row is None else float(c["macro_f1"]) - float(o_row["macro_f1"]),
                "delta_accuracy_vs_operational": None if o_row is None else float(c["accuracy"]) - float(o_row["accuracy"]),
            }
        )
    return pd.DataFrame(rows)


def build_metric_surface(anchor: pd.DataFrame, operational: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    pieces = [best_rows(anchor, "thesis_strict_baseline"), best_rows(operational, "thesis_operational_overlay")]
    if not test.empty:
        comp = test.copy()
        pieces.append(
            pd.DataFrame(
                {
                    "dataset": comp["dataset"],
                    "surface": comp["surface"],
                    "candidate": comp["candidate"],
                    "claim_status": comp.get("claim_status", "literature-comparable"),
                    "accuracy": numeric(comp, "accuracy"),
                    "macro_f1": numeric(comp, "macro_f1"),
                    "weighted_f1": numeric(comp, "weighted_f1"),
                    "binary_attack_f1": numeric(comp, "binary_attack_f1"),
                    "benign_family_fp_rate": numeric(comp, "benign_family_fp_rate"),
                    "overall_reject_rate": numeric(comp, "overall_reject_rate"),
                }
            )
        )
    return pd.concat([p for p in pieces if not p.empty], ignore_index=True) if pieces else pd.DataFrame()


def build_binary_metrics(test: pd.DataFrame) -> pd.DataFrame:
    if test.empty:
        return pd.DataFrame()
    keep = [
        "dataset",
        "surface",
        "candidate",
        "claim_status",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "binary_attack_f1",
        "binary_macro_f1",
        "benign_family_fp_rate",
        "attack_to_benign_rate",
        "overall_reject_rate",
    ]
    cols = [c for c in keep if c in test.columns]
    return test[cols].copy()


def iter_report_paths() -> Iterable[tuple[str, str, str, Path]]:
    root = CFG["runs_root"]
    if not root.exists():
        return
    for dataset_dir in root.iterdir():
        if not dataset_dir.is_dir() or dataset_dir.name == "summary":
            continue
        direct_root = dataset_dir / "direct_multiclass"
        if direct_root.exists():
            for report in direct_root.glob("*/test_classification_report.csv"):
                yield dataset_dir.name, "direct_multiclass", report.parent.name, report
        overlay_root = dataset_dir / "two_stage_overlay"
        if overlay_root.exists():
            for report in overlay_root.glob("*/test_selected_classification_report.csv"):
                yield dataset_dir.name, "two_stage_overlay", report.parent.name, report


def build_per_family() -> pd.DataFrame:
    rows: List[dict[str, object]] = []
    skip = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}
    for dataset, surface, candidate, path in iter_report_paths() or []:
        report = pd.read_csv(path, index_col=0)
        for label, rec in report.iterrows():
            if str(label) in skip:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "surface": surface,
                    "candidate": candidate,
                    "label": str(label),
                    "precision": rec.get("precision"),
                    "recall": rec.get("recall"),
                    "f1_score": rec.get("f1-score"),
                    "support": rec.get("support"),
                    "report_path": str(path),
                }
            )
    return pd.DataFrame(rows)


def build_runtime(all_rows: pd.DataFrame) -> pd.DataFrame:
    if all_rows.empty:
        return pd.DataFrame()
    out = all_rows.copy()
    for col in ["fit_seconds", "predict_seconds", "rss_mb", "n_train_rows", "n_eval_rows", "n_features"]:
        out[col] = numeric(out, col)
    group_cols = ["dataset", "surface", "candidate", "claim_status"]
    agg = (
        out.groupby(group_cols, dropna=False)
        .agg(
            total_fit_seconds=("fit_seconds", "max"),
            total_predict_seconds=("predict_seconds", "sum"),
            max_rss_mb=("rss_mb", "max"),
            max_train_rows=("n_train_rows", "max"),
            max_eval_rows=("n_eval_rows", "max"),
            n_features=("n_features", "max"),
            n_rows=("dataset", "size"),
        )
        .reset_index()
    )
    return agg


def build_literature_audit() -> pd.DataFrame:
    rows = [
        {
            "source": "CICIoT2023 original benchmark",
            "dataset": "CICIoT2023",
            "url": "https://www.mdpi.com/1424-8220/23/13/5941",
            "task_surface": "benchmark ML classification pipeline",
            "comparability_note": "Use only against matching class granularity and split protocol; do not compare open-set thesis holdouts to closed-set benchmark numbers.",
            "claim_use": "metric context",
        },
        {
            "source": "CICIDS2017 traffic-flow ML evaluation",
            "dataset": "CICIDS2017",
            "url": "https://www.mdpi.com/1424-8220/22/23/9326",
            "task_surface": "flow-based multiclass/binary ML evaluation",
            "comparability_note": "High tree-based scores are common, but flow extraction, feature policy, and split design materially change comparability.",
            "claim_use": "guardrail",
        },
        {
            "source": "Recent CICIDS2017 comparability caution",
            "dataset": "CICIDS2017",
            "url": "https://www.mdpi.com/1999-4893/18/12/749",
            "task_surface": "classical and hybrid IDS comparison",
            "comparability_note": "Cross-paper numbers vary with binary versus multiclass setup, curation, train/test protocol, imbalance handling, and leakage controls.",
            "claim_use": "guardrail",
        },
        {
            "source": "Current thesis competitive lane",
            "dataset": "CICIDS2017; CICIoT2023",
            "url": "local:competitive_metrics_pack",
            "task_surface": "processed_V5 A_stratified closed-set multiclass plus two-stage overlay",
            "comparability_note": "Report as a separate competitive lane; do not replace Protocol B/open-set results or claim transfer validation.",
            "claim_use": "local evidence",
        },
    ]
    return pd.DataFrame(rows)


def build_claim_status(all_rows: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "claim_area": "Thesis baseline preservation",
            "status": "preserved",
            "artifact": "thesis_full_scope_pack/protocol_a_anchor_summary.csv",
            "claim_status": "thesis-safe",
            "notes": "Competitive lane is separate from frozen thesis evidence.",
        },
        {
            "claim_area": "Literature-comparable closed-set metrics",
            "status": "available" if not test.empty else "pending",
            "artifact": "competitive_winner_test_results.csv",
            "claim_status": "literature-comparable",
            "notes": "Use for accuracy/F1/Macro-F1 comparison only when task surface and split are compatible.",
        },
        {
            "claim_area": "Two-stage threshold overlay",
            "status": "available" if (not all_rows.empty and (all_rows.get("surface", pd.Series(dtype=str)).astype(str) == "two_stage_overlay").any()) else "pending",
            "artifact": "competitive_results_all.csv",
            "claim_status": "thesis-safe/exploratory",
            "notes": "Closed overlay is thesis-safe; unknown/reject overlay remains exploratory unless promoted separately.",
        },
        {
            "claim_area": "Universal competitive superiority",
            "status": "unsupported",
            "artifact": "literature_comparability_audit.csv",
            "claim_status": "not claimed",
            "notes": "No broad claim is supported without matching external protocols and repeated validation.",
        },
    ]
    return pd.DataFrame(rows)


def plot_placeholder(path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, "No data available yet", ha="center", va="center", fontsize=12)
    plt.title(title)
    plt.axis("off")
    savefig(path)


def plot_leaderboard(val: pd.DataFrame) -> None:
    path = CFG["out_root"] / "figures" / "competitive_validation_leaderboard.png"
    if val.empty or "macro_f1" not in val.columns:
        plot_placeholder(path, "Competitive Validation Leaderboard")
        return
    df = val.copy()
    df["macro_f1"] = numeric(df, "macro_f1")
    df = df.dropna(subset=["macro_f1"]).sort_values("macro_f1", ascending=False).head(14)
    if df.empty:
        plot_placeholder(path, "Competitive Validation Leaderboard")
        return
    labels = df["dataset"].astype(str) + "\n" + df["candidate"].astype(str)
    plt.figure(figsize=(11, max(4, 0.45 * len(df))))
    plt.barh(labels, df["macro_f1"], color="#2A9D8F")
    plt.gca().invert_yaxis()
    plt.xlim(0, 1)
    plt.xlabel("Validation macro-F1")
    plt.title("Competitive Validation Leaderboard")
    savefig(path)


def plot_heatmap(per_family: pd.DataFrame) -> None:
    path = CFG["out_root"] / "figures" / "per_family_f1_heatmap.png"
    if per_family.empty:
        plot_placeholder(path, "Per-Family F1 Heatmap")
        return
    df = per_family.copy()
    df["f1_score"] = numeric(df, "f1_score")
    df["row"] = df["dataset"].astype(str) + " | " + df["surface"].astype(str) + " | " + df["candidate"].astype(str)
    pivot = df.pivot_table(index="row", columns="label", values="f1_score", aggfunc="max")
    if pivot.empty:
        plot_placeholder(path, "Per-Family F1 Heatmap")
        return
    pivot = pivot.sort_index()
    plt.figure(figsize=(max(8, 0.85 * len(pivot.columns)), max(4, 0.55 * len(pivot))))
    plt.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="F1")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Per-Family F1 For Selected Competitive Reports")
    savefig(path)


def plot_deltas(delta: pd.DataFrame) -> None:
    path = CFG["out_root"] / "figures" / "baseline_vs_competitive_delta.png"
    if delta.empty or "delta_macro_f1_vs_strict" not in delta.columns:
        plot_placeholder(path, "Baseline Versus Competitive Deltas")
        return
    df = delta.copy()
    df["delta_macro_f1_vs_strict"] = numeric(df, "delta_macro_f1_vs_strict")
    df["delta_accuracy_vs_strict"] = numeric(df, "delta_accuracy_vs_strict")
    labels = df["dataset"].astype(str)
    x = np.arange(len(df))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.18, df["delta_macro_f1_vs_strict"], width=0.36, label="Macro-F1 delta", color="#457B9D")
    plt.bar(x + 0.18, df["delta_accuracy_vs_strict"], width=0.36, label="Accuracy delta", color="#E76F51")
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.xticks(x, labels)
    plt.ylabel("Delta vs strict baseline")
    plt.title("Competitive Winner Versus Strict Thesis Baseline")
    plt.legend()
    savefig(path)


def plot_tradeoff(all_rows: pd.DataFrame) -> None:
    path = CFG["out_root"] / "figures" / "accuracy_macro_f1_tradeoff.png"
    if all_rows.empty:
        plot_placeholder(path, "Accuracy/Macro-F1 Tradeoff")
        return
    df = all_rows.copy()
    df["accuracy"] = numeric(df, "accuracy")
    df["macro_f1"] = numeric(df, "macro_f1")
    df = df.dropna(subset=["accuracy", "macro_f1"])
    if df.empty:
        plot_placeholder(path, "Accuracy/Macro-F1 Tradeoff")
        return
    plt.figure(figsize=(7, 5))
    for dataset, sub in df.groupby("dataset"):
        plt.scatter(sub["accuracy"], sub["macro_f1"], s=48, label=str(dataset), alpha=0.8)
    plt.xlabel("Accuracy")
    plt.ylabel("Macro-F1")
    plt.title("Competitive Accuracy/Macro-F1 Tradeoff")
    plt.legend()
    savefig(path)


def write_readme(tables: list[dict[str, object]], figures: list[dict[str, object]]) -> None:
    lines = [
        "# Competitive Metrics Pack",
        "",
        "This pack summarizes a separate metric-competition lane. It is intended for accuracy/F1/Macro-F1 comparison against literature only when the task surface, split policy, label granularity, and leakage policy are compatible.",
        "",
        "Do not use these closed-set benchmark numbers to replace Protocol B open-set or support-audited thesis claims.",
        "",
        "## Tables",
    ]
    for row in tables:
        lines.append(f"- `{row['filename']}`: {row['description']}")
    lines.extend(["", "## Figures"])
    for row in figures:
        lines.append(f"- `{row['filename']}`: {row['description']}")
    (CFG["out_root"] / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    rows: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []
    figure_rows: list[dict[str, object]] = []
    for item_id, filename, desc in TABLES:
        path = CFG["out_root"] / "tables" / filename
        row = {
            "item_id": item_id,
            "item_type": "table",
            "filename": filename,
            "description": desc,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        rows.append(row)
        table_rows.append(row)
    for item_id, filename, desc in FIGURES:
        path = CFG["out_root"] / "figures" / filename
        row = {
            "item_id": item_id,
            "item_type": "figure",
            "filename": filename,
            "description": desc,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
        }
        rows.append(row)
        figure_rows.append(row)
    pd.DataFrame(rows).to_csv(CFG["out_root"] / "output_manifest.csv", index=False)
    write_readme(table_rows, figure_rows)


def main() -> None:
    apply_smoke_overrides()
    apply_runtime_overrides()
    mkdirs()
    all_rows, val, test = load_competitive_tables()
    anchor, operational = load_baseline()

    write_table(all_rows, "competitive_results_all.csv")
    write_table(val, "competitive_validation_leaderboard.csv")
    write_table(test, "competitive_winner_test_results.csv")

    delta = build_baseline_vs_competitive(anchor, operational, test)
    surfaces = build_metric_surface(anchor, operational, test)
    binary = build_binary_metrics(test)
    per_family = build_per_family()
    runtime = build_runtime(all_rows)
    literature = build_literature_audit()
    claims = build_claim_status(all_rows, test)

    write_table(delta, "baseline_vs_competitive_summary.csv")
    write_table(surfaces, "metric_surface_comparison.csv")
    write_table(binary, "binary_metrics_summary.csv")
    write_table(per_family, "per_family_f1_summary.csv")
    write_table(runtime, "runtime_resource_costs.csv")
    write_table(literature, "literature_comparability_audit.csv")
    write_table(claims, "competitive_claim_status.csv")

    plot_leaderboard(val)
    plot_heatmap(per_family)
    plot_deltas(delta)
    plot_tradeoff(all_rows)
    write_manifest()
    print(f"Wrote competitive metrics pack: {CFG['out_root']}")


if __name__ == "__main__":
    main()
