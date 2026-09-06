#!/usr/bin/env python3
"""Build auditable source tables for the journal-final JISA figures.

This script performs no training, model selection, threshold selection, or new inference.
It transforms the frozen analysis outputs into compact plotting tables and records SHA256
hashes of every source artifact used. Rendering is handled separately so that figures can
be regenerated without changing scientific values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/12_jisa_finalization/18_figure_sources"

KNOWN_HELDOUT = ROOT / "outputs/12_jisa_finalization/11_known_vs_heldout/known_vs_heldout_primary_strict.csv"
SURFACE = ROOT / "outputs/12_jisa_finalization/07_group_safe_comparison/matched_surface_comparison.csv"
FAIL_DEST = ROOT / "outputs/12_jisa_finalization/12_failure_destinations/destination_distribution_per_run.csv"
FAIL_SUMMARY = ROOT / "outputs/12_jisa_finalization/12_failure_destinations/failure_structure_primary_fixed_rf.csv"
REJECTOR = ROOT / "outputs/12_jisa_finalization/15_rejector_tradeoff_analysis/primary_3pct_case_metrics.csv"
BUDGET_ALL = ROOT / "outputs/12_jisa_finalization/15_rejector_tradeoff_analysis/budget_selected_case_metrics.csv"
BUDGET_AUDIT = ROOT / "outputs/12_jisa_finalization/16_rejector_budget_audit/primary_3pct_test_constraint_generalization.csv"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def build_known_vs_heldout() -> pd.DataFrame:
    df = pd.read_csv(require(KNOWN_HELDOUT)).copy()
    numeric(df, ["known_recall", "heldout_udr_mean", "heldout_udr_sd"])
    cols = [
        "dataset",
        "family",
        "known_recall",
        "heldout_udr_mean",
        "heldout_udr_sd",
    ]
    out = df[cols].copy()
    out["known_minus_heldout"] = out["known_recall"] - out["heldout_udr_mean"]
    out["heldout_udr_errorbar"] = out["heldout_udr_sd"].fillna(0.0)
    return out.sort_values(["dataset", "family"]).reset_index(drop=True)


def build_surface_sensitivity() -> pd.DataFrame:
    df = pd.read_csv(require(SURFACE)).copy()
    numeric(
        df,
        [
            "historical_unknown_detection_rate",
            "new_unknown_detection_rate",
            "delta_unknown_detection_rate",
            "historical_macro_f1",
            "new_macro_f1",
            "historical_n_true_unknown",
            "new_n_true_unknown",
        ],
    )
    cols = [
        "holdout_family",
        "matched_stage1_weight_mode",
        "historical_unknown_detection_rate",
        "new_unknown_detection_rate",
        "delta_unknown_detection_rate",
        "historical_macro_f1",
        "new_macro_f1",
        "historical_n_true_unknown",
        "new_n_true_unknown",
    ]
    return df[cols].sort_values("holdout_family").reset_index(drop=True)


def build_failure_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    dist = pd.read_csv(require(FAIL_DEST)).copy()
    summary = pd.read_csv(require(FAIL_SUMMARY)).copy()
    dist = dist[dist["analysis_lane"].astype(str) == "fixed_rf_primary"].copy()
    numeric(dist, ["share_conditional_on_nonunknown", "share_of_all_unknown"])

    # Average across repeated seeds only where repeated seeds exist. CICIDS contributes
    # one frozen run per holdout, while CICIoT contributes five.
    matrix = (
        dist.groupby(["dataset", "holdout_family", "destination"], sort=True, as_index=False)
        .agg(
            residual_destination_share_mean=("share_conditional_on_nonunknown", "mean"),
            residual_destination_share_sd=("share_conditional_on_nonunknown", "std"),
            share_of_all_unknown_mean=("share_of_all_unknown", "mean"),
            n_runs=("seed", "count"),
        )
    )
    matrix["residual_destination_share_sd"] = matrix["residual_destination_share_sd"].fillna(0.0)

    summary_cols = [
        "dataset",
        "holdout_family",
        "n_runs",
        "modal_dominant_destination",
        "dominant_destination_stability_fraction",
        "unknown_detection_rate_mean",
        "dominant_destination_share_of_failures_mean",
        "destination_concentration_mean",
    ]
    compact = summary[summary_cols].copy()
    return (
        matrix.sort_values(["dataset", "holdout_family", "destination"]).reset_index(drop=True),
        compact.sort_values(["dataset", "holdout_family"]).reset_index(drop=True),
    )


def build_rejector_primary() -> pd.DataFrame:
    df = pd.read_csv(require(REJECTOR)).copy()
    numeric(
        df,
        [
            "unknown_detection_rate",
            "false_unknown_rate_all_known",
            "overall_reject_rate",
            "coverage",
            "macro_f1",
            "validation_overall_reject_rate",
            "delta_unknown_detection_rate_vs_max_confidence",
            "delta_overall_reject_rate_vs_max_confidence",
            "delta_false_unknown_rate_all_known_vs_max_confidence",
        ],
    )
    audit = pd.read_csv(require(BUDGET_AUDIT)).copy()
    audit_cols = [
        "dataset",
        "holdout_family",
        "method",
        "test_budget_exceeded",
        "test_any_nonbudget_constraint_exceeded",
        "test_any_selection_constraint_exceeded",
    ]
    df = df.merge(audit[audit_cols], on=["dataset", "holdout_family", "method"], how="left", validate="one_to_one")
    for col in audit_cols[3:]:
        df[col] = df[col].fillna(False).astype(bool)

    cols = [
        "dataset",
        "holdout_family",
        "model_profile",
        "method",
        "unknown_detection_rate",
        "false_unknown_rate_all_known",
        "overall_reject_rate",
        "coverage",
        "macro_f1",
        "validation_overall_reject_rate",
        "delta_unknown_detection_rate_vs_max_confidence",
        "delta_overall_reject_rate_vs_max_confidence",
        "delta_false_unknown_rate_all_known_vs_max_confidence",
        "test_budget_exceeded",
        "test_any_nonbudget_constraint_exceeded",
        "test_any_selection_constraint_exceeded",
    ]
    return df[cols].sort_values(["dataset", "holdout_family", "method"]).reset_index(drop=True)


def build_budget_sensitivity() -> pd.DataFrame:
    df = pd.read_csv(require(BUDGET_ALL)).copy()
    df = df[df["feasible"].astype(bool)].copy()
    numeric(df, ["budget", "unknown_detection_rate", "overall_reject_rate", "false_unknown_rate_all_known", "coverage", "macro_f1"])
    rows = []
    for (dataset, budget, method), grp in df.groupby(["dataset", "budget", "method"], sort=True):
        rows.append(
            {
                "dataset": dataset,
                "validation_budget": float(budget),
                "method": method,
                "n_feasible_cases": int(len(grp)),
                "mean_test_udr": float(grp["unknown_detection_rate"].mean()),
                "median_test_udr": float(grp["unknown_detection_rate"].median()),
                "mean_test_reject_rate": float(grp["overall_reject_rate"].mean()),
                "mean_test_false_unknown_rate": float(grp["false_unknown_rate_all_known"].mean()),
                "mean_test_coverage": float(grp["coverage"].mean()),
                "mean_test_macro_f1": float(grp["macro_f1"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    fig2 = build_known_vs_heldout()
    fig3 = build_surface_sensitivity()
    fig4_matrix, fig4_summary = build_failure_matrix()
    fig5 = build_rejector_primary()
    fig5_budget = build_budget_sensitivity()

    outputs = {
        "fig2_known_vs_heldout.csv": fig2,
        "fig3_cicids_surface_sensitivity.csv": fig3,
        "fig4_failure_destination_matrix.csv": fig4_matrix,
        "fig4_failure_destination_summary.csv": fig4_summary,
        "fig5_rejector_tradeoff_primary_3pct.csv": fig5,
        "fig5_rejector_budget_sensitivity.csv": fig5_budget,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)

    sources = [KNOWN_HELDOUT, SURFACE, FAIL_DEST, FAIL_SUMMARY, REJECTOR, BUDGET_ALL, BUDGET_AUDIT]
    source_manifest = {
        "status": "FIGURE_SOURCES_FROZEN_FROM_EDITORIAL_LEDGER",
        "scientific_reanalysis": False,
        "source_hashes_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "figure_sources": {
            "Figure 1": {
                "type": "design schematic",
                "source": "frozen experimental design and claim ledger; no quantitative plotting table",
                "purpose": "show evaluation conditions and inference boundaries, not model architecture novelty",
            },
            "Figure 2": {
                "table": "fig2_known_vs_heldout.csv",
                "purpose": "compare supported-family recall with held-out-family Unknown detection under a fixed RF profile",
                "boundary": "family-level comparison is descriptive; CICIoT error bars are seed SD, not confidence intervals",
            },
            "Figure 3": {
                "table": "fig3_cicids_surface_sensitivity.csv",
                "purpose": "show family-level UDR shifts between recovered and exact-representation-grouped CICIDS surfaces",
                "boundary": "do not attribute shifts causally to duplicate removal; partition composition also changes",
            },
            "Figure 4": {
                "tables": ["fig4_failure_destination_matrix.csv", "fig4_failure_destination_summary.csv"],
                "purpose": "show where residual held-out failures are absorbed among known labels",
                "boundary": "destination concentration does not imply semantic or causal similarity",
            },
            "Figure 5": {
                "tables": ["fig5_rejector_tradeoff_primary_3pct.csv", "fig5_rejector_budget_sensitivity.csv"],
                "purpose": "show held-out recovery versus test-time rejection/known-traffic cost after validation-only operating-point selection",
                "boundary": "3% is a validation-time budget, not a guaranteed test rejection rate",
            },
        },
    }
    (OUT / "figure_source_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": source_manifest["status"],
        "figure_source_tables": list(outputs.keys()),
        "source_artifacts_hashed": len(sources),
        "next_stage": "render vector figures from frozen source tables",
    }, indent=2))
    print(f"\nWrote figure sources to: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
