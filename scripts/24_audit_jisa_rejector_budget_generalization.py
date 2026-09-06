#!/usr/bin/env python3
"""Audit out-of-sample rejection-budget generalization for the JISA Phase-6 analysis.

The Phase-6 operating-point budget is a validation-time selection constraint. Test-set
rejection can legitimately exceed that budget after the operating point has been frozen;
excluding such cases would amount to test-set selection. This script makes that distinction
explicit and summarizes whether validation-selected operating points preserve their
coverage/error constraints on test.

It also applies BH and Bonferroni adjustments to the primary 3% paired bootstrap tests
against max-confidence. No model training or operating-point reselection occurs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/12_jisa_finalization/15_rejector_tradeoff_analysis"
OUT = ROOT / "outputs/12_jisa_finalization/16_rejector_budget_audit"
PRIMARY_BUDGET = 0.03

TEST_CONSTRAINTS = {
    "benign_family_fp_rate": 0.02,
    "benign_reject_rate": 0.10,
    "false_unknown_rate_all_known": 0.05,
    "false_unknown_rate_known_attacks": 0.10,
}


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna()
    if valid.empty:
        return out
    order = valid.sort_values(kind="mergesort")
    m = len(order)
    raw = order.to_numpy(dtype=float) * m / np.arange(1, m + 1, dtype=float)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out.loc[order.index] = adj
    return out


def main() -> None:
    metrics_path = SRC / "budget_selected_case_metrics.csv"
    paired_path = SRC / "paired_vs_max_confidence_1000.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not paired_path.exists():
        raise FileNotFoundError(paired_path)

    selected = pd.read_csv(metrics_path)
    paired = pd.read_csv(paired_path)

    primary = selected[
        (pd.to_numeric(selected["budget"], errors="coerce") - PRIMARY_BUDGET).abs() <= 1e-12
    ].copy()
    primary["validation_selection_budget"] = PRIMARY_BUDGET
    primary["validation_feasible"] = primary["feasible"].astype(bool)

    feasible = primary[primary["validation_feasible"]].copy()
    feasible["test_budget_exceeded"] = (
        pd.to_numeric(feasible["overall_reject_rate"], errors="coerce") > PRIMARY_BUDGET + 1e-12
    )

    violation_cols: list[str] = []
    for metric, cap in TEST_CONSTRAINTS.items():
        flag = f"test_{metric}_exceeded"
        feasible[flag] = pd.to_numeric(feasible[metric], errors="coerce") > float(cap) + 1e-12
        violation_cols.append(flag)
    feasible["test_any_nonbudget_constraint_exceeded"] = feasible[violation_cols].any(axis=1)
    feasible["test_any_selection_constraint_exceeded"] = (
        feasible["test_budget_exceeded"] | feasible["test_any_nonbudget_constraint_exceeded"]
    )

    method_rows = []
    for method, grp_all in primary.groupby("method", sort=True):
        grp = feasible[feasible["method"].astype(str) == str(method)].copy()
        delta = pd.to_numeric(grp.get("delta_unknown_detection_rate_vs_max_confidence"), errors="coerce")
        method_rows.append({
            "method": method,
            "total_cases": int(len(grp_all)),
            "validation_feasible_cases": int(len(grp)),
            "validation_feasible_fraction": float(len(grp) / len(grp_all)) if len(grp_all) else np.nan,
            "test_budget_exceeded_cases": int(grp["test_budget_exceeded"].sum()) if len(grp) else 0,
            "test_any_selection_constraint_exceeded_cases": int(grp["test_any_selection_constraint_exceeded"].sum()) if len(grp) else 0,
            "udr_improved_vs_max_confidence_cases": int((delta > 1e-12).sum()) if len(grp) else 0,
            "udr_worse_vs_max_confidence_cases": int((delta < -1e-12).sum()) if len(grp) else 0,
            "mean_udr_delta_vs_max_confidence": float(delta.mean()) if len(delta.dropna()) else np.nan,
            "mean_test_reject_rate": float(pd.to_numeric(grp["overall_reject_rate"], errors="coerce").mean()) if len(grp) else np.nan,
            "max_test_reject_rate": float(pd.to_numeric(grp["overall_reject_rate"], errors="coerce").max()) if len(grp) else np.nan,
        })
    method_summary = pd.DataFrame(method_rows)

    primary_paired = paired[
        (pd.to_numeric(paired["budget"], errors="coerce") - PRIMARY_BUDGET).abs() <= 1e-12
    ].copy()
    for metric in ["unknown_detection", "macro_f1"]:
        pcol = f"delta_{metric}_pvalue_bootstrap"
        if pcol not in primary_paired.columns:
            raise RuntimeError(f"Missing paired-bootstrap column: {pcol}")
        primary_paired[f"{pcol}_bh"] = bh_adjust(primary_paired[pcol])
        primary_paired[f"{pcol}_bonferroni"] = np.minimum(
            1.0,
            pd.to_numeric(primary_paired[pcol], errors="coerce") * len(primary_paired),
        )

    # Compact significance summary for UDR comparisons.
    sig_rows = []
    for method, grp in primary_paired.groupby("method_a", sort=True):
        d = pd.to_numeric(grp["delta_unknown_detection_mean"], errors="coerce")
        p_bh = pd.to_numeric(grp["delta_unknown_detection_pvalue_bootstrap_bh"], errors="coerce")
        p_bonf = pd.to_numeric(grp["delta_unknown_detection_pvalue_bootstrap_bonferroni"], errors="coerce")
        sig_rows.append({
            "method": method,
            "paired_cases": int(len(grp)),
            "positive_udr_delta_cases": int((d > 1e-12).sum()),
            "negative_udr_delta_cases": int((d < -1e-12).sum()),
            "bh_significant_positive_udr_cases": int(((d > 1e-12) & (p_bh < 0.05)).sum()),
            "bh_significant_negative_udr_cases": int(((d < -1e-12) & (p_bh < 0.05)).sum()),
            "bonferroni_significant_positive_udr_cases": int(((d > 1e-12) & (p_bonf < 0.05)).sum()),
            "bonferroni_significant_negative_udr_cases": int(((d < -1e-12) & (p_bonf < 0.05)).sum()),
        })
    significance_summary = pd.DataFrame(sig_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    feasible.sort_values(["dataset", "holdout_family", "method"]).to_csv(
        OUT / "primary_3pct_test_constraint_generalization.csv", index=False
    )
    method_summary.to_csv(OUT / "primary_3pct_method_generalization_summary.csv", index=False)
    primary_paired.sort_values(["dataset", "holdout_family", "method_a"]).to_csv(
        OUT / "primary_3pct_paired_adjusted.csv", index=False
    )
    significance_summary.to_csv(OUT / "primary_3pct_udr_significance_summary.csv", index=False)

    exceeded = feasible[feasible["test_any_selection_constraint_exceeded"]].copy()
    payload = {
        "analysis": "out-of-sample generalization of validation-selected rejector budgets",
        "primary_validation_selection_budget": PRIMARY_BUDGET,
        "selection_boundary": (
            "The 3% rejection budget and other constraints are enforced on validation only. "
            "Test exceedance is retained as an out-of-sample result and is never used to reselect an operating point."
        ),
        "validation_feasible_rows": int(len(feasible)),
        "test_any_selection_constraint_exceeded_rows": int(len(exceeded)),
        "test_budget_exceeded_rows": int(feasible["test_budget_exceeded"].sum()),
        "multiple_comparison_family": int(len(primary_paired)),
        "multiple_comparison_adjustments": ["Benjamini-Hochberg", "Bonferroni"],
        "interpretation_boundary": (
            "Constraint generalization is descriptive. A validation-feasible operating point that exceeds the "
            "budget on test is not removed, because doing so would condition model/rejector selection on test outcomes."
        ),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print("\nPrimary 3% method summary:")
    print(method_summary.to_string(index=False))
    print("\nTest constraint exceedances among validation-feasible operating points:")
    cols = [
        "dataset", "holdout_family", "method", "validation_overall_reject_rate",
        "overall_reject_rate", "false_unknown_rate_all_known", "test_budget_exceeded",
        "test_any_nonbudget_constraint_exceeded",
    ]
    if exceeded.empty:
        print("None")
    else:
        print(exceeded[cols].sort_values(["dataset", "holdout_family", "method"]).to_string(index=False))
    print("\nAdjusted paired UDR significance summary:")
    print(significance_summary.to_string(index=False))
    print(f"\nWrote analysis to: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
