#!/usr/bin/env python3
"""Build common-case validation-visible rejector summaries for JISA Phase 5E.

This is a descriptive reporting cleanup only. It does not fit models, select new
thresholds, resample rows, or run hypothesis tests. Method comparisons at the
primary 3% budget use the same joint-feasible holdout set. Cross-budget aggregate
curves are produced only when one fixed holdout set is feasible for every method at
every budget.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    REPO_ROOT
    / "outputs"
    / "12_jisa_finalization"
    / "15_rejector_tradeoff_analysis"
    / "budget_selected_case_metrics.csv"
)
OUT = REPO_ROOT / "outputs" / "13_jisa_q1_revision" / "phase5_visible_common_case"

PRIMARY_BUDGET = 0.03
TOL = 1e-12
METHODS = ["max_confidence", "margin", "entropy", "family_conditional"]
CONTROL = "max_confidence"

METHOD_MAP = {
    "control_tau": "max_confidence",
    "max_confidence": "max_confidence",
    "max-confidence": "max_confidence",
    "margin_reject": "margin",
    "margin": "margin",
    "entropy_reject": "entropy",
    "entropy": "entropy",
    "conformal_reject": "family_conditional",
    "family_conditional": "family_conditional",
    "family-conditional": "family_conditional",
}

METRIC_CANDIDATES = [
    "unknown_detection_rate",
    "macro_f1",
    "false_unknown_rate_all_known",
    "false_unknown_rate_known_attacks",
    "overall_reject_rate",
    "benign_reject_rate",
    "benign_family_fp_rate",
    "accuracy",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def normalize_holdout(value: object) -> str:
    text = str(value).strip()
    aliases = {
        "Scan_Recon": "Scan/Recon",
        "Scan-Recon": "Scan/Recon",
        "Brute Force": "BruteForce",
        "Web_App": "Web/App",
        "Web-App": "Web/App",
    }
    return aliases.get(text, text)


def normalize_method(value: object) -> str:
    key = str(value).strip().lower()
    return METHOD_MAP.get(key, str(value).strip())


def load_source() -> tuple[pd.DataFrame, list[str]]:
    if not SOURCE.exists():
        raise RuntimeError(f"Missing frozen Phase-5E source surface: {SOURCE}")
    df = pd.read_csv(SOURCE)
    required = {"dataset", "holdout_family", "method", "budget", "feasible"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Source surface missing required columns: {missing}")

    df = df.copy()
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["holdout_family"] = df["holdout_family"].map(normalize_holdout)
    df["method_original"] = df["method"].astype(str)
    df["method"] = df["method"].map(normalize_method)
    unknown_methods = sorted(set(df["method"].astype(str)) - set(METHODS))
    if unknown_methods:
        raise RuntimeError(f"Unexpected rejector methods after normalization: {unknown_methods}")

    df["budget"] = pd.to_numeric(df["budget"], errors="raise")
    df["feasible_bool"] = df["feasible"].map(as_bool)

    metrics = [m for m in METRIC_CANDIDATES if m in df.columns]
    if not metrics:
        raise RuntimeError("No expected test metrics found on Phase-5E source surface")
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df, metrics


def require_unique_cells(df: pd.DataFrame, keys: list[str], label: str) -> None:
    counts = df.groupby(keys, dropna=False).size()
    bad = counts[counts != 1]
    if not bad.empty:
        raise RuntimeError(f"{label} does not have one row per {keys}: {bad.head(20).to_dict()}")


def primary_surface(df: pd.DataFrame) -> pd.DataFrame:
    p = df[np.isclose(df["budget"].to_numpy(dtype=float), PRIMARY_BUDGET, atol=TOL, rtol=0.0)].copy()
    if p.empty:
        raise RuntimeError("No 3% budget rows found in Phase-5E source surface")
    require_unique_cells(p, ["dataset", "holdout_family", "method"], "Primary 3% surface")

    for (dataset, holdout), g in p.groupby(["dataset", "holdout_family"], sort=True):
        got = set(g["method"].astype(str))
        if got != set(METHODS):
            raise RuntimeError(
                f"Primary 3% case {dataset}/{holdout} is not method-complete: got={sorted(got)} expected={METHODS}"
            )
    return p


def primary_coverage(p: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, method), g in p.groupby(["dataset", "method"], sort=True):
        total = int(g["holdout_family"].nunique())
        feasible_holdouts = sorted(g.loc[g["feasible_bool"], "holdout_family"].astype(str).unique().tolist())
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "n_holdouts_total": total,
                "n_holdouts_feasible": len(feasible_holdouts),
                "feasibility_fraction": float(len(feasible_holdouts) / total) if total else float("nan"),
                "feasible_holdouts": "|".join(feasible_holdouts),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "method"]).reset_index(drop=True)


def joint_primary_cases(p: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, holdout), g in p.groupby(["dataset", "holdout_family"], sort=True):
        methods = set(g["method"].astype(str))
        all_methods_present = methods == set(METHODS)
        all_feasible = bool(all_methods_present and g["feasible_bool"].all())
        rows.append(
            {
                "dataset": dataset,
                "holdout_family": holdout,
                "n_methods": int(len(methods)),
                "all_methods_present": all_methods_present,
                "all_methods_validation_feasible_3pct": all_feasible,
                "included_in_primary_common_case": all_feasible,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "holdout_family"]).reset_index(drop=True)


def common_primary_summary(
    p: pd.DataFrame, joint: pd.DataFrame, metrics: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    included = joint[joint["included_in_primary_common_case"] == True][  # noqa: E712
        ["dataset", "holdout_family"]
    ].copy()
    common = p.merge(included, on=["dataset", "holdout_family"], how="inner", validate="many_to_one")

    summary_rows: list[dict[str, Any]] = []
    for (dataset, method), g in common.groupby(["dataset", "method"], sort=True):
        rec: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "n_common_holdouts": int(g["holdout_family"].nunique()),
            "common_holdouts": "|".join(sorted(g["holdout_family"].astype(str).unique().tolist())),
            "denominator": "joint validation-feasible holdouts at 3% shared by all four methods",
        }
        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)
            rec[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            rec[f"{metric}_sd_across_holdouts"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            )
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["dataset", "method"]).reset_index(drop=True)

    delta_rows: list[dict[str, Any]] = []
    for dataset, dg in common.groupby("dataset", sort=True):
        ctrl = dg[dg["method"] == CONTROL][["holdout_family", *metrics]].copy()
        ctrl = ctrl.rename(columns={m: f"control_{m}" for m in metrics})
        for method in [m for m in METHODS if m != CONTROL]:
            mg = dg[dg["method"] == method][["holdout_family", *metrics]].copy()
            paired = mg.merge(ctrl, on="holdout_family", how="inner", validate="one_to_one")
            for _, row in paired.iterrows():
                rec: dict[str, Any] = {
                    "dataset": dataset,
                    "holdout_family": row["holdout_family"],
                    "method": method,
                    "control_method": CONTROL,
                }
                for metric in metrics:
                    rec[f"delta_{metric}_vs_max_confidence"] = float(row[metric] - row[f"control_{metric}"])
                delta_rows.append(rec)
    deltas = pd.DataFrame(delta_rows)
    if not deltas.empty:
        deltas = deltas.sort_values(["dataset", "method", "holdout_family"]).reset_index(drop=True)
    return summary, deltas


def budget_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, budget, method), g in df.groupby(["dataset", "budget", "method"], sort=True):
        total = int(g["holdout_family"].nunique())
        feasible = sorted(g.loc[g["feasible_bool"], "holdout_family"].astype(str).unique().tolist())
        rows.append(
            {
                "dataset": dataset,
                "budget": float(budget),
                "method": method,
                "n_holdouts_total": total,
                "n_holdouts_feasible": len(feasible),
                "feasibility_fraction": float(len(feasible) / total) if total else float("nan"),
                "feasible_holdouts": "|".join(feasible),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "budget", "method"]).reset_index(drop=True)


def global_budget_common_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    rows: list[dict[str, Any]] = []
    budgets_by_dataset: dict[str, list[float]] = {}
    for dataset, dg in df.groupby("dataset", sort=True):
        budgets = sorted(float(x) for x in dg["budget"].dropna().unique().tolist())
        budgets_by_dataset[str(dataset)] = budgets
        expected_cells = len(budgets) * len(METHODS)
        for holdout, hg in dg.groupby("holdout_family", sort=True):
            unique_cells = hg[["budget", "method"]].drop_duplicates()
            complete = int(len(unique_cells)) == expected_cells
            one_per_cell = not hg.duplicated(subset=["budget", "method"]).any()
            all_feasible = bool(complete and one_per_cell and hg["feasible_bool"].all())
            rows.append(
                {
                    "dataset": dataset,
                    "holdout_family": holdout,
                    "n_budgets": len(budgets),
                    "budgets": "|".join(f"{b:.12g}" for b in budgets),
                    "expected_method_budget_cells": expected_cells,
                    "observed_unique_method_budget_cells": int(len(unique_cells)),
                    "complete_method_budget_grid": bool(complete and one_per_cell),
                    "all_methods_all_budgets_validation_feasible": all_feasible,
                    "included_in_fixed_budget_denominator": all_feasible,
                }
            )
    out = pd.DataFrame(rows).sort_values(["dataset", "holdout_family"]).reset_index(drop=True)
    return out, budgets_by_dataset


def fixed_budget_summary(
    df: pd.DataFrame, fixed_cases: pd.DataFrame, metrics: list[str]
) -> pd.DataFrame:
    included = fixed_cases[fixed_cases["included_in_fixed_budget_denominator"] == True][  # noqa: E712
        ["dataset", "holdout_family"]
    ].copy()
    fixed = df.merge(included, on=["dataset", "holdout_family"], how="inner", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for (dataset, budget, method), g in fixed.groupby(["dataset", "budget", "method"], sort=True):
        rec: dict[str, Any] = {
            "dataset": dataset,
            "budget": float(budget),
            "method": method,
            "n_fixed_common_holdouts": int(g["holdout_family"].nunique()),
            "fixed_common_holdouts": "|".join(sorted(g["holdout_family"].astype(str).unique().tolist())),
            "denominator": "same holdouts feasible for all four methods at every reported budget",
        }
        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)
            rec[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            rec[f"{metric}_sd_across_holdouts"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            )
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["dataset", "budget", "method"]).reset_index(drop=True)
    return out


def main() -> int:
    df, metrics = load_source()
    require_unique_cells(df, ["dataset", "holdout_family", "budget", "method"], "Full budget surface")

    p = primary_surface(df)
    coverage = primary_coverage(p)
    joint = joint_primary_cases(p)
    primary_summary, primary_deltas = common_primary_summary(p, joint, metrics)
    budget_cov = budget_coverage(df)
    fixed_cases, budgets_by_dataset = global_budget_common_cases(df)
    budget_summary = fixed_budget_summary(df, fixed_cases, metrics)

    OUT.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(OUT / "primary_3pct_feasibility_coverage.csv", index=False)
    joint.to_csv(OUT / "primary_3pct_joint_feasible_cases.csv", index=False)
    primary_summary.to_csv(OUT / "primary_3pct_common_case_method_summary.csv", index=False)
    primary_deltas.to_csv(OUT / "primary_3pct_common_case_paired_deltas.csv", index=False)
    fixed_cases.to_csv(OUT / "budget_global_common_cases.csv", index=False)
    budget_summary.to_csv(OUT / "budget_fixed_denominator_summary.csv", index=False)
    budget_cov.to_csv(OUT / "budget_feasibility_coverage.csv", index=False)

    primary_counts = {
        str(dataset): int(g["included_in_primary_common_case"].astype(bool).sum())
        for dataset, g in joint.groupby("dataset", sort=True)
    }
    fixed_counts = {
        str(dataset): int(g["included_in_fixed_budget_denominator"].astype(bool).sum())
        for dataset, g in fixed_cases.groupby("dataset", sort=True)
    }
    fixed_curve_authorized = {dataset: count > 0 for dataset, count in fixed_counts.items()}

    manifest = {
        "status": "built_from_frozen_validation_visible_case_metrics",
        "source": str(SOURCE.relative_to(REPO_ROOT)),
        "source_sha256": sha256_file(SOURCE),
        "primary_budget": PRIMARY_BUDGET,
        "methods": METHODS,
        "control_method": CONTROL,
        "metrics": metrics,
        "budgets_by_dataset": budgets_by_dataset,
        "primary_joint_feasible_case_counts": primary_counts,
        "global_fixed_budget_case_counts": fixed_counts,
        "aggregate_budget_curve_authorized_by_dataset": fixed_curve_authorized,
        "new_model_training": False,
        "new_threshold_selection": False,
        "new_statistical_tests": False,
        "new_resampling": False,
        "test_labels_used_for_new_selection": False,
        "feasibility_denominator_rule": "validation feasibility only; test constraint exceedance is a separate generalization diagnostic",
        "old_unequal_feasible_subset_means_authorized_for_direct_method_comparison": False,
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("Primary 3% feasibility coverage:")
    print(coverage.to_string(index=False))
    print("")
    print("Primary 3% joint-feasible common cases:")
    print(joint.to_string(index=False))
    print("")
    print("Primary 3% common-case method summary:")
    view_metrics = [m for m in ["unknown_detection_rate", "macro_f1", "false_unknown_rate_all_known", "overall_reject_rate"] if m in metrics]
    view_cols = ["dataset", "method", "n_common_holdouts"] + [f"{m}_mean" for m in view_metrics]
    if primary_summary.empty:
        print("  NONE")
    else:
        print(primary_summary[view_cols].to_string(index=False))
    print("")
    print("Fixed cross-budget denominator:")
    for dataset in sorted(fixed_counts):
        print(
            f"  {dataset}: n_fixed_common_holdouts={fixed_counts[dataset]} "
            f"budgets={budgets_by_dataset.get(dataset, [])} "
            f"aggregate_curve_authorized={fixed_curve_authorized[dataset]}"
        )
    print(f"output: {OUT}")
    print("PASS=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
