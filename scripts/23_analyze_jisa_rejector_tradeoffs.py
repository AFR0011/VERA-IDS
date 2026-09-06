#!/usr/bin/env python3
"""Analyze JISA rejector trade-offs from the completed canonical replay scores.

This script does not retrain models. It reuses the validation candidate grids and saved
validation/test score frames produced by the canonical Phase-6 rejector replay.

Why this post-processing exists
-------------------------------
The native open-set selector historically hard-coded a 10% overall-rejection ceiling,
while an older policy YAML recorded a 1% ceiling. For the journal-final analysis we avoid
silently inheriting either value. Each method is reselected on validation under common
rejection budgets of 1%, 3%, 5%, and 10%. The 3% budget is primary because the accepted
journal-final Protocol-B configurations for both datasets use a 3% maximum overall
rejection constraint. The remaining budgets are sensitivity points.

Selection remains validation-only. Test labels are used only after one operating point
per method/budget/case has been frozen from validation metrics.

Methods are reported as:
  * max_confidence      (native control_tau)
  * margin              (native margin_reject)
  * entropy             (native entropy_reject)
  * family_conditional  (native conformal_reject; conformal-inspired only)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ids_eval_framework._native import protocol_b_grid, protocol_b_open_set

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/12_jisa_finalization/14_rejector_tradeoff_runs"
OUT = ROOT / "outputs/12_jisa_finalization/15_rejector_tradeoff_analysis"
PRIMARY_BUDGET = 0.03
BUDGETS = [0.01, 0.03, 0.05, 0.10]
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 123

METHOD_LABELS = {
    "control_tau": "max_confidence",
    "margin_reject": "margin",
    "entropy_reject": "entropy",
    "conformal_reject": "family_conditional",
}

OTHER_CONSTRAINTS = {
    "benign_family_fp_rate": 0.02,
    "benign_reject_rate": 0.10,
    "false_unknown_rate_all_known": 0.05,
    "false_unknown_rate_known_attacks": 0.10,
}


def parse_json_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    return dict(json.loads(text)) if text else {}


def normalize_param_json(value: object) -> str:
    return json.dumps(parse_json_dict(value), sort_keys=True, separators=(",", ":"))


def select_validation_candidate(grid: pd.DataFrame, budget: float) -> tuple[pd.Series | None, int]:
    work = grid.copy()
    numeric_cols = [
        "unknown_detection_rate",
        "macro_f1",
        "false_unknown_rate_all_known",
        "false_unknown_rate_known_attacks",
        "benign_family_fp_rate",
        "benign_reject_rate",
        "overall_reject_rate",
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    mask = work["overall_reject_rate"] <= float(budget) + 1e-12
    for col, cap in OTHER_CONSTRAINTS.items():
        mask &= work[col] <= float(cap) + 1e-12
    feasible = work.loc[mask].copy()
    if feasible.empty:
        return None, 0

    feasible["_param_norm"] = feasible["selection_param_json"].map(normalize_param_json)
    feasible = feasible.sort_values(
        ["unknown_detection_rate", "macro_f1", "false_unknown_rate_all_known", "_param_norm"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return feasible.iloc[0], int(len(feasible))


def case_dirs() -> list[Path]:
    plan = RUN_ROOT / "scenario_plan.csv"
    if not plan.exists():
        raise FileNotFoundError(plan)
    df = pd.read_csv(plan)
    if len(df) != 12:
        raise RuntimeError(f"Expected 12 canonical rejector cases, found {len(df)}")
    dirs: list[Path] = []
    for _, row in df.sort_values(["dataset", "holdout_family"]).iterrows():
        profile = str(row.get("model_profile", "") or "").strip()
        profile_token = f"__{protocol_b_open_set.sanitize_token(profile)}" if profile else ""
        run_dir = RUN_ROOT / (
            f"{row['dataset']}__holdout_{protocol_b_open_set.sanitize_token(row['holdout_family'])}"
            f"{profile_token}__winner_replay"
        )
        dirs.append(run_dir)
    return dirs


def evaluate_selected_case(run_dir: Path, case_idx: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    needed = ["scenario_manifest.json", "winner_reference.json", "method_grid.csv", "test_scores.csv.gz"]
    for name in needed:
        path = run_dir / name
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = json.loads((run_dir / "scenario_manifest.json").read_text(encoding="utf-8"))
    winner = json.loads((run_dir / "winner_reference.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(run_dir / "method_grid.csv")
    test_scores = pd.read_csv(run_dir / "test_scores.csv.gz")

    dataset = str(winner["dataset"])
    holdout = str(winner["holdout_family"])
    profile = str(winner.get("model_profile", winner.get("model_family", "")))
    families = list(manifest["valid_known_families"])
    unknown_label = str(manifest.get("unknown_label", "Unknown"))

    selected_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    for budget_idx, budget in enumerate(BUDGETS):
        selected_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        selected_metrics: dict[str, dict[str, object]] = {}
        selection_records: dict[str, dict[str, object]] = {}

        for native_method, report_method in METHOD_LABELS.items():
            method_grid = grid[grid["method"].astype(str) == native_method].copy()
            if method_grid.empty:
                raise RuntimeError(f"{run_dir}: missing grid rows for {native_method}")
            best, feasible_count = select_validation_candidate(method_grid, budget)
            if best is None:
                selected_rows.append({
                    "dataset": dataset,
                    "holdout_family": holdout,
                    "model_profile": profile,
                    "budget": budget,
                    "is_primary_budget": bool(abs(budget - PRIMARY_BUDGET) <= 1e-12),
                    "method": report_method,
                    "native_method": native_method,
                    "feasible": False,
                    "feasible_validation_candidates": 0,
                })
                continue

            params = parse_json_dict(best["selection_param_json"])
            thr_high = float(best["thr_high"])
            pred, reject, uncertainty = protocol_b_open_set.predict_for_method(
                test_scores, native_method, thr_high, families, unknown_label, params
            )
            metrics = protocol_b_open_set.evaluate_predictions(
                protocol_b_grid, test_scores, pred, reject, uncertainty, families, unknown_label
            )
            y_true = test_scores["y_true_sys"].astype(str).to_numpy(dtype=object)
            ci = protocol_b_open_set.bootstrap_ci(
                protocol_b_grid,
                y_true,
                pred,
                reject,
                families,
                unknown_label,
                BOOTSTRAP_RESAMPLES,
                BOOTSTRAP_SEED + case_idx * 100 + budget_idx,
            )
            record = {
                "dataset": dataset,
                "holdout_family": holdout,
                "model_profile": profile,
                "budget": budget,
                "is_primary_budget": bool(abs(budget - PRIMARY_BUDGET) <= 1e-12),
                "method": report_method,
                "native_method": native_method,
                "feasible": True,
                "feasible_validation_candidates": feasible_count,
                "thr_high": thr_high,
                "selection_param_json": json.dumps(params, sort_keys=True),
                "validation_unknown_detection_rate": float(best["unknown_detection_rate"]),
                "validation_macro_f1": float(best["macro_f1"]),
                "validation_false_unknown_rate_all_known": float(best["false_unknown_rate_all_known"]),
                "validation_overall_reject_rate": float(best["overall_reject_rate"]),
                **metrics,
                **ci,
            }
            selected_rows.append(record)
            selected_predictions[report_method] = (pred.copy(), reject.copy())
            selected_metrics[report_method] = metrics
            selection_records[report_method] = record

        # Pair every feasible rejector against max-confidence selected under the same budget.
        if "max_confidence" in selected_predictions:
            control_pred, control_reject = selected_predictions["max_confidence"]
            y_true = test_scores["y_true_sys"].astype(str).to_numpy(dtype=object)
            for report_method, (pred, reject) in selected_predictions.items():
                if report_method == "max_confidence":
                    continue
                diff = protocol_b_open_set.paired_bootstrap_diff(
                    protocol_b_grid,
                    y_true,
                    pred,
                    reject,
                    control_pred,
                    control_reject,
                    families,
                    unknown_label,
                    BOOTSTRAP_RESAMPLES,
                    BOOTSTRAP_SEED + 50000 + case_idx * 100 + budget_idx,
                )
                paired_rows.append({
                    "dataset": dataset,
                    "holdout_family": holdout,
                    "model_profile": profile,
                    "budget": budget,
                    "is_primary_budget": bool(abs(budget - PRIMARY_BUDGET) <= 1e-12),
                    "method_a": report_method,
                    "method_b": "max_confidence",
                    **diff,
                })

    return selected_rows, paired_rows


def summarize(selected: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "unknown_detection_rate",
        "false_unknown_rate_all_known",
        "overall_reject_rate",
        "coverage",
        "macro_f1",
    ]
    rows: list[dict[str, object]] = []
    feasible = selected[selected["feasible"].astype(bool)].copy()
    for (dataset, budget, method), grp in feasible.groupby(["dataset", "budget", "method"], sort=True):
        row: dict[str, object] = {
            "dataset": dataset,
            "budget": float(budget),
            "method": method,
            "n_feasible_cases": int(len(grp)),
        }
        for metric in metrics:
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{metric}_max"] = float(vals.max()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_control_deltas(selected: pd.DataFrame) -> pd.DataFrame:
    out = selected.copy()
    feasible = out[out["feasible"].astype(bool)].copy()
    control = feasible[feasible["method"] == "max_confidence"][
        ["dataset", "holdout_family", "budget", "unknown_detection_rate", "macro_f1", "false_unknown_rate_all_known", "overall_reject_rate"]
    ].rename(columns={
        "unknown_detection_rate": "control_unknown_detection_rate",
        "macro_f1": "control_macro_f1",
        "false_unknown_rate_all_known": "control_false_unknown_rate_all_known",
        "overall_reject_rate": "control_overall_reject_rate",
    })
    out = out.merge(control, on=["dataset", "holdout_family", "budget"], how="left", validate="many_to_one")
    for metric in ["unknown_detection_rate", "macro_f1", "false_unknown_rate_all_known", "overall_reject_rate"]:
        out[f"delta_{metric}_vs_max_confidence"] = (
            pd.to_numeric(out.get(metric), errors="coerce") - pd.to_numeric(out.get(f"control_{metric}"), errors="coerce")
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []

    dirs = case_dirs()
    for case_idx, run_dir in enumerate(dirs, start=1):
        case_selected, case_paired = evaluate_selected_case(run_dir, case_idx)
        selected_rows.extend(case_selected)
        paired_rows.extend(case_paired)

    selected = pd.DataFrame(selected_rows)
    paired = pd.DataFrame(paired_rows)
    selected = add_control_deltas(selected)
    summary = summarize(selected)

    selected.to_csv(OUT / "budget_selected_case_metrics.csv", index=False)
    paired.to_csv(OUT / "paired_vs_max_confidence_1000.csv", index=False)
    summary.to_csv(OUT / "method_summary_by_dataset_budget.csv", index=False)

    primary = selected[(selected["is_primary_budget"].astype(bool)) & (selected["feasible"].astype(bool))].copy()
    primary.to_csv(OUT / "primary_3pct_case_metrics.csv", index=False)

    primary_paired = paired[paired["is_primary_budget"].astype(bool)].copy() if not paired.empty else paired
    primary_paired.to_csv(OUT / "primary_3pct_paired_vs_max_confidence.csv", index=False)

    feasibility = (
        selected.groupby(["budget", "method"], sort=True)["feasible"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "feasible_cases", "count": "total_cases"})
    )
    feasibility["feasible_fraction"] = feasibility["feasible_cases"] / feasibility["total_cases"]
    feasibility.to_csv(OUT / "feasibility_by_budget_method.csv", index=False)

    improvement = primary[primary["method"] != "max_confidence"].copy()
    improvement_counts = {}
    for method, grp in improvement.groupby("method", sort=True):
        d = pd.to_numeric(grp["delta_unknown_detection_rate_vs_max_confidence"], errors="coerce")
        improvement_counts[str(method)] = {
            "feasible_cases": int(len(grp)),
            "udr_improved_cases": int((d > 1e-12).sum()),
            "udr_tied_cases": int((d.abs() <= 1e-12).sum()),
            "udr_worse_cases": int((d < -1e-12).sum()),
            "mean_udr_delta": float(d.mean()) if len(d) else np.nan,
        }

    payload = {
        "analysis": "validation-selected rejector trade-offs under common rejection budgets",
        "cases": 12,
        "budgets": BUDGETS,
        "primary_budget": PRIMARY_BUDGET,
        "primary_budget_rationale": "matches the 3% maximum overall rejection constraint in both journal-final Protocol-B configurations",
        "selection_discipline": "validation only; test metrics are evaluated only after operating-point selection",
        "selection_objective": "maximize validation UDR, then validation macro-F1, then minimize validation false-Unknown rate, with deterministic parameter tie-break",
        "constraints": {**OTHER_CONSTRAINTS, "overall_reject_rate": "budget-specific"},
        "methods": list(METHOD_LABELS.values()),
        "family_conditional_note": "conformal-inspired family-conditional thresholding; no formal conformal coverage guarantee is claimed",
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "primary_improvement_counts_vs_max_confidence": improvement_counts,
        "interpretation_boundary": "Rejection is evaluated as a UDR-versus-known-traffic-cost trade-off; no rejector is treated as universally best.",
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print("\nFeasibility by budget/method:")
    print(feasibility.to_string(index=False))
    print("\nPrimary 3% case metrics:")
    cols = [
        "dataset", "holdout_family", "method", "unknown_detection_rate",
        "false_unknown_rate_all_known", "overall_reject_rate", "coverage", "macro_f1",
        "delta_unknown_detection_rate_vs_max_confidence",
    ]
    print(primary[cols].sort_values(["dataset", "holdout_family", "method"]).to_string(index=False))
    print(f"\nWrote analysis to: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
