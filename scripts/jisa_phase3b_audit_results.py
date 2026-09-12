#!/usr/bin/env python3
"""Audit the completed JISA Phase-3B heldout-validation-blind rejector surface.

Read-only with respect to scientific outputs. Verifies all 30 seed x holdout cases,
all four rejectors x four budgets, validation-only feasibility constraints, manifest
selection flags, the expected 5%/10% ceiling equivalence, and consistency of the
stored five-seed summary with case-level selected points.

Audit artifacts are written only under .release-audit/jisa_phase3b_results/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "jisa_phase3_validation_blind.yml"
AUDIT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase3b_results"
METHODS = ["max_confidence", "margin", "entropy", "family_conditional"]
TOL = 1e-10


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def finite01(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x) and -TOL <= x <= 1.0 + TOL


def params_signature(row: pd.Series) -> str:
    method = str(row.get("method", ""))
    if method == "family_conditional":
        return f"alpha={row.get('alpha', '')}|thr={row.get('family_thresholds_json', '')}"
    return f"threshold={row.get('uncertainty_threshold', '')}"


def check_feasible_row(row: pd.Series, rcfg: dict[str, Any]) -> tuple[bool, str]:
    budget = float(row["budget"])
    checks = {
        "budget": float(row["val_overall_reject_rate"]) <= budget + TOL,
        "benign_family_fp": float(row["val_benign_family_fp_rate"]) <= float(rcfg["max_benign_family_fp_rate"]) + TOL,
        "benign_reject": float(row["val_benign_reject_rate"]) <= float(rcfg["max_benign_reject_rate"]) + TOL,
        "fur_all_known": float(row["val_false_unknown_rate_all_known"]) <= float(rcfg["max_false_unknown_rate_all_known"]) + TOL,
        "fur_known_attacks": float(row["val_false_unknown_rate_known_attacks"]) <= float(rcfg["max_false_unknown_rate_known_attacks"]) + TOL,
    }
    bad = [k for k, ok in checks.items() if not ok]
    return not bad, "ok" if not bad else "|".join(bad)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_yaml(args.config.resolve())
    exp = cfg["experiment"]
    rcfg = cfg["rejector_replay"]
    seeds = [int(x) for x in exp["seeds"]]
    holdouts = [str(x) for x in exp["holdouts"]]
    budgets = [float(x) for x in rcfg["budgets"]]
    primary = float(rcfg["primary_budget"])
    root = REPO_ROOT / str(exp["output_root"])
    replay_root = root / "rejector_replay"
    summary_root = root / "summary"

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    all_frames: list[pd.DataFrame] = []

    expected_pairs = {(m, b) for m in METHODS for b in budgets}

    for seed in seeds:
        for holdout in holdouts:
            case_dir = replay_root / f"seed_{seed}" / holdout.replace("/", "_")
            csv_path = case_dir / "selected_points.csv"
            manifest_path = case_dir / "case_manifest.json"
            rec: dict[str, Any] = {
                "seed": seed,
                "holdout_family": holdout,
                "case_dir": str(case_dir),
                "csv_exists": csv_path.exists(),
                "manifest_exists": manifest_path.exists(),
            }
            if not (csv_path.exists() and manifest_path.exists()):
                rec["PASS"] = False
                rec["reason"] = "missing_case_artifacts"
                case_rows.append(rec)
                continue

            df = pd.read_csv(csv_path)
            manifest = load_json(manifest_path)
            all_frames.append(df.copy())

            rec["row_count"] = int(len(df))
            rec["row_count_ok"] = len(df) == len(METHODS) * len(budgets)
            pairs = {(str(r.method), float(r.budget)) for r in df[["method", "budget"]].itertuples(index=False)}
            rec["method_budget_matrix_ok"] = pairs == expected_pairs and len(pairs) == len(df)
            rec["manifest_ok"] = bool(
                str(manifest.get("dataset")) == str(exp["dataset"])
                and int(manifest.get("seed", -1)) == seed
                and str(manifest.get("holdout_family")) == holdout
                and manifest.get("heldout_validation_used_for_rejector_selection") is False
                and manifest.get("test_used_for_rejector_selection") is False
                and abs(float(manifest.get("primary_budget", -1.0)) - primary) <= TOL
                and [float(x) for x in manifest.get("budgets", [])] == budgets
                and set(str(x) for x in manifest.get("methods", [])) == set(METHODS)
            )

            feasible_mask = df["feasible"].map(as_bool)
            rec["feasible_count"] = int(feasible_mask.sum())
            feasible_ok = True
            bad_reasons: list[str] = []
            test_metrics_ok = True
            for idx, row in df.loc[feasible_mask].iterrows():
                ok, reason = check_feasible_row(row, rcfg)
                if not ok:
                    feasible_ok = False
                    bad_reasons.append(f"row{idx}:{reason}")
                for col in [
                    "test_unknown_detection_rate",
                    "test_false_unknown_rate_all_known",
                    "test_false_unknown_rate_known_attacks",
                    "test_overall_reject_rate",
                    "test_benign_reject_rate",
                    "test_benign_family_fp_rate",
                    "test_macro_f1_supported",
                    "test_accuracy",
                ]:
                    if col not in df.columns or not finite01(row.get(col)):
                        test_metrics_ok = False
            rec["validation_constraints_ok"] = feasible_ok
            rec["validation_constraint_failures"] = ";".join(bad_reasons)
            rec["test_metrics_ok"] = test_metrics_ok

            # Because validation contains only known traffic and all-known FUR is capped at 5%,
            # the 10% nominal budget must select the same maximum-rejection point as 5%.
            ceiling_ok = True
            for method in METHODS:
                g5 = df[(df["method"].astype(str) == method) & np.isclose(pd.to_numeric(df["budget"], errors="coerce"), 0.05)]
                g10 = df[(df["method"].astype(str) == method) & np.isclose(pd.to_numeric(df["budget"], errors="coerce"), 0.10)]
                if len(g5) != 1 or len(g10) != 1:
                    ceiling_ok = False
                    continue
                a, b = g5.iloc[0], g10.iloc[0]
                if as_bool(a["feasible"]) != as_bool(b["feasible"]):
                    ceiling_ok = False
                    continue
                if as_bool(a["feasible"]):
                    if abs(float(a["val_overall_reject_rate"]) - float(b["val_overall_reject_rate"])) > TOL:
                        ceiling_ok = False
                    if params_signature(a) != params_signature(b):
                        ceiling_ok = False
            rec["five_ten_ceiling_equivalence_ok"] = ceiling_ok

            checks = [
                rec["row_count_ok"],
                rec["method_budget_matrix_ok"],
                rec["manifest_ok"],
                rec["validation_constraints_ok"],
                rec["test_metrics_ok"],
                rec["five_ten_ceiling_equivalence_ok"],
            ]
            rec["PASS"] = bool(all(checks))
            rec["reason"] = "ok" if rec["PASS"] else "case_check_failed"
            case_rows.append(rec)

    case_df = pd.DataFrame(case_rows).sort_values(["holdout_family", "seed"])
    case_df.to_csv(AUDIT_ROOT / "case_audit.csv", index=False)

    expected_cases = len(seeds) * len(holdouts)
    passed_cases = int(case_df["PASS"].fillna(False).sum()) if not case_df.empty else 0

    all_df = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    summary_consistent = False
    primary_coverage_ok = False
    summary_path = summary_root / "phase3b_five_seed_summary.csv"
    summary_detail: list[dict[str, Any]] = []

    if not all_df.empty:
        primary_df = all_df[np.isclose(pd.to_numeric(all_df["budget"], errors="coerce"), primary)].copy()
        primary_df["_feasible"] = primary_df["feasible"].map(as_bool)
        coverage = primary_df.groupby(["holdout_family", "method"])["_feasible"].sum()
        primary_coverage_ok = bool(len(coverage) == len(holdouts) * len(METHODS) and (coverage == len(seeds)).all())

    if summary_path.exists() and not all_df.empty:
        stored = pd.read_csv(summary_path)
        recomputed_rows: list[dict[str, Any]] = []
        metrics = [
            "test_unknown_detection_rate",
            "test_false_unknown_rate_all_known",
            "test_macro_f1_supported",
        ]
        for (holdout, method, budget), g in all_df.groupby(["holdout_family", "method", "budget"], sort=True):
            fg = g[g["feasible"].map(as_bool)].copy()
            rec = {
                "holdout_family": holdout,
                "method": method,
                "budget": float(budget),
                "n_total": int(len(g)),
                "n_feasible": int(len(fg)),
            }
            for col in metrics:
                vals = pd.to_numeric(fg[col], errors="coerce").dropna()
                rec[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            recomputed_rows.append(rec)
        recomputed = pd.DataFrame(recomputed_rows)
        merged = stored.merge(
            recomputed,
            on=["holdout_family", "method", "budget"],
            how="outer",
            suffixes=("_stored", "_recomputed"),
            indicator=True,
        )
        summary_consistent = bool((merged["_merge"] == "both").all())
        for col in ["n_total", "n_feasible"]:
            if summary_consistent:
                summary_consistent = bool((merged[f"{col}_stored"] == merged[f"{col}_recomputed"]).all())
        for col in metrics:
            if summary_consistent:
                a = pd.to_numeric(merged[f"{col}_mean_stored"], errors="coerce")
                b = pd.to_numeric(merged[f"{col}_mean_recomputed"], errors="coerce")
                summary_consistent = bool(np.allclose(a, b, rtol=0.0, atol=1e-12, equal_nan=True))
        merged.to_csv(AUDIT_ROOT / "summary_consistency.csv", index=False)

    pass_all = bool(
        len(case_df) == expected_cases
        and passed_cases == expected_cases
        and primary_coverage_ok
        and summary_consistent
    )

    payload = {
        "cases_expected": expected_cases,
        "cases_found": int(len(case_df)),
        "cases_passed": passed_cases,
        "primary_budget": primary,
        "primary_all_24_holdout_method_groups_have_five_feasible_seeds": primary_coverage_ok,
        "stored_summary_matches_case_recomputation": summary_consistent,
        "pass": pass_all,
    }
    (AUDIT_ROOT / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"cases: {len(case_df)}/{expected_cases}")
    print(f"case checks passed: {passed_cases}/{expected_cases}")
    print(f"primary 3% groups with 5/5 feasible seeds: {primary_coverage_ok}")
    print(f"stored five-seed summary matches case recomputation: {summary_consistent}")
    if not case_df.empty and passed_cases != expected_cases:
        print(case_df.loc[~case_df["PASS"].fillna(False), ["seed", "holdout_family", "reason", "validation_constraint_failures"]].to_string(index=False))
    print(f"audit: {AUDIT_ROOT / 'audit.json'}")
    print(f"PASS={pass_all}")
    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
