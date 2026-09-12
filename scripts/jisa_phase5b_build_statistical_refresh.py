#!/usr/bin/env python3
"""Build the JISA Phase-5B claim-oriented statistical refresh package.

No model fitting, threshold tuning, or row-level resampling occurs here. The builder
uses the frozen Phase-3B five-seed rejector results and the Phase-5A joinability
decision. It produces seed-robustness and paired descriptive summaries at the
primary 3% budget, inventories validation-visible selected-method artifacts, and
records whether a separate cluster-aware Phase 5C is scientifically authorized.
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
JOINABILITY_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase5_joinability"
JOINABILITY_DECISION = JOINABILITY_ROOT / "decision.json"
JOINABILITY_SURFACES = JOINABILITY_ROOT / "surface_joinability.csv"

PHASE3_ROOT = REPO_ROOT / "outputs" / "13_jisa_q1_revision" / "phase3_validation_blind"
PHASE3_CASE_RESULTS = PHASE3_ROOT / "summary" / "phase3b_case_results.csv"
VISIBLE_ROOT = REPO_ROOT / "outputs" / "12_jisa_finalization" / "14_rejector_tradeoff_runs"

OUT = REPO_ROOT / "outputs" / "13_jisa_q1_revision" / "phase5_statistical_refresh"

PRIMARY_BUDGET = 0.03
SEEDS = [123, 124, 125, 126, 127]
HOLDOUTS = ["Botnet", "BruteForce", "DDoS", "DoS", "Other", "Scan/Recon"]
METHODS = ["max_confidence", "margin", "entropy", "family_conditional"]
CONTROL = "max_confidence"
TOL = 1e-12

METRICS = [
    "test_unknown_detection_rate",
    "test_false_unknown_rate_all_known",
    "test_false_unknown_rate_known_attacks",
    "test_overall_reject_rate",
    "test_macro_f1_supported",
    "test_accuracy",
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
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def normalize_holdout(value: object) -> str:
    text = str(value).strip()
    aliases = {
        "Scan_Recon": "Scan/Recon",
        "Scan-Recon": "Scan/Recon",
        "Brute Force": "BruteForce",
    }
    return aliases.get(text, text)


def load_phase5a() -> tuple[dict[str, Any], pd.DataFrame]:
    if not JOINABILITY_DECISION.exists():
        raise RuntimeError(f"Missing Phase-5A decision: {JOINABILITY_DECISION}")
    decision = json.loads(JOINABILITY_DECISION.read_text(encoding="utf-8"))
    if not bool(decision.get("READY_FOR_PHASE5B_SPEC")):
        raise RuntimeError("Phase-5A did not authorize Phase-5B specification/execution.")
    surfaces = pd.read_csv(JOINABILITY_SURFACES) if JOINABILITY_SURFACES.exists() else pd.DataFrame()
    return decision, surfaces


def load_primary_blind() -> pd.DataFrame:
    if not PHASE3_CASE_RESULTS.exists():
        raise RuntimeError(
            f"Missing frozen Phase-3B case-results surface: {PHASE3_CASE_RESULTS}. "
            "Run the frozen Phase-3B summarize step first."
        )
    df = pd.read_csv(PHASE3_CASE_RESULTS)
    required = {"dataset", "seed", "holdout_family", "method", "budget", "feasible", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Phase-3B case-results missing required columns: {missing}")

    df = df[df["dataset"].astype(str) == "CICIoT2023"].copy()
    df["holdout_family"] = df["holdout_family"].map(normalize_holdout)
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["budget"] = pd.to_numeric(df["budget"], errors="raise")
    df = df[np.isclose(df["budget"].to_numpy(dtype=float), PRIMARY_BUDGET, atol=TOL, rtol=0.0)].copy()
    df["feasible_bool"] = df["feasible"].map(as_bool)

    got_seeds = sorted(set(df["seed"]))
    got_holdouts = sorted(set(df["holdout_family"].astype(str)))
    got_methods = sorted(set(df["method"].astype(str)))
    if got_seeds != SEEDS:
        raise RuntimeError(f"Unexpected Phase-3B primary seeds: {got_seeds}; expected={SEEDS}")
    if set(got_holdouts) != set(HOLDOUTS):
        raise RuntimeError(f"Unexpected Phase-3B primary holdouts: {got_holdouts}; expected={HOLDOUTS}")
    if set(got_methods) != set(METHODS):
        raise RuntimeError(f"Unexpected Phase-3B primary methods: {got_methods}; expected={METHODS}")

    expected_rows = len(SEEDS) * len(HOLDOUTS) * len(METHODS)
    if len(df) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} primary Phase-3B rows; found {len(df)}")

    dup = df.groupby(["seed", "holdout_family", "method"]).size()
    if not (dup == 1).all():
        raise RuntimeError(f"Phase-3B primary cells are not unique: {dup[dup != 1].to_dict()}")

    # The frozen Phase-3B audit established 5/5 feasibility for every primary group.
    # Fail closed if the local result surface no longer agrees.
    if not bool(df["feasible_bool"].all()):
        bad = df.loc[~df["feasible_bool"], ["seed", "holdout_family", "method"]]
        raise RuntimeError(f"Primary Phase-3B contains infeasible cells:\n{bad.to_string(index=False)}")

    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if df[col].isna().any():
            raise RuntimeError(f"NaN in primary Phase-3B metric {col}")
    return df


def five_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (holdout, method), g in df.groupby(["holdout_family", "method"], sort=True):
        rec: dict[str, Any] = {
            "holdout_family": holdout,
            "method": method,
            "n_seeds": int(len(g)),
            "n_feasible": int(g["feasible_bool"].sum()),
            "seeds": "|".join(str(x) for x in sorted(g["seed"].astype(int).tolist())),
        }
        for metric in METRICS:
            vals = g[metric].to_numpy(dtype=float)
            rec[f"{metric}_mean"] = float(np.mean(vals))
            rec[f"{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            rec[f"{metric}_min"] = float(np.min(vals))
            rec[f"{metric}_max"] = float(np.max(vals))
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["holdout_family", "method"]).reset_index(drop=True)
    if len(out) != len(HOLDOUTS) * len(METHODS):
        raise RuntimeError(f"Expected 24 five-seed summary rows; found {len(out)}")
    if not (out["n_seeds"] == len(SEEDS)).all():
        raise RuntimeError("At least one holdout/method cell does not contain all five seeds")
    return out


def paired_seed_deltas(df: pd.DataFrame) -> pd.DataFrame:
    control = df[df["method"].astype(str) == CONTROL].copy()
    control_cols = ["seed", "holdout_family"] + METRICS
    control = control[control_cols].rename(columns={m: f"control_{m}" for m in METRICS})

    rows: list[pd.DataFrame] = []
    for method in [m for m in METHODS if m != CONTROL]:
        x = df[df["method"].astype(str) == method].copy()
        merged = x.merge(control, on=["seed", "holdout_family"], how="inner", validate="one_to_one")
        if len(merged) != len(SEEDS) * len(HOLDOUTS):
            raise RuntimeError(f"Incomplete paired seed/control merge for {method}: {len(merged)} rows")
        out = merged[["seed", "holdout_family"]].copy()
        out["method"] = method
        out["control_method"] = CONTROL
        for metric in METRICS:
            out[f"delta_{metric}_vs_max_confidence"] = (
                merged[metric].to_numpy(dtype=float) - merged[f"control_{metric}"].to_numpy(dtype=float)
            )
        rows.append(out)
    return pd.concat(rows, ignore_index=True).sort_values(["holdout_family", "method", "seed"]).reset_index(drop=True)


def paired_delta_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [c for c in deltas.columns if c.startswith("delta_")]
    rows: list[dict[str, Any]] = []
    for (holdout, method), g in deltas.groupby(["holdout_family", "method"], sort=True):
        rec: dict[str, Any] = {"holdout_family": holdout, "method": method, "n_paired_seeds": int(len(g))}
        for col in delta_cols:
            vals = g[col].to_numpy(dtype=float)
            rec[f"{col}_mean"] = float(np.mean(vals))
            rec[f"{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            rec[f"{col}_min"] = float(np.min(vals))
            rec[f"{col}_max"] = float(np.max(vals))
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["holdout_family", "method"]).reset_index(drop=True)
    if len(out) != len(HOLDOUTS) * (len(METHODS) - 1):
        raise RuntimeError(f"Expected 18 paired-delta summary rows; found {len(out)}")
    return out


def equal_holdout_summary(five: pd.DataFrame, delta_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, g in five.groupby("method", sort=True):
        rec: dict[str, Any] = {
            "method": method,
            "n_holdouts_equal_weight": int(len(g)),
            "aggregation": "equal-weight arithmetic mean of six holdout-specific five-seed means",
            "inferential_status": "descriptive benchmark-scenario average; not a population estimator",
        }
        for metric in METRICS:
            col = f"{metric}_mean"
            vals = g[col].to_numpy(dtype=float)
            rec[f"equal_holdout_{metric}_mean"] = float(np.mean(vals))
            rec[f"holdout_to_holdout_{metric}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)

    # Add equal-holdout mean paired deltas for non-control methods without pretending
    # these six fixed benchmark scenarios are a sampled population.
    delta_cols = [c for c in delta_summary.columns if c.startswith("delta_") and c.endswith("_mean")]
    delta_rows: list[dict[str, Any]] = []
    for method, g in delta_summary.groupby("method", sort=True):
        rec: dict[str, Any] = {"method": method}
        for col in delta_cols:
            vals = g[col].to_numpy(dtype=float)
            rec[f"equal_holdout_{col}"] = float(np.mean(vals))
            rec[f"holdout_to_holdout_{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        delta_rows.append(rec)
    if delta_rows:
        out = out.merge(pd.DataFrame(delta_rows), on="method", how="left", validate="one_to_one")
    return out


def visible_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not VISIBLE_ROOT.exists():
        return pd.DataFrame(rows)
    for path in sorted(VISIBLE_ROOT.glob("*/selected_methods.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            rows.append({"path": str(path.relative_to(REPO_ROOT)), "read_error": f"{type(exc).__name__}: {exc}"})
            continue
        cols = list(df.columns)
        dataset = str(df["dataset"].iloc[0]) if "dataset" in df.columns and len(df) else ""
        holdout = normalize_holdout(df["holdout_family"].iloc[0]) if "holdout_family" in df.columns and len(df) else ""
        if not dataset or not holdout:
            name = path.parent.name
            if not dataset:
                dataset = "CICIoT2023" if "CICIoT2023" in name else ("CICIDS2017" if "CICIDS2017" in name else "unknown")
            if not holdout:
                token = name.split("__holdout_", 1)[1].split("__", 1)[0] if "__holdout_" in name else "unknown"
                holdout = normalize_holdout(token)

        budget_candidates = [
            c for c in cols
            if c.lower() in {"budget", "rejection_budget", "max_overall_reject_rate", "overall_reject_budget"}
            or "budget" in c.lower()
        ]
        feasible_candidates = [c for c in cols if c.lower() in {"feasible", "ok", "is_feasible"}]
        method_col = "method" if "method" in cols else ""
        metric_hits = [
            c for c in cols
            if any(term in c.lower() for term in ["unknown_detection", "false_unknown", "macro_f1", "overall_reject"])
        ]

        primary_rows = 0
        budget_col = budget_candidates[0] if budget_candidates else ""
        if budget_col:
            numeric = pd.to_numeric(df[budget_col], errors="coerce")
            primary_rows = int(np.isclose(numeric.to_numpy(dtype=float), PRIMARY_BUDGET, atol=TOL, rtol=0.0, equal_nan=False).sum())

        rows.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "dataset": dataset,
                "holdout_family": holdout,
                "n_rows": int(len(df)),
                "columns": "|".join(cols),
                "method_column": method_col,
                "budget_column": budget_col,
                "feasible_column": feasible_candidates[0] if feasible_candidates else "",
                "metric_columns": "|".join(metric_hits),
                "primary_3pct_rows": primary_rows,
                "common_case_schema_ready": bool(method_col and budget_col and feasible_candidates and metric_hits),
            }
        )
    return pd.DataFrame(rows)


def build_scope(decision: dict[str, Any], surfaces: pd.DataFrame, visible_inv: pd.DataFrame) -> dict[str, Any]:
    direct_claim_surfaces = pd.DataFrame()
    candidate_claim_surfaces = pd.DataFrame()
    if not surfaces.empty and {"surface_guess", "evidence_class"}.issubset(surfaces.columns):
        claim = surfaces[surfaces["surface_guess"].astype(str).isin(["phase3_validation_blind", "validation_visible_rejectors"])].copy()
        direct_claim_surfaces = claim[claim["evidence_class"].astype(str) == "direct_provenance"].copy()
        candidate_claim_surfaces = claim[claim["evidence_class"].astype(str) == "candidate_mapping_unverified"].copy()

    phase5c_required = bool(len(direct_claim_surfaces) > 0)
    visible_ready_files = int(visible_inv["common_case_schema_ready"].sum()) if not visible_inv.empty and "common_case_schema_ready" in visible_inv.columns else 0
    visible_primary_rows = int(visible_inv["primary_3pct_rows"].sum()) if not visible_inv.empty and "primary_3pct_rows" in visible_inv.columns else 0

    return {
        "phase5a_ready_for_phase5b": bool(decision.get("READY_FOR_PHASE5B_SPEC")),
        "phase3_row_id_synthetic_verified": bool(decision.get("phase3_row_id_synthetic_verified_from_source")),
        "phase3_cluster_aware_row_resampling_ready_now": bool(decision.get("phase3_cluster_aware_row_resampling_ready_now")),
        "direct_provenance_claim_surfaces": int(len(direct_claim_surfaces)),
        "candidate_mapping_unverified_claim_surfaces": int(len(candidate_claim_surfaces)),
        "phase5c_cluster_cardinality_gate_required": phase5c_required,
        "row_level_bootstrap_pvalues_authorized_as_population_inference": False,
        "row_level_bootstrap_intervals_if_retained": "conditional on observed test rows and fixed fitted model",
        "phase3_seed_summary_interpretation": "training-randomness robustness only; not population sampling uncertainty",
        "formal_tests_run_in_phase5b": False,
        "new_multiple_testing_family_created": False,
        "validation_visible_selected_method_files": int(len(visible_inv)),
        "validation_visible_files_with_common_case_schema": visible_ready_files,
        "validation_visible_primary_3pct_rows_detected": visible_primary_rows,
        "validation_visible_common_case_next_step": (
            "build common/joint-feasible 3% paired deltas only from files whose budget, feasibility, method and metric columns are explicit"
        ),
    }


def main() -> int:
    decision, surfaces = load_phase5a()
    primary = load_primary_blind()
    five = five_seed_summary(primary)
    deltas = paired_seed_deltas(primary)
    delta_summary = paired_delta_summary(deltas)
    equal = equal_holdout_summary(five, delta_summary)
    visible_inv = visible_inventory()
    scope = build_scope(decision, surfaces, visible_inv)

    OUT.mkdir(parents=True, exist_ok=True)
    five.to_csv(OUT / "blind_primary_3pct_five_seed_summary.csv", index=False)
    deltas.to_csv(OUT / "blind_primary_3pct_paired_seed_deltas.csv", index=False)
    delta_summary.to_csv(OUT / "blind_primary_3pct_paired_delta_summary.csv", index=False)
    equal.to_csv(OUT / "blind_primary_3pct_equal_holdout_summary.csv", index=False)
    visible_inv.to_csv(OUT / "visible_selected_methods_inventory.csv", index=False)
    (OUT / "inferential_scope.json").write_text(json.dumps(scope, indent=2, sort_keys=True), encoding="utf-8")

    inputs: dict[str, Any] = {
        "phase5a_decision": {"path": str(JOINABILITY_DECISION), "sha256": sha256_file(JOINABILITY_DECISION)},
        "phase3b_case_results": {"path": str(PHASE3_CASE_RESULTS), "sha256": sha256_file(PHASE3_CASE_RESULTS)},
    }
    if JOINABILITY_SURFACES.exists():
        inputs["phase5a_surface_joinability"] = {
            "path": str(JOINABILITY_SURFACES),
            "sha256": sha256_file(JOINABILITY_SURFACES),
        }
    manifest = {
        "status": "built_from_frozen_results_no_model_rerun_no_formal_tests",
        "primary_budget": PRIMARY_BUDGET,
        "dataset": "CICIoT2023",
        "seeds": SEEDS,
        "holdouts": HOLDOUTS,
        "methods": METHODS,
        "control_method": CONTROL,
        "inputs": inputs,
        "new_model_training": False,
        "new_threshold_selection": False,
        "test_labels_used_for_new_selection": False,
        "formal_null_hypothesis_tests_run": False,
        "row_bootstrap_pvalues_promoted": False,
        "phase5c_required": bool(scope["phase5c_cluster_cardinality_gate_required"]),
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    view_cols = [
        "holdout_family",
        "method",
        "n_seeds",
        "test_unknown_detection_rate_mean",
        "test_unknown_detection_rate_sd",
        "test_false_unknown_rate_all_known_mean",
        "test_macro_f1_supported_mean",
    ]
    print("Phase-3 blind primary 3% five-seed summary:")
    print(five[view_cols].to_string(index=False))
    print("")
    print("Equal-holdout descriptive means:")
    equal_cols = [
        "method",
        "n_holdouts_equal_weight",
        "equal_holdout_test_unknown_detection_rate_mean",
        "equal_holdout_test_false_unknown_rate_all_known_mean",
        "equal_holdout_test_macro_f1_supported_mean",
    ]
    print(equal[equal_cols].to_string(index=False))
    print("")
    print(f"paired seed rows vs max_confidence: {len(deltas)}/{len(HOLDOUTS) * len(SEEDS) * (len(METHODS)-1)}")
    print(f"paired holdout-method delta summaries: {len(delta_summary)}/18")
    print(f"validation-visible selected_methods files inventoried: {len(visible_inv)}")
    if not visible_inv.empty:
        print(f"validation-visible files with explicit common-case schema: {int(visible_inv['common_case_schema_ready'].sum())}")
        print(f"validation-visible 3% rows detected: {int(visible_inv['primary_3pct_rows'].sum())}")
    print(f"Phase-3 cluster-aware row resampling ready now: {scope['phase3_cluster_aware_row_resampling_ready_now']}")
    print(f"direct-provenance claim surfaces: {scope['direct_provenance_claim_surfaces']}")
    print(f"candidate-mapping-unverified claim surfaces: {scope['candidate_mapping_unverified_claim_surfaces']}")
    print(f"PHASE5C_REQUIRED={scope['phase5c_cluster_cardinality_gate_required']}")
    print(f"output: {OUT}")
    print("PASS=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
