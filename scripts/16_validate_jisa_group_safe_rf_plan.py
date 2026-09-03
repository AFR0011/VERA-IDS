#!/usr/bin/env python3
"""Preflight the frozen CICIDS2017 group-safe fixed-RF Protocol-B run plan."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework._native import protocol_b_grid  # noqa: E402
from ids_eval_framework.src.paths import deep_update, load_config  # noqa: E402

EXPECTED_FINGERPRINT = "9de273bc7474dd92a69c6c35a03646e2158a15deef79816847e220f555bdd486"
EXPECTED_HOLDOUTS = {"Botnet", "BruteForce", "DDoS", "DoS", "Scan/Recon", "Web/App"}
EXPECTED_PROFILES = {"rf_inv_family_clipped", "rf_class_weight_balanced"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/jisa_group_safe_rf.yml")
    args = ap.parse_args()

    config = load_config(args.config)

    summary_path = REPO_ROOT / "outputs/12_jisa_finalization/04_group_safe_surface/group_safe_surface_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing materialized surface summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("aggregate_surface_fingerprint_sha256") != EXPECTED_FINGERPRINT:
        raise RuntimeError("Group-safe surface fingerprint differs from the frozen Phase-1 materialization.")
    if not bool(summary.get("zero_cross_split_exact_feature_overlap")):
        raise RuntimeError("Materialized surface does not certify zero cross-split exact-feature overlap.")
    if not bool(summary.get("all_families_supported")):
        raise RuntimeError("Materialized surface no longer satisfies family support requirements.")

    eligible_path = REPO_ROOT / "outputs/12_jisa_finalization/05_group_safe_support_audit/CICIDS2017/eligible_holdouts.csv"
    if not eligible_path.is_file():
        raise RuntimeError(f"Missing support-audit result: {eligible_path}")
    eligible = pd.read_csv(eligible_path)
    valid = eligible.loc[eligible["scenario_valid"].astype(str).str.lower().eq("true")].copy()
    got_holdouts = set(valid["holdout_family"].astype(str))
    if got_holdouts != EXPECTED_HOLDOUTS:
        raise RuntimeError(f"Expected six holdouts {sorted(EXPECTED_HOLDOUTS)}, got {sorted(got_holdouts)}")

    grid_cfg = (config.get("protocol_b_grid") or {}).get("legacy_overrides") or {}
    original = deepcopy(protocol_b_grid.CFG)
    try:
        deep_update(protocol_b_grid.CFG, grid_cfg)
        manifests = protocol_b_grid.discover_manifests()
        plan = protocol_b_grid.build_run_plan(manifests)
    finally:
        protocol_b_grid.CFG.clear()
        protocol_b_grid.CFG.update(original)

    if len(manifests) != 6:
        raise RuntimeError(f"Expected 6 manifests, got {len(manifests)}")
    if len(plan) != 12:
        raise RuntimeError(f"Expected exactly 12 fixed-RF runs, got {len(plan)}")

    by_holdout: dict[str, set[str]] = {}
    for row in plan:
        holdout = str(row["holdout_family"])
        profile = str(row["model_profile"])
        by_holdout.setdefault(holdout, set()).add(profile)
        if str(row["model_family"]) != "rf":
            raise RuntimeError(f"Non-RF model leaked into plan: {row}")
        if not bool(row["apply_loao_stage1"]):
            raise RuntimeError(f"Non-strict Stage-1 LOAO run leaked into plan: {row}")
        if str(row.get("stage2_weight_mode")) != "balanced":
            raise RuntimeError(f"Unexpected Stage-2 weighting mode: {row}")
        for key in ("stage1_params", "stage2_params"):
            params = dict(row[key])
            expected = {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            }
            if params != expected:
                raise RuntimeError(f"Unexpected fixed RF parameters in {key}: {params}")

    if set(by_holdout) != EXPECTED_HOLDOUTS:
        raise RuntimeError(f"Plan holdouts differ from expected: {sorted(by_holdout)}")
    for holdout, profiles in sorted(by_holdout.items()):
        if profiles != EXPECTED_PROFILES:
            raise RuntimeError(f"{holdout}: expected both fixed profiles, got {sorted(profiles)}")

    weight_counts = Counter(str(row["stage1_weight_mode"]) for row in plan)
    print("PASS: frozen group-safe surface fingerprint matches")
    print("PASS: zero cross-split exact-feature overlap remains certified")
    print("PASS: all 6 Protocol-B holdouts remain support-admissible")
    print("PASS: run plan contains exactly 12 RF runs (6 holdouts x 2 fixed weighting lanes)")
    print("PASS: all runs use strict Stage-1 LOAO, balanced Stage-2 weighting, and the frozen 300-tree RF parameters")
    print(f"JISA_GROUP_SAFE_RF_PLAN_RUNS={len(plan)}")
    print(f"JISA_GROUP_SAFE_RF_PLAN_HOLDOUTS={len(by_holdout)}")
    print(f"JISA_GROUP_SAFE_RF_PLAN_WEIGHT_COUNTS={dict(weight_counts)}")
    print(f"JISA_GROUP_SAFE_SURFACE_SHA256={EXPECTED_FINGERPRINT}")


if __name__ == "__main__":
    main()
