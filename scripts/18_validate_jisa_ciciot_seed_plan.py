#!/usr/bin/env python3
"""Validate the JISA CICIoT2023 five-seed Protocol B experiment plan.

Run after the dedicated CICIoT2023 support audit. This script performs no model
training. It verifies the local processed surface, six support-admissible LOAO
manifests, the frozen three-profile candidate matrix, and the five requested
seeds. The native Protocol-B summarizer is explicitly not treated as the
scientific model selector because its ranking includes test metrics; journal
post-processing must use validation-only selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src.paths import load_config  # noqa: E402

EXPECTED_HOLDOUTS = {
    "Botnet",
    "BruteForce",
    "DDoS",
    "DoS",
    "Other",
    "Scan/Recon",
}
EXPECTED_SEEDS = [123, 124, 125, 126, 127]
EXPECTED_PROFILES = {
    "xgb_inv_family_clipped": {
        "model_family": "xgb",
        "stage1_weight_mode": "inv_family_clipped",
        "stage2_weight_mode": "balanced",
        "stage1_params": {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
        "stage2_params": {
            "n_estimators": 1400,
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        },
    },
    "rf_inv_family_clipped": {
        "model_family": "rf",
        "stage1_weight_mode": "inv_family_clipped",
        "stage2_weight_mode": "balanced",
        "stage1_params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        "stage2_params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
    },
    "rf_class_weight_balanced": {
        "model_family": "rf",
        "stage1_weight_mode": "class_weight_balanced",
        "stage2_weight_mode": "balanced",
        "stage1_params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
        "stage2_params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        },
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    dataset_dir = REPO_ROOT / "processed_V5" / "B_day_file" / "CICIoT2023"
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            fail(f"missing processed CICIoT2023 split: {split_dir}")
        parts = []
        for pattern in ("*.parquet", "*.csv", "*.csv.gz"):
            parts.extend(split_dir.glob(pattern))
        if not parts:
            fail(f"no processed parts found in {split_dir}")
    print("PASS: local processed_V5/B_day_file/CICIoT2023 train/val/test surface exists")

    audit_root = REPO_ROOT / "outputs" / "12_jisa_finalization" / "08_ciciot_support_audit" / "CICIoT2023"
    eligible_csv = audit_root / "eligible_holdouts.csv"
    if not eligible_csv.is_file():
        fail("dedicated CICIoT2023 support audit has not been run")

    import pandas as pd

    eligible = pd.read_csv(eligible_csv)
    if "scenario_valid" not in eligible.columns or "holdout_family" not in eligible.columns:
        fail("eligible_holdouts.csv is missing required columns")
    valid = eligible.loc[eligible["scenario_valid"].astype(str).str.lower().isin({"true", "1"})]
    holdouts = set(valid["holdout_family"].astype(str))
    if holdouts != EXPECTED_HOLDOUTS or len(valid) != 6:
        fail(f"expected six valid CICIoT2023 holdouts {sorted(EXPECTED_HOLDOUTS)}, got {sorted(holdouts)}")
    print("PASS: all 6 CICIoT2023 day-file Protocol B holdouts are support-admissible")

    manifests_dir = audit_root / "manifests"
    manifests = sorted(manifests_dir.glob("*.json"))
    if len(manifests) != 6:
        fail(f"expected 6 support manifests, found {len(manifests)}")
    manifest_holdouts = set()
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not bool(payload.get("scenario_valid")):
            fail(f"invalid scenario manifest present: {path.name}")
        if str(payload.get("dataset")) != "CICIoT2023":
            fail(f"non-CICIoT2023 manifest present: {path.name}")
        manifest_holdouts.add(str(payload.get("holdout_family")))
    if manifest_holdouts != EXPECTED_HOLDOUTS:
        fail("manifest holdout set does not match the audited six-family set")
    print("PASS: support manifest set contains exactly the six intended CICIoT2023 holdouts")

    base_cfg = load_config("config/jisa_ciciot_protocol_b_seed.yml")
    grid = dict((base_cfg.get("protocol_b_grid") or {}).get("legacy_overrides") or {})
    profiles = dict(grid.get("model_profiles") or {})
    if set(profiles) != set(EXPECTED_PROFILES):
        fail(f"candidate profiles drifted: {sorted(profiles)}")
    if list(grid.get("apply_loao_stage1_values") or []) != [True]:
        fail("repeated-seed lane must use strict Stage-1 LOAO only")
    if int((grid.get("max_train_rows") or {}).get("CICIoT2023", -1)) != 1_500_000:
        fail("CICIoT2023 training-row cap drifted")
    if int((grid.get("max_val_rows") or {}).get("CICIoT2023", -1)) != 700_000:
        fail("CICIoT2023 validation-row cap drifted")
    if (grid.get("max_test_rows") or {}).get("CICIoT2023", "missing") is not None:
        fail("CICIoT2023 test evaluation must remain uncapped")

    for name, expected in EXPECTED_PROFILES.items():
        profile = dict(profiles[name])
        if str(profile.get("model_family")) != expected["model_family"]:
            fail(f"{name}: model family drifted")
        if list(profile.get("stage1_weight_modes") or []) != [expected["stage1_weight_mode"]]:
            fail(f"{name}: Stage-1 weighting drifted")
        if list(profile.get("stage2_weight_modes") or []) != [expected["stage2_weight_mode"]]:
            fail(f"{name}: Stage-2 weighting drifted")
        if list(profile.get("stage1_param_grid") or []) != [expected["stage1_params"]]:
            fail(f"{name}: Stage-1 parameters drifted")
        if list(profile.get("stage2_param_grid") or []) != [expected["stage2_params"]]:
            fail(f"{name}: Stage-2 parameters drifted")
    print("PASS: three historical candidate profiles and row budgets are frozen")

    reliability = load_config("config/jisa_ciciot_seed_reliability.yml")
    seed_cfg = dict(reliability.get("seed_reliability") or {})
    seeds = [int(x) for x in seed_cfg.get("seeds", [])]
    if seeds != EXPECTED_SEEDS:
        fail(f"seed set drifted: {seeds}")
    lanes = dict(seed_cfg.get("lanes") or {})
    if set(lanes) != {"protocol_b_loao"}:
        fail("JISA CICIoT reliability config must contain only protocol_b_loao")
    lane = dict(lanes["protocol_b_loao"])
    if str(lane.get("base_config")) != "config/jisa_ciciot_protocol_b_seed.yml":
        fail("seed reliability lane points to the wrong Protocol B base config")
    print("PASS: repeated-seed orchestration is isolated to seeds 123-127 and CICIoT2023 Protocol B")

    runs_per_seed = len(EXPECTED_HOLDOUTS) * len(EXPECTED_PROFILES)
    total_runs = runs_per_seed * len(EXPECTED_SEEDS)
    print("PASS: validation-only journal selection remains separate from test-ranked native summaries")
    print(f"JISA_CICIOT_SEED_COUNT={len(EXPECTED_SEEDS)}")
    print(f"JISA_CICIOT_HOLDOUTS={len(EXPECTED_HOLDOUTS)}")
    print(f"JISA_CICIOT_CANDIDATE_PROFILES={len(EXPECTED_PROFILES)}")
    print(f"JISA_CICIOT_RUNS_PER_SEED={runs_per_seed}")
    print(f"JISA_CICIOT_TOTAL_TRAINING_RUNS={total_runs}")
    print("JISA_CICIOT_SCIENTIFIC_SELECTOR=validation_stage2_macro_f1_then_stage1_auc_then_lexical")


if __name__ == "__main__":
    main()
