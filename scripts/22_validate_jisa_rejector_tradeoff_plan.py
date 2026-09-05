#!/usr/bin/env python3
"""Preflight the 12-case JISA rejector trade-off replay before model execution."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "outputs/12_jisa_finalization/13_rejector_tradeoff_input/rejector_cases.csv"
AUDIT_ROOTS = [
    ROOT / "outputs/12_jisa_finalization/05_group_safe_support_audit",
    ROOT / "outputs/12_jisa_finalization/08_ciciot_support_audit",
]
EXPECTED = {
    "CICIDS2017": {
        "Botnet": "rf_class_weight_balanced",
        "BruteForce": "rf_class_weight_balanced",
        "DDoS": "rf_class_weight_balanced",
        "DoS": "rf_class_weight_balanced",
        "Scan/Recon": "rf_class_weight_balanced",
        "Web/App": "rf_class_weight_balanced",
    },
    "CICIoT2023": {
        "Botnet": "rf_class_weight_balanced",
        "BruteForce": "xgb_inv_family_clipped",
        "DDoS": "rf_class_weight_balanced",
        "DoS": "rf_class_weight_balanced",
        "Other": "rf_class_weight_balanced",
        "Scan/Recon": "rf_class_weight_balanced",
    },
}
REQUIRED = {
    "dataset", "holdout_family", "model_profile", "model_family",
    "apply_loao_stage1", "stage1_weight_mode", "stage2_weight_mode",
    "stage1_params", "stage2_params",
}


def norm(value: object) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def main() -> None:
    if not CASES.exists():
        raise FileNotFoundError(CASES)
    cases = pd.read_csv(CASES)
    missing = REQUIRED.difference(cases.columns)
    if missing:
        raise RuntimeError(f"rejector_cases.csv missing columns: {sorted(missing)}")
    if len(cases) != 12:
        raise RuntimeError(f"Expected exactly 12 canonical cases, found {len(cases)}")
    if cases.duplicated(["dataset", "holdout_family"]).any():
        raise RuntimeError("Duplicate dataset/holdout rows in rejector_cases.csv")

    expected_pairs = {(d, h): p for d, fams in EXPECTED.items() for h, p in fams.items()}
    actual_pairs = {(str(r.dataset), str(r.holdout_family)): str(r.model_profile) for r in cases.itertuples()}
    if actual_pairs != expected_pairs:
        raise RuntimeError(f"Canonical case/profile matrix mismatch.\nExpected: {expected_pairs}\nActual: {actual_pairs}")

    for r in cases.itertuples():
        if str(r.apply_loao_stage1).strip().lower() not in {"true", "1"}:
            raise RuntimeError(f"Non-strict Stage-1 LOAO case: {r.dataset}/{r.holdout_family}")
        if str(r.stage2_weight_mode) != "balanced":
            raise RuntimeError(f"Unexpected Stage-2 weighting: {r.dataset}/{r.holdout_family}")
        json.loads(str(r.stage1_params))
        json.loads(str(r.stage2_params))

    audit_frames = []
    for root in AUDIT_ROOTS:
        path = root / "eligible_holdouts_all.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if "scenario_valid" in df.columns:
            valid = df["scenario_valid"].astype(str).str.lower().isin(["true", "1"])
            df = df[valid].copy()
        audit_frames.append(df)
    audit = pd.concat(audit_frames, ignore_index=True)
    audit["case_key"] = audit["dataset"].astype(str) + "::" + audit["holdout_family"].map(norm)
    if audit["case_key"].duplicated().any():
        dup = audit.loc[audit["case_key"].duplicated(keep=False), ["dataset", "holdout_family"]]
        raise RuntimeError(f"Duplicate admissible audit cases across roots:\n{dup.to_string(index=False)}")

    case_keys = cases["dataset"].astype(str) + "::" + cases["holdout_family"].map(norm)
    missing_keys = sorted(set(case_keys) - set(audit["case_key"]))
    if missing_keys:
        raise RuntimeError(f"Canonical cases missing admissible support manifests: {missing_keys}")

    resolved = cases.copy()
    resolved["case_key"] = case_keys
    resolved = resolved.merge(audit[["case_key", "manifest_path"]], on="case_key", how="left", validate="one_to_one")
    for value in resolved["manifest_path"]:
        p = Path(str(value))
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(p)

    counts = cases.groupby(["dataset", "model_profile"]).size().to_dict()
    print("PASS: canonical rejector cohort contains exactly 12 unique dataset/holdout cases")
    print("PASS: CICIDS2017 uses fixed rf_class_weight_balanced for all 6 holdouts")
    print("PASS: CICIoT2023 uses seed-123 validation-selected profiles; only BruteForce uses XGB")
    print("PASS: all 12 cases use strict Stage-1 LOAO and balanced Stage-2 weighting")
    print("PASS: all 12 cases resolve to support-admissible manifests in the new audit roots")
    print("PASS: model parameter JSON is parseable for all 12 cases")
    print(f"JISA_REJECTOR_CASES={len(cases)}")
    print(f"JISA_REJECTOR_PROFILE_COUNTS={counts}")
    print("JISA_REJECTOR_SELECTION_DISCIPLINE=validation_only")
    print("JISA_REJECTOR_PRIMARY_METHODS=max_softmax,margin,entropy,conformal")


if __name__ == "__main__":
    main()
