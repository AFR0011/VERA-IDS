#!/usr/bin/env python3
"""Verify the frozen v2026.10-comnet comparison snapshot before tagging a release.

This command performs no training. It checks only committed aggregate evidence and
release metadata.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release" / "v2026.10-comnet"
RAW = REL / "raw"

REQUIRED = [
    REL / "README.md",
    REL / "PROVENANCE.md",
    REL / "RELEASE_NOTES.md",
    RAW / "model_competence_summary.csv",
    RAW / "primary_vera_five_seed_reference.csv",
    RAW / "external_protocol_a_summary.csv",
    RAW / "external_protocol_b_holdout_summary.csv",
    RAW / "external_protocol_b_family_heterogeneity_summary.csv",
    RAW / "final_campaign_gate.txt",
]


def fail(message: str) -> None:
    raise SystemExit(f"[FAIL] {message}")


def assert_all_five(frame: pd.DataFrame, column: str, label: str) -> None:
    if column not in frame.columns:
        fail(f"{label}: missing seed-count column {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        fail(f"{label}: no numeric seed counts")
    bad = values[values != 5]
    if not bad.empty:
        fail(f"{label}: expected five-seed coverage, found {sorted(set(bad.astype(int)))}")


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
    if missing:
        fail("missing release files: " + ", ".join(missing))

    competence = pd.read_csv(RAW / "model_competence_summary.csv")
    primary = pd.read_csv(RAW / "primary_vera_five_seed_reference.csv")
    ext_a = pd.read_csv(RAW / "external_protocol_a_summary.csv")
    ext_b = pd.read_csv(RAW / "external_protocol_b_holdout_summary.csv")
    hetero = pd.read_csv(RAW / "external_protocol_b_family_heterogeneity_summary.csv")

    expected_datasets = {"CICIDS2017", "CICIoT2023"}
    if set(competence["dataset"].astype(str)) != expected_datasets:
        fail(f"competence datasets differ from expected {sorted(expected_datasets)}")

    expected_models = {"extra_trees", "lgbm", "catboost"}
    models = set(competence["model_family"].astype(str))
    if not expected_models.issubset(models):
        fail(f"competence models missing: {sorted(expected_models - models)}")

    assert_all_five(competence, "n", "model competence")
    assert_all_five(primary, "n", "primary RF/XGB reference")
    assert_all_five(ext_a, "n_seeds", "external Protocol A")
    assert_all_five(ext_b, "n_seeds", "external Protocol B holdout")
    assert_all_five(hetero, "n_seeds", "external Protocol B heterogeneity")

    profiles = set(hetero["model_profile"].astype(str))
    expected_profiles = {
        "adewole2025_xgb_profile",
        "hung2026_xgb_profile",
        "keskin2026_lgbm_profile",
    }
    if profiles != expected_profiles:
        fail(
            "Protocol-B heterogeneity profiles differ from expected: "
            f"expected={sorted(expected_profiles)} actual={sorted(profiles)}"
        )

    holdout = ext_b.copy()
    expected_holdouts = {"Botnet", "BruteForce", "DDoS", "DoS", "Other", "Scan/Recon"}
    for profile in expected_profiles:
        observed = set(
            holdout.loc[holdout["model_profile"].astype(str) == profile, "holdout_family"].astype(str)
        )
        if observed != expected_holdouts:
            fail(
                f"{profile}: Protocol-B holdout coverage mismatch; "
                f"expected={sorted(expected_holdouts)} actual={sorted(observed)}"
            )

    gate_text = (RAW / "final_campaign_gate.txt").read_text(encoding="utf-8", errors="replace")
    required_gate_phrases = [
        "competence repeated summary present: True",
        "primary five-seed reference present: True",
        "external Protocol A repeated summary present: True",
        "external Protocol A seed count range: 5..5",
        "external Protocol B summary present: True",
        "external Protocol B seed count range: 5..5",
    ]
    absent = [x for x in required_gate_phrases if x not in gate_text]
    if absent:
        fail("final campaign gate is missing expected completion lines: " + "; ".join(absent))

    print("[PASS] v2026.10-comnet release snapshot is complete.")
    print(f"[PASS] competence rows: {len(competence)}")
    print(f"[PASS] primary-reference rows: {len(primary)}")
    print(f"[PASS] external Protocol-A rows: {len(ext_a)}")
    print(f"[PASS] external Protocol-B holdout rows: {len(ext_b)}")
    print(f"[PASS] external Protocol-B heterogeneity rows: {len(hetero)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
