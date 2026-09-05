#!/usr/bin/env python3
"""Pair Protocol-A known-family RF recognition with Protocol-B held-out-family RF UDR.

Primary scientific use
----------------------
Ask whether a family that is easy to recognize when represented in training is also easy
for the same model family to identify as Unknown when excluded from training.

This analysis deliberately fixes the model family/profile rather than selecting a winner
per holdout:
  * Protocol A: RF confusion-matrix evidence, strict policy primary; strict_tau retained.
  * CICIDS2017 Protocol B: rf_class_weight_balanced on the exact-group-safe surface.
  * CICIoT2023 Protocol B: rf_class_weight_balanced across seeds 123-127 on B_day_file.

No test metric is used for model selection because no model selection occurs here.
Spearman correlations are descriptive only (n=6 families per dataset).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs/evidence/protocol_a_confusion_matrices.jsonl"
CICIDS_B = ROOT / "outputs/12_jisa_finalization/06_group_safe_protocol_b_rf/aggregate_results.csv"
CICIOT_SEEDS = ROOT / "outputs/12_jisa_finalization/09_ciciot_seed_reliability/protocol_b_loao_ciciot2023"
OUT = ROOT / "outputs/12_jisa_finalization/11_known_vs_heldout"
SEEDS = [123, 124, 125, 126, 127]
PROFILE = "rf_class_weight_balanced"
POLICIES = ["strict", "strict_tau"]


def load_protocol_a_rf() -> pd.DataFrame:
    if not EVIDENCE.exists():
        raise FileNotFoundError(EVIDENCE)
    rows = []
    with EVIDENCE.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("surface") != "core" or rec.get("model_family") != "rf":
                continue
            if rec.get("profile") not in (None, "", "null"):
                continue
            if rec.get("policy_variant") not in POLICIES:
                continue
            labels = list(rec["labels"])
            mat = np.asarray(rec["matrix"], dtype=float)
            for i, family in enumerate(labels):
                if family in {"Benign", "Unknown"}:
                    continue
                support = float(mat[i, :].sum())
                tp = float(mat[i, i])
                pred_total = float(mat[:, i].sum())
                recall = tp / support if support else np.nan
                precision = tp / pred_total if pred_total else np.nan
                f1 = (
                    2.0 * precision * recall / (precision + recall)
                    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
                    else np.nan
                )
                rows.append({
                    "dataset": rec["dataset"],
                    "family": family,
                    "protocol_a_policy": rec["policy_variant"],
                    "known_support": int(support),
                    "known_recall": recall,
                    "known_precision": precision,
                    "known_f1": f1,
                    "protocol_a_evidence_id": rec.get("evidence_id"),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No Protocol-A RF confusion-matrix evidence found")
    dup = df.duplicated(["dataset", "family", "protocol_a_policy"], keep=False)
    if dup.any():
        raise RuntimeError("Protocol-A RF evidence is not unique for dataset/family/policy")
    return df


def load_cicids_fixed_rf() -> pd.DataFrame:
    if not CICIDS_B.exists():
        raise FileNotFoundError(CICIDS_B)
    df = pd.read_csv(CICIDS_B)
    df = df[df["model_profile"].astype(str) == PROFILE].copy()
    if len(df) != 6 or df["holdout_family"].nunique() != 6:
        raise RuntimeError(f"Expected 6 CICIDS fixed-RF rows, found {len(df)}")
    out = df[[
        "holdout_family", "unknown_detection_rate", "macro_f1",
        "false_unknown_rate_all_known", "overall_reject_rate",
    ]].copy()
    out = out.rename(columns={"holdout_family": "family"})
    out["dataset"] = "CICIDS2017"
    out["heldout_udr_mean"] = pd.to_numeric(out.pop("unknown_detection_rate"), errors="raise")
    out["heldout_udr_sd"] = np.nan
    out["protocol_b_seeds"] = 1
    out["protocol_b_profile"] = PROFILE
    return out


def load_ciciot_fixed_rf() -> pd.DataFrame:
    parts = []
    for seed in SEEDS:
        path = CICIOT_SEEDS / f"seed_{seed}" / "runs" / "aggregate_results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        sub = df[df["model_profile"].astype(str) == PROFILE].copy()
        if len(sub) != 6 or sub["holdout_family"].nunique() != 6:
            raise RuntimeError(f"Seed {seed}: expected 6 fixed-RF rows, found {len(sub)}")
        sub["seed"] = seed
        parts.append(sub)
    all_rf = pd.concat(parts, ignore_index=True)
    metrics = ["unknown_detection_rate", "macro_f1", "false_unknown_rate_all_known", "overall_reject_rate"]
    for col in metrics:
        all_rf[col] = pd.to_numeric(all_rf[col], errors="raise")
    rows = []
    for family, grp in all_rf.groupby("holdout_family", sort=True):
        rows.append({
            "dataset": "CICIoT2023",
            "family": family,
            "heldout_udr_mean": grp["unknown_detection_rate"].mean(),
            "heldout_udr_sd": grp["unknown_detection_rate"].std(ddof=1),
            "macro_f1": grp["macro_f1"].mean(),
            "false_unknown_rate_all_known": grp["false_unknown_rate_all_known"].mean(),
            "overall_reject_rate": grp["overall_reject_rate"].mean(),
            "protocol_b_seeds": len(grp),
            "protocol_b_profile": PROFILE,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = load_protocol_a_rf()
    b = pd.concat([load_cicids_fixed_rf(), load_ciciot_fixed_rf()], ignore_index=True)

    merged = a.merge(b, on=["dataset", "family"], how="inner", validate="many_to_one")
    expected = 2 * 6 * len(POLICIES)
    if len(merged) != expected:
        raise RuntimeError(f"Expected {expected} paired rows, found {len(merged)}")
    merged = merged.sort_values(["dataset", "protocol_a_policy", "family"]).reset_index(drop=True)
    merged.to_csv(OUT / "known_vs_heldout_pairing.csv", index=False)

    primary = merged[merged["protocol_a_policy"] == "strict"].copy()
    primary.to_csv(OUT / "known_vs_heldout_primary_strict.csv", index=False)

    corr_rows = []
    for dataset, grp in primary.groupby("dataset", sort=True):
        rho, p = spearmanr(grp["known_recall"], grp["heldout_udr_mean"])
        corr_rows.append({
            "dataset": dataset,
            "n_families": len(grp),
            "spearman_rho_known_recall_vs_heldout_udr": rho,
            "spearman_p_descriptive_only": p,
        })
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT / "descriptive_correlations.csv", index=False)

    summary = {
        "analysis": "matched RF known-family recognition versus held-out-family unknown detection",
        "protocol_a_primary_policy": "strict",
        "protocol_a_secondary_policy": "strict_tau",
        "protocol_b_fixed_profile": PROFILE,
        "cicids_protocol_b_surface": "exact-feature-group-safe sensitivity surface, seed 123",
        "ciciot_protocol_b_surface": "B_day_file, fixed RF profile averaged over seeds 123-127",
        "model_selection_performed": False,
        "datasets": {},
        "interpretation_boundary": "Descriptive family-level comparison only. With n=6 families per dataset, correlations are not treated as confirmatory inference.",
    }
    for dataset, grp in primary.groupby("dataset", sort=True):
        rho = float(corr.loc[corr["dataset"] == dataset, "spearman_rho_known_recall_vs_heldout_udr"].iloc[0])
        summary["datasets"][dataset] = {
            "families": int(len(grp)),
            "known_recall_min": float(grp["known_recall"].min()),
            "known_recall_max": float(grp["known_recall"].max()),
            "heldout_udr_min": float(grp["heldout_udr_mean"].min()),
            "heldout_udr_max": float(grp["heldout_udr_mean"].max()),
            "spearman_rho_descriptive": rho,
        }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nPrimary strict-policy pairing:")
    print(primary[["dataset", "family", "known_recall", "heldout_udr_mean", "heldout_udr_sd"]].to_string(index=False))
    print(f"\nWrote analysis to: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
