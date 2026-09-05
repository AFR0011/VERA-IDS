#!/usr/bin/env python3
"""Quantify where held-out families go when they are not rejected as Unknown.

Primary analysis uses one fixed RF profile across both datasets:
  * CICIDS2017: group-safe Protocol B, seed 123, rf_class_weight_balanced.
  * CICIoT2023: day/file Protocol B, seeds 123-127, rf_class_weight_balanced.

For each held-out family and run, the script reads confusion_matrix_system_test.csv,
extracts the true-Unknown row, and separates:
  * Unknown detections (UDR), from
  * residual non-Unknown destinations.

Residual failure structure is summarized by:
  * dominant destination,
  * dominant-destination share conditional on failure,
  * normalized Shannon entropy across all available non-Unknown destinations,
  * concentration = 1 - normalized entropy.

These are descriptive diagnostics. A dominant destination is not interpreted as semantic
or causal equivalence between attack families.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CICIDS_AGG = ROOT / "outputs/12_jisa_finalization/06_group_safe_protocol_b_rf/aggregate_results.csv"
CICIOT_SEED_ROOT = ROOT / "outputs/12_jisa_finalization/09_ciciot_seed_reliability/protocol_b_loao_ciciot2023"
CICIOT_SELECTED = ROOT / "outputs/12_jisa_finalization/10_ciciot_seed_analysis/validation_selected_seed_results.csv"
OUT = ROOT / "outputs/12_jisa_finalization/12_failure_destinations"
SEEDS = [123, 124, 125, 126, 127]
PROFILE = "rf_class_weight_balanced"
UNKNOWN = "Unknown"


def resolve_run_dir(value: object) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def analyze_confusion(run_dir: Path, *, dataset: str, family: str, seed: int | None, profile: str, lane: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    cm_path = run_dir / "confusion_matrix_system_test.csv"
    if not cm_path.exists():
        raise FileNotFoundError(cm_path)
    cm = pd.read_csv(cm_path, index_col=0)
    cm.index = cm.index.astype(str)
    cm.columns = cm.columns.astype(str)
    if UNKNOWN not in cm.index or UNKNOWN not in cm.columns:
        raise RuntimeError(f"Unknown row/column missing from {cm_path}")

    row = pd.to_numeric(cm.loc[UNKNOWN], errors="raise").astype(float)
    n_total = float(row.sum())
    if n_total <= 0:
        raise RuntimeError(f"No true-Unknown rows in {cm_path}")
    unknown_count = float(row.get(UNKNOWN, 0.0))
    nonunknown = row.drop(labels=[UNKNOWN])
    failure_total = float(nonunknown.sum())
    udr = unknown_count / n_total

    possible = list(nonunknown.index)
    k = len(possible)
    if failure_total > 0:
        probs = nonunknown.to_numpy(dtype=float) / failure_total
        nz = probs[probs > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        norm_entropy = entropy / math.log(k) if k > 1 else 0.0
        concentration = 1.0 - norm_entropy
        dominant = str(nonunknown.idxmax())
        dominant_count = float(nonunknown.max())
        dominant_share_failure = dominant_count / failure_total
    else:
        entropy = 0.0
        norm_entropy = 0.0
        concentration = 1.0
        dominant = "None"
        dominant_count = 0.0
        dominant_share_failure = np.nan

    summary = {
        "dataset": dataset,
        "holdout_family": family,
        "seed": seed,
        "model_profile": profile,
        "analysis_lane": lane,
        "n_true_unknown": int(n_total),
        "unknown_count": int(unknown_count),
        "unknown_detection_rate": udr,
        "nonunknown_failure_count": int(failure_total),
        "dominant_destination": dominant,
        "dominant_destination_count": int(dominant_count),
        "dominant_destination_share_of_failures": dominant_share_failure,
        "dominant_destination_share_of_all_unknown": dominant_count / n_total,
        "destination_entropy": entropy,
        "normalized_destination_entropy": norm_entropy,
        "destination_concentration": concentration,
        "possible_nonunknown_destinations": k,
        "run_dir": str(run_dir),
    }
    destinations = []
    for dest, count in nonunknown.items():
        destinations.append({
            "dataset": dataset,
            "holdout_family": family,
            "seed": seed,
            "model_profile": profile,
            "analysis_lane": lane,
            "destination": str(dest),
            "count": int(count),
            "share_of_all_unknown": float(count / n_total),
            "share_conditional_on_nonunknown": float(count / failure_total) if failure_total > 0 else np.nan,
        })
    return summary, destinations


def load_cicids_fixed_rf() -> pd.DataFrame:
    df = pd.read_csv(CICIDS_AGG)
    sub = df[(df["dataset"].astype(str) == "CICIDS2017") & (df["model_profile"].astype(str) == PROFILE)].copy()
    if len(sub) != 6 or sub["holdout_family"].nunique() != 6:
        raise RuntimeError(f"Expected six CICIDS fixed-RF rows, found {len(sub)}")
    sub["seed"] = 123
    return sub


def load_ciciot_fixed_rf() -> pd.DataFrame:
    parts = []
    for seed in SEEDS:
        path = CICIOT_SEED_ROOT / f"seed_{seed}" / "runs" / "aggregate_results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        sub = df[(df["dataset"].astype(str) == "CICIoT2023") & (df["model_profile"].astype(str) == PROFILE)].copy()
        if len(sub) != 6 or sub["holdout_family"].nunique() != 6:
            raise RuntimeError(f"Seed {seed}: expected six CICIoT fixed-RF rows, found {len(sub)}")
        sub["seed"] = seed
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def summarize_seeded(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "unknown_detection_rate",
        "dominant_destination_share_of_failures",
        "normalized_destination_entropy",
        "destination_concentration",
    ]
    for (dataset, family, lane), grp in per_run.groupby(["dataset", "holdout_family", "analysis_lane"], sort=True):
        sinks = Counter(grp["dominant_destination"].astype(str))
        modal_sink, modal_count = sinks.most_common(1)[0]
        row = {
            "dataset": dataset,
            "holdout_family": family,
            "analysis_lane": lane,
            "n_runs": int(len(grp)),
            "modal_dominant_destination": modal_sink,
            "dominant_destination_stability_fraction": modal_count / len(grp),
        }
        for metric in metrics:
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{metric}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{metric}_max"] = float(vals.max()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_run_rows: list[dict[str, object]] = []
    dest_rows: list[dict[str, object]] = []

    fixed = pd.concat([load_cicids_fixed_rf(), load_ciciot_fixed_rf()], ignore_index=True)
    for _, rec in fixed.iterrows():
        summary, destinations = analyze_confusion(
            resolve_run_dir(rec["run_dir"]),
            dataset=str(rec["dataset"]),
            family=str(rec["holdout_family"]),
            seed=int(rec["seed"]),
            profile=str(rec["model_profile"]),
            lane="fixed_rf_primary",
        )
        per_run_rows.append(summary)
        dest_rows.extend(destinations)

    # Secondary CICIoT analysis using the validation-selected profile in each seed/holdout.
    if CICIOT_SELECTED.exists():
        selected = pd.read_csv(CICIOT_SELECTED)
        if len(selected) != 30:
            raise RuntimeError(f"Expected 30 validation-selected CICIoT rows, found {len(selected)}")
        for _, rec in selected.iterrows():
            summary, destinations = analyze_confusion(
                resolve_run_dir(rec["run_dir"]),
                dataset="CICIoT2023",
                family=str(rec["holdout_family"]),
                seed=int(rec["seed"]),
                profile=str(rec["model_profile"]),
                lane="validation_selected_secondary",
            )
            per_run_rows.append(summary)
            dest_rows.extend(destinations)

    per_run = pd.DataFrame(per_run_rows).sort_values(["analysis_lane", "dataset", "holdout_family", "seed"]).reset_index(drop=True)
    destinations = pd.DataFrame(dest_rows).sort_values(["analysis_lane", "dataset", "holdout_family", "seed", "destination"]).reset_index(drop=True)
    summary = summarize_seeded(per_run)

    per_run.to_csv(OUT / "failure_structure_per_run.csv", index=False)
    destinations.to_csv(OUT / "destination_distribution_per_run.csv", index=False)
    summary.to_csv(OUT / "failure_structure_summary.csv", index=False)

    primary = summary[summary["analysis_lane"] == "fixed_rf_primary"].copy()
    primary.to_csv(OUT / "failure_structure_primary_fixed_rf.csv", index=False)

    payload = {
        "analysis": "held-out-family residual destination concentration",
        "primary_lane": "fixed_rf_primary",
        "primary_profile": PROFILE,
        "cicids_surface": "B_group_safe, seed 123",
        "ciciot_surface": "B_day_file, seeds 123-127",
        "secondary_lane": "validation_selected_secondary for CICIoT2023 only",
        "entropy_definition": "Shannon entropy of non-Unknown destinations, normalized by log(number of available non-Unknown destination labels)",
        "concentration_definition": "1 - normalized destination entropy",
        "interpretation_boundary": "Destination concentration is descriptive and does not imply semantic or causal equivalence between held-out and sink families.",
        "primary_rows": int(len(primary)),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print("\nPrimary fixed-RF failure structure:")
    cols = [
        "dataset", "holdout_family", "n_runs", "modal_dominant_destination",
        "dominant_destination_stability_fraction",
        "unknown_detection_rate_mean",
        "dominant_destination_share_of_failures_mean",
        "destination_concentration_mean",
    ]
    print(primary[cols].to_string(index=False))
    print(f"\nWrote analysis to: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
