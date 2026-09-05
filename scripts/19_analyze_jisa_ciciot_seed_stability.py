#!/usr/bin/env python3
"""Analyze CICIoT2023 Protocol B repeated-seed stability for JISA finalization.

Scientific selection rule (frozen before examining these repeated-seed results):
for each seed x held-out family, select the candidate with highest validation Stage-2
macro-F1, then highest validation Stage-1 AUROC, then lexical run-name order.

The native Protocol B summary uses a test-dependent ranking score and is therefore not
used for the journal's scientific candidate selection in this analysis.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EXPECTED_SEEDS = [123, 124, 125, 126, 127]
EXPECTED_HOLDOUTS = ["Botnet", "BruteForce", "DDoS", "DoS", "Other", "Scan/Recon"]
EXPECTED_PROFILES = [
    "xgb_inv_family_clipped",
    "rf_inv_family_clipped",
    "rf_class_weight_balanced",
]
SELECTION_TOL = 1e-12

METRICS = [
    "unknown_detection_rate",
    "macro_f1",
    "accuracy",
    "false_unknown_rate_all_known",
    "false_unknown_rate_known_attacks",
    "overall_reject_rate",
    "benign_family_fp_rate",
    "stage1_auc_val",
    "stage2_macro_f1_val",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seed-root",
        default="outputs/12_jisa_finalization/09_ciciot_seed_reliability/protocol_b_loao_ciciot2023",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/12_jisa_finalization/10_ciciot_seed_analysis",
    )
    return p.parse_args()


def numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def population_sd(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0 if len(arr) == 1 else float("nan")
    return float(np.std(arr, ddof=1))


def load_seed(seed_root: Path, seed: int) -> pd.DataFrame:
    agg = seed_root / f"seed_{seed}" / "runs" / "aggregate_results.csv"
    if not agg.exists():
        raise FileNotFoundError(f"Missing aggregate results for seed {seed}: {agg}")
    df = pd.read_csv(agg)
    if df.empty:
        raise RuntimeError(f"Aggregate results are empty for seed {seed}: {agg}")
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == "CICIoT2023"].copy()
    df["seed"] = int(seed)
    numeric(
        df,
        METRICS
        + [
            "stage1_thr_high",
            "tau",
        ],
    )
    return df


def validate_seed_matrix(df: pd.DataFrame, seed: int) -> None:
    required = {
        "holdout_family",
        "model_profile",
        "stage2_macro_f1_val",
        "stage1_auc_val",
        "run_name",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Seed {seed} missing required columns: {sorted(missing)}")

    if len(df) != 18:
        raise RuntimeError(f"Seed {seed} has {len(df)} rows; expected exactly 18")

    holdouts = sorted(df["holdout_family"].astype(str).unique().tolist())
    if holdouts != sorted(EXPECTED_HOLDOUTS):
        raise RuntimeError(f"Seed {seed} holdouts mismatch: {holdouts}")

    for holdout, grp in df.groupby("holdout_family", sort=True):
        if len(grp) != 3:
            raise RuntimeError(f"Seed {seed} holdout {holdout} has {len(grp)} candidates; expected 3")
        profiles = sorted(grp["model_profile"].astype(str).tolist())
        if profiles != sorted(EXPECTED_PROFILES):
            raise RuntimeError(
                f"Seed {seed} holdout {holdout} profile mismatch: {profiles}; "
                f"expected {sorted(EXPECTED_PROFILES)}"
            )


def select_validation_winner(grp: pd.DataFrame) -> tuple[pd.Series, dict[str, object]]:
    work = grp.copy()
    work["_s2"] = pd.to_numeric(work["stage2_macro_f1_val"], errors="coerce").fillna(-np.inf)
    work["_s1"] = pd.to_numeric(work["stage1_auc_val"], errors="coerce").fillna(-np.inf)
    work = work.sort_values(
        ["_s2", "_s1", "run_name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    winner = work.iloc[0].copy()
    runner_up = work.iloc[1].copy()
    top_s2 = float(winner["_s2"])
    second_s2 = float(runner_up["_s2"])
    s2_gap = top_s2 - second_s2
    top_s1 = float(winner["_s1"])
    second_s1 = float(runner_up["_s1"])

    s2_tied = work[np.abs(work["_s2"] - top_s2) <= SELECTION_TOL].copy()
    s1_tied_within_s2 = s2_tied[np.abs(s2_tied["_s1"] - top_s1) <= SELECTION_TOL].copy()

    diag = {
        "selected_model_profile": str(winner["model_profile"]),
        "selected_run_name": str(winner["run_name"]),
        "runner_up_model_profile": str(runner_up["model_profile"]),
        "top_stage2_macro_f1_val": top_s2,
        "second_stage2_macro_f1_val": second_s2,
        "stage2_macro_f1_val_gap": float(s2_gap),
        "stage2_top_tie_count": int(len(s2_tied)),
        "stage2_top_tied_profiles": "|".join(sorted(s2_tied["model_profile"].astype(str).tolist())),
        "top_stage1_auc_val": top_s1,
        "second_stage1_auc_val": second_s1,
        "full_selection_tie_count": int(len(s1_tied_within_s2)),
        "full_selection_tied_profiles": "|".join(
            sorted(s1_tied_within_s2["model_profile"].astype(str).tolist())
        ),
        "lexical_tiebreak_needed": bool(len(s1_tied_within_s2) > 1),
    }
    return winner, diag


def metric_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for holdout, grp in selected.groupby("holdout_family", sort=True):
        for metric in METRICS:
            if metric not in grp.columns:
                continue
            vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "holdout_family": holdout,
                    "metric": metric,
                    "n": int(len(vals)),
                    "mean": float(vals.mean()),
                    "sd": population_sd(vals),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
            )
    return pd.DataFrame(rows)


def selection_frequency(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for holdout, grp in selected.groupby("holdout_family", sort=True):
        counts = Counter(grp["model_profile"].astype(str))
        for profile in EXPECTED_PROFILES:
            count = int(counts.get(profile, 0))
            rows.append(
                {
                    "holdout_family": holdout,
                    "model_profile": profile,
                    "selected_count": count,
                    "selected_fraction": count / len(EXPECTED_SEEDS),
                }
            )
    return pd.DataFrame(rows)


def operating_point_frequency(selected: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["holdout_family", "model_profile", "stage1_thr_high", "tau"] if c in selected.columns]
    if len(cols) < 3:
        return pd.DataFrame()
    group_cols = cols
    out = (
        selected.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="selected_count")
        .sort_values(["holdout_family", "selected_count"], ascending=[True, False])
        .reset_index(drop=True)
    )
    out["selected_fraction_within_holdout"] = out["selected_count"] / len(EXPECTED_SEEDS)
    return out


def main() -> None:
    args = parse_args()
    seed_root = Path(args.seed_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.Series] = []
    diagnostics: list[dict[str, object]] = []

    for seed in EXPECTED_SEEDS:
        df = load_seed(seed_root, seed)
        validate_seed_matrix(df, seed)
        all_rows.append(df)
        for holdout in EXPECTED_HOLDOUTS:
            grp = df[df["holdout_family"].astype(str) == holdout].copy()
            winner, diag = select_validation_winner(grp)
            selected_rows.append(winner)
            diagnostics.append({"seed": seed, "holdout_family": holdout, **diag})

    all_df = pd.concat(all_rows, ignore_index=True)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected = selected.sort_values(["holdout_family", "seed"]).reset_index(drop=True)
    diag_df = pd.DataFrame(diagnostics).sort_values(["holdout_family", "seed"]).reset_index(drop=True)

    if len(all_df) != 90:
        raise RuntimeError(f"Expected 90 total candidate runs, found {len(all_df)}")
    if len(selected) != 30:
        raise RuntimeError(f"Expected 30 validation-selected seed/holdout winners, found {len(selected)}")

    selected.to_csv(out_dir / "validation_selected_seed_results.csv", index=False)
    diag_df.to_csv(out_dir / "selection_margin_by_seed_holdout.csv", index=False)

    ms = metric_summary(selected)
    ms.to_csv(out_dir / "metric_summary_by_holdout.csv", index=False)

    sf = selection_frequency(selected)
    sf.to_csv(out_dir / "candidate_selection_frequency.csv", index=False)

    op = operating_point_frequency(selected)
    op.to_csv(out_dir / "operating_point_frequency.csv", index=False)

    overall_selection_counts = Counter(selected["model_profile"].astype(str))
    per_holdout_unique = (
        selected.groupby("holdout_family")["model_profile"].nunique().sort_index().to_dict()
    )
    stable_holdouts = [k for k, v in per_holdout_unique.items() if int(v) == 1]
    unstable_holdouts = [k for k, v in per_holdout_unique.items() if int(v) > 1]

    zero_s2_ties = int((diag_df["stage2_macro_f1_val_gap"].abs() <= SELECTION_TOL).sum())
    lexical_ties = int(diag_df["lexical_tiebreak_needed"].astype(bool).sum())

    summary = {
        "dataset": "CICIoT2023",
        "seeds": EXPECTED_SEEDS,
        "holdouts": EXPECTED_HOLDOUTS,
        "candidate_profiles": EXPECTED_PROFILES,
        "candidate_runs": int(len(all_df)),
        "validation_selected_cases": int(len(selected)),
        "selection_rule": "max validation stage2_macro_f1_val; then max validation stage1_auc_val; then lexical run_name",
        "test_metrics_used_for_selection": False,
        "overall_candidate_selection_counts": dict(sorted(overall_selection_counts.items())),
        "holdouts_with_same_selected_profile_all_5_seeds": stable_holdouts,
        "holdouts_with_profile_switching_across_seeds": unstable_holdouts,
        "stage2_exact_top_tie_cases": zero_s2_ties,
        "full_validation_ties_requiring_lexical_tiebreak": lexical_ties,
        "selection_tolerance": SELECTION_TOL,
    }

    # Add compact UDR and macro-F1 mean/SD per holdout for quick console inspection.
    for metric in ["unknown_detection_rate", "macro_f1", "false_unknown_rate_all_known", "overall_reject_rate"]:
        metric_rows = ms[ms["metric"] == metric]
        summary[f"{metric}_by_holdout"] = {
            str(r["holdout_family"]): {"mean": float(r["mean"]), "sd": float(r["sd"])}
            for _, r in metric_rows.iterrows()
        }

    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nCandidate selection frequency:")
    print(sf.to_string(index=False))
    print("\nUDR stability by holdout:")
    udr = ms[ms["metric"] == "unknown_detection_rate"][
        ["holdout_family", "n", "mean", "sd", "min", "max"]
    ]
    print(udr.to_string(index=False))
    print("\nMacro-F1 stability by holdout:")
    mf1 = ms[ms["metric"] == "macro_f1"][
        ["holdout_family", "n", "mean", "sd", "min", "max"]
    ]
    print(mf1.to_string(index=False))
    print(f"\nWrote analysis to: {out_dir}")


if __name__ == "__main__":
    main()
