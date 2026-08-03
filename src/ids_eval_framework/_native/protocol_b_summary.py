#!/usr/bin/env python3
"""
6.SummarizeProtocolBGrid.py
==========================

Purpose
-------
Aggregate Protocol B grid-search results into comparison tables that are easier to read
and present.

This script does not train anything. It reads the outputs from 5.ProtocolB_GridRunner.py,
then produces:
    - one cleaned aggregate table
    - one "best per holdout" table
    - one "best per dataset/model family" table
    - a simple Pareto-style shortlist

Why this matters
----------------
A raw grid search is noisy. You need a summarizer that answers questions like:
    - Which runs are the strongest open-set candidates?
    - Does RF help or not under the same valid scenarios?
    - Which settings trade off unknown detection vs false-unknown cleanly?
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================
CFG: Dict[str, object] = {
    # Root folder used by the grid runner.
    "runs_root": "protocolB_grid_runs step 3 - CICIDS2017 sweep",

    # Aggregate CSV written by the grid runner.
    "aggregate_csv": os.path.join("protocolB_grid_runs step 3 - CICIDS2017 sweep", "aggregate_results.csv"),

    # Output folder for summary tables.
    "out_dir": os.path.join("protocolB_grid_runs step 3 - CICIDS2017 sweep", "summary"),

    # Frozen step-2 baseline comparison pack.
    "baseline_best_per_holdout_csv": os.path.join(
        "protocolB_grid_runs step 2 stage-1 LOAO",
        "summary",
        "best_per_holdout.csv",
    ),
    "best_per_holdout_group_cols": ["dataset", "holdout_family"],

    # A simple weighted ranking score. Adjust the coefficients to match what you want to
    # emphasize in the thesis writeup.
    "ranking": {
        # Positive terms
        "w_unknown_detection_rate": 4.0,
        "w_macro_f1": 2.0,
        "w_stage1_auc_val": 1.0,
        "w_stage2_macro_f1_val": 1.0,

        # Negative terms
        "w_false_unknown_rate_all_known": 4.0,
        "w_false_unknown_rate_known_attacks": 3.0,
        "w_benign_family_fp_rate": 2.0,
        "w_overall_reject_rate": 1.0,
    },

    # Columns that should be preserved near the front of the summary tables.
    "front_columns": [
        "dataset",
        "holdout_family",
        "model_profile",
        "model_family",
        "apply_loao_stage1",
        "stage1_weight_mode",
        "stage2_weight_mode",
        "n_valid_known_families",
        "stage1_thr_high",
        "tau",
        "unknown_detection_rate",
        "false_unknown_rate_all_known",
        "false_unknown_rate_known_attacks",
        "benign_family_fp_rate",
        "overall_reject_rate",
        "macro_f1",
        "accuracy",
        "stage1_auc_val",
        "stage2_macro_f1_val",
        "run_name",
        "run_dir",
    ],
}


# =============================================================================
# HELPERS
# =============================================================================
def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def compute_rank_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute a practical ranking score.

    This is not a theorem. It is a readable decision aid.

    The score rewards:
      - unknown detection
n      - macro-F1
      - stronger validation behavior

    The score penalizes:
      - false unknowns on known traffic
      - benign-family false positives
      - overly reject-heavy settings
    """
    r = dict(CFG["ranking"])

    udr = safe_num(df.get("unknown_detection_rate", pd.Series([np.nan] * len(df)))).fillna(0.0)
    macro = safe_num(df.get("macro_f1", pd.Series([np.nan] * len(df)))).fillna(0.0)
    auc1 = safe_num(df.get("stage1_auc_val", pd.Series([np.nan] * len(df)))).fillna(0.0)
    f1_2 = safe_num(df.get("stage2_macro_f1_val", pd.Series([np.nan] * len(df)))).fillna(0.0)

    fur_all = safe_num(df.get("false_unknown_rate_all_known", pd.Series([np.nan] * len(df)))).fillna(1.0)
    fur_att = safe_num(df.get("false_unknown_rate_known_attacks", pd.Series([np.nan] * len(df)))).fillna(1.0)
    bfpr = safe_num(df.get("benign_family_fp_rate", pd.Series([np.nan] * len(df)))).fillna(1.0)
    orr = safe_num(df.get("overall_reject_rate", pd.Series([np.nan] * len(df)))).fillna(1.0)

    return (
        r["w_unknown_detection_rate"] * udr
        + r["w_macro_f1"] * macro
        + r["w_stage1_auc_val"] * auc1
        + r["w_stage2_macro_f1_val"] * f1_2
        - r["w_false_unknown_rate_all_known"] * fur_all
        - r["w_false_unknown_rate_known_attacks"] * fur_att
        - r["w_benign_family_fp_rate"] * bfpr
        - r["w_overall_reject_rate"] * orr
    )


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple Pareto shortlist.

    A row survives if no other row is at least as good on all major objectives and strictly
    better on at least one.

    Objectives used:
      maximize unknown_detection_rate
      maximize macro_f1
      minimize false_unknown_rate_all_known
      minimize benign_family_fp_rate
      minimize overall_reject_rate
    """
    if df.empty:
        return df.copy()

    cols = {
        "unknown_detection_rate": (True, safe_num(df["unknown_detection_rate"]).fillna(0.0).to_numpy()),
        "macro_f1": (True, safe_num(df["macro_f1"]).fillna(0.0).to_numpy()),
        "false_unknown_rate_all_known": (False, safe_num(df["false_unknown_rate_all_known"]).fillna(1.0).to_numpy()),
        "benign_family_fp_rate": (False, safe_num(df["benign_family_fp_rate"]).fillna(1.0).to_numpy()),
        "overall_reject_rate": (False, safe_num(df["overall_reject_rate"]).fillna(1.0).to_numpy()),
    }

    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            better_or_equal_all = True
            strictly_better_any = False
            for _, (maximize, values) in cols.items():
                vi = values[i]
                vj = values[j]
                if maximize:
                    if vj < vi - 1e-12:
                        better_or_equal_all = False
                        break
                    if vj > vi + 1e-12:
                        strictly_better_any = True
                else:
                    if vj > vi + 1e-12:
                        better_or_equal_all = False
                        break
                    if vj < vi - 1e-12:
                        strictly_better_any = True
            if better_or_equal_all and strictly_better_any:
                dominated = True
                break
        keep[i] = not dominated
    return df.loc[keep].copy()


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [c for c in CFG["front_columns"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    agg_path = str(CFG["aggregate_csv"])
    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"aggregate_csv not found: {agg_path}")

    out_dir = str(CFG["out_dir"])
    safe_mkdir(out_dir)

    df = pd.read_csv(agg_path)
    if df.empty:
        raise RuntimeError("Aggregate results CSV is empty.")
    if "model_profile" not in df.columns:
        df["model_profile"] = df.get("model_family", "unknown")
    if "stage2_weight_mode" not in df.columns:
        df["stage2_weight_mode"] = "balanced"

    # Numeric cleanup.
    for c in [
        "unknown_detection_rate",
        "false_unknown_rate_all_known",
        "false_unknown_rate_known_attacks",
        "benign_family_fp_rate",
        "overall_reject_rate",
        "macro_f1",
        "accuracy",
        "stage1_auc_val",
        "stage2_macro_f1_val",
        "tau",
        "stage1_thr_high",
    ]:
        if c in df.columns:
            df[c] = safe_num(df[c])

    df["rank_score"] = compute_rank_score(df)
    df = reorder_columns(df)

    full_sorted = df.sort_values(
        ["dataset", "holdout_family", "rank_score", "unknown_detection_rate", "macro_f1"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)
    full_sorted.to_csv(os.path.join(out_dir, "all_runs_ranked.csv"), index=False)

    # Best run per configured holdout grouping. The native thesis lane keeps one
    # winner per holdout; reference-profile runs can opt into one winner per
    # holdout/profile without changing the default behavior.
    group_cols = [str(c) for c in CFG.get("best_per_holdout_group_cols", ["dataset", "holdout_family"])]
    group_cols = [c for c in group_cols if c in full_sorted.columns]
    if not group_cols:
        group_cols = ["dataset", "holdout_family"]
    grp_holdout = (
        full_sorted.sort_values(["rank_score", "unknown_detection_rate", "macro_f1"], ascending=[False, False, False])
        .groupby(group_cols, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    grp_holdout.to_csv(os.path.join(out_dir, "best_per_holdout.csv"), index=False)

    # Best run per dataset + model family.
    grp_model = (
        full_sorted.sort_values(["rank_score", "unknown_detection_rate", "macro_f1"], ascending=[False, False, False])
        .groupby(["dataset", "model_profile"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    grp_model.to_csv(os.path.join(out_dir, "best_per_dataset_model.csv"), index=False)

    # Pareto shortlist.
    pf = pareto_front(full_sorted)
    pf = pf.sort_values(["dataset", "rank_score", "unknown_detection_rate", "macro_f1"], ascending=[True, False, False, False]).reset_index(drop=True)
    pf.to_csv(os.path.join(out_dir, "pareto_shortlist.csv"), index=False)

    # A compact model-family comparison table can be handy for supervisor meetings.
    compact = full_sorted[
        [c for c in [
            "dataset",
            "holdout_family",
            "model_profile",
            "model_family",
            "apply_loao_stage1",
            "stage1_weight_mode",
            "stage2_weight_mode",
            "unknown_detection_rate",
            "false_unknown_rate_all_known",
            "false_unknown_rate_known_attacks",
            "benign_family_fp_rate",
            "overall_reject_rate",
            "macro_f1",
            "accuracy",
            "stage1_auc_val",
            "stage2_macro_f1_val",
            "rank_score",
            "run_name",
        ] if c in full_sorted.columns]
    ].copy()
    compact.to_csv(os.path.join(out_dir, "compact_comparison_table.csv"), index=False)

    baseline_best_path = str(CFG.get("baseline_best_per_holdout_csv", ""))
    if baseline_best_path and os.path.exists(baseline_best_path):
        baseline_best = pd.read_csv(baseline_best_path)
        compare_cols = [
            c for c in [
                "dataset",
                "holdout_family",
                "model_profile",
                "model_family",
                "apply_loao_stage1",
                "stage1_weight_mode",
                "stage2_weight_mode",
                "unknown_detection_rate",
                "false_unknown_rate_all_known",
                "false_unknown_rate_known_attacks",
                "benign_family_fp_rate",
                "overall_reject_rate",
                "macro_f1",
                "accuracy",
                "stage1_auc_val",
                "stage2_macro_f1_val",
                "rank_score",
                "run_name",
            ]
            if c in grp_holdout.columns or c in baseline_best.columns
        ]
        combo = pd.concat(
            [
                baseline_best.reindex(columns=compare_cols),
                grp_holdout.reindex(columns=compare_cols),
            ],
            ignore_index=True,
        )
        combo.to_csv(os.path.join(out_dir, "two_dataset_best_per_holdout_comparison.csv"), index=False)

    # Small JSON snapshot for quick inspection.
    snapshot = {
        "n_runs": int(len(full_sorted)),
        "n_datasets": int(full_sorted["dataset"].nunique()) if "dataset" in full_sorted.columns else None,
        "n_holdouts": int(full_sorted[["dataset", "holdout_family"]].drop_duplicates().shape[0]) if {"dataset", "holdout_family"}.issubset(full_sorted.columns) else None,
        "top_5_runs": full_sorted.head(5).to_dict("records"),
        "baseline_best_per_holdout_csv": baseline_best_path if baseline_best_path and os.path.exists(baseline_best_path) else None,
    }
    with open(os.path.join(out_dir, "summary_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=lambda x: None if pd.isna(x) else x)

    print(f"Wrote summary files to: {out_dir}")


if __name__ == "__main__":
    main()
