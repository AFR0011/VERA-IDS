#!/usr/bin/env python3
"""Compare historical recovered CICIDS2017 Protocol B with the new group-safe RF surface.

Primary comparison rule
-----------------------
For each held-out family, use the stage-1 RF weighting mode that was selected in the
historical recovered-split evidence and match it to the same weighting lane on the
new B_group_safe surface. This freezes the modeling choice before looking at the new
test outcome and isolates the evaluation-surface change as cleanly as possible.

A secondary robustness table compares the two predeclared RF weighting lanes on the
new surface. No statistical significance claim is made by this script; it produces
descriptive paired changes only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [
    "unknown_detection_rate",
    "false_unknown_rate_all_known",
    "false_unknown_rate_known_attacks",
    "benign_family_fp_rate",
    "benign_reject_rate",
    "overall_reject_rate",
    "macro_f1",
    "accuracy",
    "stage1_auc_val",
    "stage2_macro_f1_val",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--historical",
        default="outputs/summaries/protocol_b_best_per_holdout.csv",
    )
    ap.add_argument(
        "--group-safe",
        default="outputs/12_jisa_finalization/06_group_safe_protocol_b_rf/aggregate_results.csv",
    )
    ap.add_argument(
        "--out-root",
        default="outputs/12_jisa_finalization/07_group_safe_comparison",
    )
    args = ap.parse_args()

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    old = pd.read_csv(args.historical)
    new = pd.read_csv(args.group_safe)

    old = old.loc[
        (old["dataset"].astype(str) == "CICIDS2017")
        & (old["model_family"].astype(str) == "rf")
    ].copy()
    if "split_variant" in old.columns:
        old = old.loc[
            old["split_variant"].astype(str).str.contains(
                "recovered contiguous-within-day", case=False, na=False
            )
        ].copy()

    new = new.loc[
        (new["dataset"].astype(str) == "CICIDS2017")
        & (new["model_family"].astype(str) == "rf")
    ].copy()

    if len(old) != 6:
        raise RuntimeError(f"Expected 6 historical CICIDS2017 RF rows, found {len(old)}")
    if len(new) != 12:
        raise RuntimeError(f"Expected 12 group-safe CICIDS2017 RF rows, found {len(new)}")

    matched_rows = []
    for _, o in old.sort_values("holdout_family").iterrows():
        fam = str(o["holdout_family"])
        weight = str(o["stage1_weight_mode"])
        cand = new.loc[
            (new["holdout_family"].astype(str) == fam)
            & (new["stage1_weight_mode"].astype(str) == weight)
        ]
        if len(cand) != 1:
            raise RuntimeError(
                f"Expected exactly one group-safe match for {fam}/{weight}, found {len(cand)}"
            )
        n = cand.iloc[0]
        row = {
            "holdout_family": fam,
            "matched_stage1_weight_mode": weight,
            "historical_protocol": str(o.get("protocol", "B_day_file")),
            "historical_split_variant": str(o.get("split_variant", "recovered contiguous-within-day")),
            "new_protocol": "B_group_safe",
            "historical_stage1_thr_high": o.get("stage1_thr_high"),
            "new_stage1_thr_high": n.get("stage1_thr_high"),
            "historical_tau": o.get("tau"),
            "new_tau": n.get("tau"),
            "historical_n_true_unknown": o.get("n_true_unknown"),
            "new_n_true_unknown": n.get("n_true_unknown"),
        }
        for metric in METRICS:
            ov = pd.to_numeric(pd.Series([o.get(metric)]), errors="coerce").iloc[0]
            nv = pd.to_numeric(pd.Series([n.get(metric)]), errors="coerce").iloc[0]
            row[f"historical_{metric}"] = ov
            row[f"new_{metric}"] = nv
            row[f"delta_{metric}"] = nv - ov if pd.notna(ov) and pd.notna(nv) else np.nan
            row[f"abs_delta_{metric}"] = abs(nv - ov) if pd.notna(ov) and pd.notna(nv) else np.nan
        matched_rows.append(row)

    matched = pd.DataFrame(matched_rows).sort_values("holdout_family").reset_index(drop=True)
    matched.to_csv(out / "matched_surface_comparison.csv", index=False)

    metric_summary = []
    for metric in METRICS:
        deltas = pd.to_numeric(matched[f"delta_{metric}"], errors="coerce")
        abs_deltas = pd.to_numeric(matched[f"abs_delta_{metric}"], errors="coerce")
        old_vals = pd.to_numeric(matched[f"historical_{metric}"], errors="coerce")
        new_vals = pd.to_numeric(matched[f"new_{metric}"], errors="coerce")
        metric_summary.append(
            {
                "metric": metric,
                "historical_mean": float(old_vals.mean()),
                "group_safe_mean": float(new_vals.mean()),
                "mean_delta": float(deltas.mean()),
                "mean_absolute_delta": float(abs_deltas.mean()),
                "max_absolute_delta": float(abs_deltas.max()),
            }
        )
    pd.DataFrame(metric_summary).to_csv(out / "matched_metric_summary.csv", index=False)

    # New-surface weighting-lane sensitivity, paired by holdout.
    lane_rows = []
    for fam, grp in new.groupby("holdout_family", sort=True):
        if len(grp) != 2:
            raise RuntimeError(f"Expected two RF weighting lanes for {fam}, found {len(grp)}")
        a = grp.loc[grp["stage1_weight_mode"].astype(str) == "class_weight_balanced"]
        b = grp.loc[grp["stage1_weight_mode"].astype(str) == "inv_family_clipped"]
        if len(a) != 1 or len(b) != 1:
            raise RuntimeError(f"Missing expected RF weighting lanes for {fam}")
        a, b = a.iloc[0], b.iloc[0]
        row = {"holdout_family": str(fam)}
        for metric in METRICS:
            av = pd.to_numeric(pd.Series([a.get(metric)]), errors="coerce").iloc[0]
            bv = pd.to_numeric(pd.Series([b.get(metric)]), errors="coerce").iloc[0]
            row[f"class_weight_balanced_{metric}"] = av
            row[f"inv_family_clipped_{metric}"] = bv
            row[f"absolute_lane_difference_{metric}"] = abs(av - bv) if pd.notna(av) and pd.notna(bv) else np.nan
        lane_rows.append(row)
    lane = pd.DataFrame(lane_rows).sort_values("holdout_family").reset_index(drop=True)
    lane.to_csv(out / "new_surface_weighting_lane_sensitivity.csv", index=False)

    summary = {
        "matched_holdouts": int(len(matched)),
        "historical_selection_frozen_before_new_results": True,
        "primary_comparison_rule": "match each holdout to its historically selected RF stage-1 weighting mode",
        "mean_absolute_udr_surface_shift": float(matched["abs_delta_unknown_detection_rate"].mean()),
        "max_absolute_udr_surface_shift": float(matched["abs_delta_unknown_detection_rate"].max()),
        "mean_absolute_udr_weighting_lane_difference_new_surface": float(
            lane["absolute_lane_difference_unknown_detection_rate"].mean()
        ),
        "max_absolute_udr_weighting_lane_difference_new_surface": float(
            lane["absolute_lane_difference_unknown_detection_rate"].max()
        ),
        "descriptive_spearman_udr_old_vs_new": float(
            matched["historical_unknown_detection_rate"].corr(
                matched["new_unknown_detection_rate"], method="spearman"
            )
        ),
        "interpretation_boundary": (
            "Descriptive paired comparison only. The split change also changes the sampled train/validation/test observations and unknown denominators; "
            "this script does not assign causal effects or statistical significance to the split policy."
        ),
    }
    (out / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nMatched UDR comparison:")
    print(
        matched[
            [
                "holdout_family",
                "matched_stage1_weight_mode",
                "historical_unknown_detection_rate",
                "new_unknown_detection_rate",
                "delta_unknown_detection_rate",
                "historical_macro_f1",
                "new_macro_f1",
                "delta_macro_f1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
