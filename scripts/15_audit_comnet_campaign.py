#!/usr/bin/env python3
"""Audit the completed Computer Networks comparison campaign.

Descriptive only. Reads completed output artifacts and prints compact gate tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def show(title: str, frame: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if frame.empty:
        print("(no rows)")
        return
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 260,
        "display.max_colwidth", 60,
    ):
        print(frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def competence(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = repo / "outputs" / "12_model_competence" / "summary"
    alt = read_csv(root / "model_competence_summary.csv")
    primary = read_csv(root / "primary_vera_five_seed_reference.csv")

    if not alt.empty:
        keep_metrics = {
            "stage1_auc",
            "stage1_fpr",
            "stage2_macro_f1_present",
            "system_macro_f1_supported_labels",
            "system_accuracy",
            "system_benign_family_fp_rate",
            "strict_tau_macro_f1_supported_labels",
            "strict_tau_accuracy",
            "strict_tau_benign_family_fp_rate",
        }
        alt = alt[alt["metric"].astype(str).isin(keep_metrics)].copy()
        alt = alt[
            ["dataset", "model_family", "metric", "n", "mean", "sd", "min", "max"]
        ].sort_values(["dataset", "model_family", "metric"])

    if not primary.empty:
        keep_metrics = {
            "stage1_roc_auc",
            "stage2_macro_f1_present",
            "system_macro_f1_supported_labels",
            "system_accuracy",
            "benign_family_fp_rate",
        }
        primary = primary[
            primary["metric"].astype(str).isin(keep_metrics)
        ].copy()
        primary = primary[
            ["comparison_key", "dataset", "metric", "n", "mean", "sd", "min", "max"]
        ].sort_values(["dataset", "comparison_key", "metric"])
    return alt, primary


def external_a(repo: Path) -> pd.DataFrame:
    path = (
        repo
        / "outputs"
        / "13_external_profile_robustness"
        / "summary"
        / "protocol_a_external_profile_summary.csv"
    )
    frame = read_csv(path)
    if frame.empty:
        return frame
    keep_metrics = {
        "stage1_auc",
        "stage1_fpr",
        "stage2_macro_f1_present",
        "system_macro_f1_supported_labels",
        "system_accuracy",
        "system_benign_family_fp_rate",
        "system_overall_reject_rate",
    }
    frame = frame[frame["metric"].astype(str).isin(keep_metrics)].copy()
    return frame[
        ["model_profile", "paper", "dataset", "metric", "n_seeds", "mean", "sd", "min", "max"]
    ].sort_values(["dataset", "model_profile", "metric"])


def external_b(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = repo / "outputs" / "13_external_profile_robustness" / "summary"
    holdout = read_csv(root / "protocol_b_holdout_seed_summary.csv")
    hetero = read_csv(root / "protocol_b_family_heterogeneity_summary.csv")

    if not holdout.empty:
        keep_metrics = {
            "macro_f1",
            "unknown_detection_rate",
            "false_unknown_rate_all_known",
            "benign_family_fp_rate",
            "overall_reject_rate",
        }
        holdout = holdout[holdout["metric"].astype(str).isin(keep_metrics)].copy()
        holdout = holdout[
            [
                "model_profile",
                "paper",
                "dataset",
                "holdout_family",
                "metric",
                "n_seeds",
                "mean",
                "sd",
                "min",
                "max",
            ]
        ].sort_values(["dataset", "model_profile", "holdout_family", "metric"])

    if not hetero.empty:
        hetero = hetero[
            [
                "model_profile",
                "paper",
                "dataset",
                "metric",
                "summary",
                "n_seeds",
                "mean",
                "sd",
                "min",
                "max",
            ]
        ].sort_values(["dataset", "model_profile", "metric", "summary"])
    return holdout, hetero


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completed ComNet comparison campaign.")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()

    alt, primary = competence(repo)
    pa = external_a(repo)
    pb, hetero = external_b(repo)

    show("Competence alternatives: repeated runs", alt)
    show("Primary RF/XGB: existing five-seed reference", primary)
    show("External profiles: Protocol A repeated runs", pa)
    show("External profiles: Protocol B holdout summary", pb)
    show("External profiles: Protocol B family heterogeneity", hetero)

    print("\nGate checklist:")
    print(f"- competence repeated summary present: {not alt.empty}")
    print(f"- primary five-seed reference present: {not primary.empty}")
    print(f"- external Protocol A repeated summary present: {not pa.empty}")
    if not pa.empty:
        n = pd.to_numeric(pa["n_seeds"], errors="coerce").dropna()
        print(f"- external Protocol A seed count range: {int(n.min()) if len(n) else 0}..{int(n.max()) if len(n) else 0}")
    print(f"- external Protocol B summary present: {not pb.empty}")
    if not pb.empty:
        n = pd.to_numeric(pb["n_seeds"], errors="coerce").dropna()
        print(f"- external Protocol B seed count range: {int(n.min()) if len(n) else 0}..{int(n.max()) if len(n) else 0}")


if __name__ == "__main__":
    main()
