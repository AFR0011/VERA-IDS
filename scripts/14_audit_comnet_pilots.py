#!/usr/bin/env python3
"""Summarize the completed Computer Networks comparison pilots for review.

This script is descriptive only. It deliberately does not auto-declare a model
"competent" or trigger the five-seed campaign.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required pilot output not found: {path}")
    return pd.read_csv(path)


def competence_table(repo: Path, dataset: str, seed: int) -> pd.DataFrame:
    root = repo / "outputs" / "12_model_competence" / "summary"
    alternatives = read_csv(root / "model_competence_runs.csv")
    alternatives = alternatives[
        (alternatives["dataset"].astype(str) == dataset)
        & (pd.to_numeric(alternatives["seed"], errors="coerce") == int(seed))
    ].copy()
    alternatives["source"] = "new_alternative"

    primary = read_csv(root / "primary_vera_single_run_reference.csv")
    primary = primary[
        (primary["dataset"].astype(str) == dataset)
        & (primary["model_family"].astype(str).isin(["rf", "xgb"]))
        & (primary["policy_variant"].astype(str) == "strict")
    ].copy()
    primary = primary.rename(
        columns={
            "stage1_roc_auc": "stage1_auc",
            "benign_family_fp_rate": "system_benign_family_fp_rate",
        }
    )
    primary["source"] = "primary_vera_reference"
    primary["seed"] = int(seed)

    cols = [
        "source",
        "model_family",
        "dataset",
        "seed",
        "stage1_auc",
        "stage1_fpr",
        "stage1_tpr",
        "stage2_macro_f1_present",
        "stage2_accuracy",
        "system_macro_f1_supported_labels",
        "system_accuracy",
        "system_benign_family_fp_rate",
    ]
    combined = pd.concat(
        [
            primary.reindex(columns=cols),
            alternatives.reindex(columns=cols),
        ],
        ignore_index=True,
    )

    numeric = [
        "stage1_auc",
        "stage1_fpr",
        "stage2_macro_f1_present",
        "system_macro_f1_supported_labels",
        "system_accuracy",
        "system_benign_family_fp_rate",
    ]
    for col in numeric:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    primary_rows = combined[combined["source"] == "primary_vera_reference"]
    if not primary_rows.empty:
        best_stage2 = primary_rows["stage2_macro_f1_present"].max()
        best_system = primary_rows["system_macro_f1_supported_labels"].max()
        lowest_fpr = primary_rows["system_benign_family_fp_rate"].min()
        combined["delta_vs_best_primary_stage2_f1"] = (
            combined["stage2_macro_f1_present"] - best_stage2
        )
        combined["delta_vs_best_primary_system_f1"] = (
            combined["system_macro_f1_supported_labels"] - best_system
        )
        combined["delta_vs_lowest_primary_system_fpr"] = (
            combined["system_benign_family_fp_rate"] - lowest_fpr
        )
    return combined.sort_values(["source", "model_family"]).reset_index(drop=True)


def external_protocol_a_table(repo: Path, seed: int) -> pd.DataFrame:
    aggregate = (
        repo
        / "outputs"
        / "13_external_profile_robustness"
        / "summary"
        / "protocol_a_external_profile_runs.csv"
    )
    if aggregate.exists():
        frame = read_csv(aggregate)
    else:
        frame = read_csv(
            repo
            / "outputs"
            / "13_external_profile_robustness"
            / f"seed_{seed}"
            / "protocol_a"
            / "summary"
            / "protocol_a_reference_profile_summary.csv"
        )
        if "seed" not in frame.columns:
            frame.insert(0, "seed", int(seed))

    if "seed" in frame.columns:
        frame = frame[
            pd.to_numeric(frame["seed"], errors="coerce") == int(seed)
        ].copy()

    cols = [
        "model_profile",
        "paper",
        "dataset",
        "stage1_auc",
        "stage1_fpr",
        "stage1_tpr",
        "stage1_threshold",
        "stage2_macro_f1_present",
        "stage2_accuracy",
        "system_macro_f1_supported_labels",
        "system_accuracy",
        "system_benign_family_fp_rate",
        "system_overall_reject_rate",
    ]
    out = frame.reindex(columns=cols).copy()
    out["component_to_system_f1_delta"] = (
        pd.to_numeric(out["system_macro_f1_supported_labels"], errors="coerce")
        - pd.to_numeric(out["stage2_macro_f1_present"], errors="coerce")
    )
    return out.sort_values(
        ["dataset", "model_profile"]
    ).reset_index(drop=True)


def print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if frame.empty:
        print("(no rows)")
        return
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.max_colwidth", 48,
    ):
        print(frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit completed ComNet pilot outputs.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset", default="CICIoT2023")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out = repo / "outputs" / "14_comnet_pilot_audit"
    out.mkdir(parents=True, exist_ok=True)

    competence = competence_table(repo, args.dataset, args.seed)
    external = external_protocol_a_table(repo, args.seed)

    competence.to_csv(out / f"competence_{args.dataset}_seed_{args.seed}.csv", index=False)
    external.to_csv(out / f"external_protocol_a_seed_{args.seed}.csv", index=False)

    print_table(f"Competence pilot: {args.dataset}, seed {args.seed}", competence)
    print_table(f"External Protocol A pilot: seed {args.seed}", external)
    print(f"\nWrote pilot audit CSVs to: {out}")


if __name__ == "__main__":
    main()
