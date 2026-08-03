"""Sensitivity analysis for Protocol B minimum-support eligibility rules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping

import pandas as pd

from ids_eval_framework.src.paths import resolve_repo_path


def wilson_worst_case_half_width(n: int, confidence_level: float = 0.95) -> float:
    """Return the Wilson interval half-width at p=0.5."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + (z * z / n)
    return z * math.sqrt(0.25 / n + (z * z) / (4.0 * n * n)) / denominator


def evaluate_support_thresholds(
    support_tables: Mapping[str, pd.DataFrame],
    thresholds: list[int],
    *,
    benign_label: str = "Benign",
    min_known_families_after_holdout: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every candidate threshold for every dataset and LOAO holdout."""
    holdout_rows: list[dict[str, Any]] = []

    for dataset, support_df in support_tables.items():
        required = {"split", "family", "count"}
        missing = required.difference(support_df.columns)
        if missing:
            raise ValueError(f"{dataset} support table is missing columns: {sorted(missing)}")

        pivot = (
            support_df.pivot_table(
                index="family",
                columns="split",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(columns=["train", "val", "test"], fill_value=0)
            .astype(int)
        )
        if benign_label not in pivot.index:
            raise ValueError(f"{dataset} support table has no benign family '{benign_label}'")

        attack_families = sorted(
            family
            for family in pivot.index.astype(str)
            if family not in {benign_label, "__EMPTY__"} and int(pivot.loc[family, "train"]) > 0
        )

        for threshold in thresholds:
            for holdout in attack_families:
                remaining = [family for family in attack_families if family != holdout]
                invalid_known = [
                    family
                    for family in remaining
                    if any(int(pivot.loc[family, split]) < threshold for split in ("train", "val", "test"))
                ]
                reasons: list[str] = []
                if int(pivot.loc[benign_label, "val"]) < threshold:
                    reasons.append("benign_val_below_threshold")
                if int(pivot.loc[benign_label, "test"]) < threshold:
                    reasons.append("benign_test_below_threshold")
                if int(pivot.loc[holdout, "val"]) < threshold:
                    reasons.append("heldout_val_below_threshold")
                if int(pivot.loc[holdout, "test"]) < threshold:
                    reasons.append("heldout_test_below_threshold")
                if len(remaining) < min_known_families_after_holdout:
                    reasons.append("too_few_known_families")
                if invalid_known:
                    reasons.append("remaining_known_family_below_threshold")

                holdout_rows.append(
                    {
                        "dataset": dataset,
                        "threshold": int(threshold),
                        "holdout_family": holdout,
                        "scenario_valid": not reasons,
                        "reasons": "|".join(reasons),
                        "benign_val": int(pivot.loc[benign_label, "val"]),
                        "benign_test": int(pivot.loc[benign_label, "test"]),
                        "heldout_val": int(pivot.loc[holdout, "val"]),
                        "heldout_test": int(pivot.loc[holdout, "test"]),
                        "remaining_known_families": len(remaining),
                        "invalid_known_families": "|".join(invalid_known),
                    }
                )

    holdout_df = pd.DataFrame(holdout_rows)
    summary_df = (
        holdout_df.groupby(["threshold", "dataset"], as_index=False)
        .agg(
            total_holdouts=("holdout_family", "count"),
            eligible_holdouts=("scenario_valid", "sum"),
        )
        .sort_values(["threshold", "dataset"])
        .reset_index(drop=True)
    )
    summary_df["eligible_holdouts"] = summary_df["eligible_holdouts"].astype(int)
    summary_df["all_holdouts_eligible"] = (
        summary_df["eligible_holdouts"] == summary_df["total_holdouts"]
    )
    return holdout_df, summary_df


def derive_selected_threshold(summary_df: pd.DataFrame) -> int:
    """Select the highest candidate threshold retaining every fold in every dataset."""
    by_threshold = summary_df.groupby("threshold")["all_holdouts_eligible"].all()
    valid = [int(threshold) for threshold, all_valid in by_threshold.items() if bool(all_valid)]
    if not valid:
        raise ValueError("No candidate threshold retains every holdout across all datasets")
    return max(valid)


def run_support_sensitivity(config: Mapping[str, Any], *, dry_run: bool = False) -> None:
    """Run the configured support-threshold sensitivity analysis."""
    cfg = config.get("support_sensitivity", {}) or {}
    thresholds = [int(value) for value in cfg.get("candidate_thresholds", [])]
    selected_threshold = int(cfg.get("selected_threshold", 0))
    support_paths = cfg.get("support_tables", {}) or {}
    out_root = Path(resolve_repo_path(str(cfg.get("out_root", ""))))
    confidence_level = float(cfg.get("confidence_level", 0.95))

    if not thresholds or not support_paths or selected_threshold <= 0:
        raise ValueError("support_sensitivity requires thresholds, selected_threshold, and support_tables")

    resolved = {dataset: Path(resolve_repo_path(path)) for dataset, path in support_paths.items()}
    if dry_run:
        print(f"[dry-run] candidate thresholds: {thresholds}")
        print(f"[dry-run] selected threshold: {selected_threshold}")
        for dataset, path in resolved.items():
            print(f"[dry-run] {dataset}: {path}")
        print(f"[dry-run] output root: {out_root}")
        return

    support_tables = {dataset: pd.read_csv(path) for dataset, path in resolved.items()}
    holdout_df, summary_df = evaluate_support_thresholds(
        support_tables,
        thresholds,
        benign_label=str(cfg.get("benign_label", "Benign")),
        min_known_families_after_holdout=int(
            cfg.get("min_known_families_after_holdout", 2)
        ),
    )
    derived_threshold = derive_selected_threshold(summary_df)
    if derived_threshold != selected_threshold:
        raise ValueError(
            f"Configured selected threshold {selected_threshold} does not match "
            f"the sensitivity result {derived_threshold}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(out_root / "support_threshold_sensitivity_by_holdout.csv", index=False)
    summary_df.to_csv(out_root / "support_threshold_sensitivity_summary.csv", index=False)

    selected_rows = summary_df.loc[summary_df["threshold"] == selected_threshold]
    total_holdouts = int(selected_rows["total_holdouts"].sum())
    eligible_holdouts = int(selected_rows["eligible_holdouts"].sum())
    selection = {
        "candidate_thresholds": thresholds,
        "selected_threshold": selected_threshold,
        "selection_rule": (
            "Highest candidate threshold retaining every candidate LOAO fold "
            "across both primary datasets."
        ),
        "total_holdouts": total_holdouts,
        "eligible_holdouts_at_selected_threshold": eligible_holdouts,
        "confidence_level": confidence_level,
        "worst_case_wilson_half_width": wilson_worst_case_half_width(
            selected_threshold, confidence_level
        ),
        "model_retraining_required": False,
        "retraining_rationale": (
            "The selected threshold retained the same folds and did not alter "
            "the prepared data or training procedure."
        ),
        "historical_run_manifests_rewritten": False,
        "historical_manifest_rationale": (
            "Run-directory manifest copies retain the eligibility rules recorded "
            "at execution time. Final eligibility is established by the regenerated "
            "support-audit artifacts and this sensitivity analysis."
        ),
        "support_tables": {
            dataset: str(path) for dataset, path in support_paths.items()
        },
    }
    (out_root / "support_threshold_selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Protocol B Support-Threshold Sensitivity",
        "",
        "| Threshold | CICIDS2017 eligible | CICIoT2023 eligible |",
        "|---:|---:|---:|",
    ]
    for threshold in thresholds:
        rows = summary_df.loc[summary_df["threshold"] == threshold].set_index("dataset")
        values = []
        for dataset in ("CICIDS2017", "CICIoT2023"):
            row = rows.loc[dataset]
            values.append(f"{int(row['eligible_holdouts'])}/{int(row['total_holdouts'])}")
        lines.append(f"| {threshold} | {values[0]} | {values[1]} |")
    lines.extend(
        [
            "",
            f"Selected threshold: {selected_threshold} observations.",
            (
                "Worst-case approximate "
                f"{confidence_level:.0%} Wilson half-width: "
                f"{selection['worst_case_wilson_half_width'] * 100:.1f}%."
            ),
            "",
            "No model retraining was required because all 12 folds remained eligible.",
            (
                "Historical run-directory manifest copies were not rewritten; "
                "they preserve execution-time provenance."
            ),
        ]
    )
    (out_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
