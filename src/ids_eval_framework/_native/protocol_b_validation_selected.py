#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


CFG = {
    "aggregate_csvs": [
        Path("protocolB_grid_runs step 2 stage-1 LOAO/aggregate_results.csv"),
        Path("protocolB_grid_runs step 3 - CICIDS2017 sweep/aggregate_results.csv"),
    ],
    "out_root": Path("q2_validation_selected_protocol_b"),
    "run_roots": {
        "CICIoT2023": Path("protocolB_grid_runs step 2 stage-1 LOAO"),
        "CICIDS2017": Path("protocolB_grid_runs step 3 - CICIDS2017 sweep"),
    },
}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + (z * z / total)
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def add_interval(row: dict, rate_key: str, denominator: int, prefix: str) -> None:
    rate = float(row[rate_key])
    successes = int(round(rate * denominator))
    low, high = wilson_interval(successes, denominator)
    row[f"{prefix}_successes"] = successes
    row[f"{prefix}_denominator"] = denominator
    row[f"{prefix}_ci_low"] = low
    row[f"{prefix}_ci_high"] = high


def sanitize_token(value: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[\\/:*?\"<>|\s]+", "_", str(value).strip())).strip("._") or "NA"


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def resolve_run_dir(record: dict) -> Path:
    dataset = str(record["dataset"])
    root = CFG["run_roots"][dataset]
    required_tokens = [
        dataset,
        f"holdout_{record['holdout_family']}",
        str(record["model_family"]),
        f"loaoS1_{int(bool(record['apply_loao_stage1']))}",
        f"w_{record['stage1_weight_mode']}",
    ]
    candidates = []
    for matrix_path in root.rglob("confusion_matrix_system_test.csv"):
        normalized_path = normalize_token(matrix_path.parent.relative_to(root))
        if all(normalize_token(token) in normalized_path for token in required_tokens):
            candidates.append(matrix_path.parent)
    candidates = sorted(candidates)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one archived run for {required_tokens}, found {len(candidates)}")
    return candidates[0]


def sink_row(record: dict) -> dict:
    run_dir = resolve_run_dir(record)
    matrix_path = run_dir / "confusion_matrix_system_test.csv"
    matrix = pd.read_csv(matrix_path, index_col=0)
    unknown = matrix.loc["Unknown"].drop(labels=["Unknown"], errors="ignore")
    unknown = pd.to_numeric(unknown, errors="coerce").fillna(0).sort_values(ascending=False)
    total_unknown = int(pd.to_numeric(matrix.loc["Unknown"], errors="coerce").fillna(0).sum())
    top_label = str(unknown.index[0])
    top_count = int(unknown.iloc[0])
    return {
        "dataset": record["dataset"],
        "holdout_family": record["holdout_family"],
        "selected_run_name": run_dir.name,
        "top_sink_label": top_label,
        "top_sink_count": top_count,
        "top_sink_share": top_count / total_unknown if total_unknown else float("nan"),
        "n_true_unknown": total_unknown,
        "confusion_matrix_source": str(matrix_path),
    }


def main() -> None:
    out_root = CFG["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([pd.read_csv(path) for path in CFG["aggregate_csvs"]], ignore_index=True)
    for column in ("stage2_macro_f1_val", "stage1_auc_val"):
        combined[column] = pd.to_numeric(combined[column], errors="coerce")

    selected = (
        combined.sort_values(
            ["dataset", "holdout_family", "stage2_macro_f1_val", "stage1_auc_val", "run_name"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["dataset", "holdout_family"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected["selection_criterion"] = (
        "maximum validation Stage-2 macro-F1; tie-break maximum validation Stage-1 AUROC; "
        "final tie-break lexical run name"
    )
    selected["test_labels_used_for_selection"] = False

    output_rows = []
    sink_rows = []
    for record in selected.to_dict("records"):
        resolved_run_dir = resolve_run_dir(record)
        record["archived_run_name"] = resolved_run_dir.name
        record["archived_run_dir"] = str(resolved_run_dir)
        n_unknown = int(record["n_true_unknown"])
        n_known = int(record["n_known_all"])
        n_total = n_unknown + n_known
        add_interval(record, "unknown_detection_rate", n_unknown, "unknown_detection")
        add_interval(record, "false_unknown_rate_all_known", n_known, "false_unknown")
        add_interval(record, "overall_reject_rate", n_total, "reject_rate")
        output_rows.append(record)
        sink_rows.append(sink_row(record))

    pd.DataFrame(output_rows).to_csv(out_root / "validation_selected_protocol_b_results.csv", index=False)
    pd.DataFrame(sink_rows).to_csv(out_root / "validation_selected_sink_summary.csv", index=False)

    legacy = pd.read_csv("thesis_core_pack/protocol_b_best_per_holdout_standardized.csv")
    comparison = selected[["dataset", "holdout_family", "run_name"]].merge(
        legacy[["dataset", "holdout_family", "run_name"]].rename(columns={"run_name": "legacy_run_name"}),
        on=["dataset", "holdout_family"],
        how="left",
    )
    comparison["same_as_legacy_test_ranked_run"] = comparison["run_name"] == comparison["legacy_run_name"]
    comparison.to_csv(out_root / "legacy_selection_comparison.csv", index=False)


if __name__ == "__main__":
    main()
