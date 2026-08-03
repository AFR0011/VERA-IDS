#!/usr/bin/env python3
"""
8.BuildThesisCorePack.py
========================

Purpose
-------
Build one thesis-facing comparison pack from the current Protocol A and Protocol B artifacts.

Outputs
-------
This script materializes:
    - a normalized Protocol A summary export
    - a standardized Protocol B best-per-holdout table
    - a two-dataset best-per-holdout comparison table with explicit split variants
    - a failure-mode table derived from the winner confusion matrices
    - narrative holdout notes (strong / mixed / hard failure)
    - step-4 targeted follow-up frontier and disposition tables when available
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


CFG: Dict[str, object] = {
    "protocol_a_summary_csv": os.path.join(
        "runs_two_stage_V5_A_core",
        "summary",
        "protocol_a_core_summary.csv",
    ),
    "protocol_b_sources": [
        {
            "best_csv": os.path.join(
                "protocolB_grid_runs step 2 stage-1 LOAO",
                "summary",
                "best_per_holdout.csv",
            ),
            "protocol": "B_day_file",
            "split_variant": "day-file Protocol B baseline",
        },
        {
            "best_csv": os.path.join(
                "protocolB_grid_runs step 3 - CICIDS2017 sweep",
                "summary",
                "best_per_holdout.csv",
            ),
            "protocol": "B_day_file",
            "split_variant": "recovered contiguous-within-day Protocol B variant",
        },
    ],
    "step4_aggregate_csv": os.path.join(
        "protocolB_grid_runs step 4 - targeted followups",
        "aggregate_results.csv",
    ),
    "step4_runs_root": "protocolB_grid_runs step 4 - targeted followups",
    "validation_selected_protocol_b_csv": None,
    "validation_selected_sink_csv": None,
    "out_root": "thesis_core_pack",
}


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_num(series: pd.Series, fill: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill)


def resolve_repo_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.path.dirname(__file__), path))


def load_protocol_a_summary() -> pd.DataFrame:
    path = resolve_repo_path(str(CFG["protocol_a_summary_csv"]))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Protocol A summary not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("Protocol A summary CSV is empty.")
    return df


def load_protocol_b_best_rows() -> pd.DataFrame:
    validation_selected_csv = CFG.get("validation_selected_protocol_b_csv")
    if validation_selected_csv:
        path = resolve_repo_path(str(validation_selected_csv))
        if os.path.exists(path):
            out = pd.read_csv(path)
            if out.empty:
                raise RuntimeError("Validation-selected Protocol B CSV is empty.")
            if "protocol" not in out.columns:
                out["protocol"] = "B_day_file"
            if "split_variant" not in out.columns:
                out["split_variant"] = out["dataset"].map(
                    {
                        "CICIoT2023": "day-file Protocol B baseline",
                        "CICIDS2017": "recovered contiguous-within-day Protocol B variant",
                    }
                ).fillna("validation-selected Protocol B")
            out["selection_source"] = "validation_selected_protocol_b_results.csv"
            front = [
                "dataset",
                "protocol",
                "split_variant",
                "holdout_family",
                "model_family",
                "stage1_weight_mode",
                "apply_loao_stage1",
                "unknown_detection_rate",
                "unknown_detection_ci_low",
                "unknown_detection_ci_high",
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
                "run_dir",
                "selection_criterion",
                "test_labels_used_for_selection",
            ]
            keep = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
            return out[keep].copy()

    rows = []
    for src in CFG["protocol_b_sources"]:
        path = resolve_repo_path(str(src["best_csv"]))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Protocol B best-per-holdout CSV not found: {path}")
        df = pd.read_csv(path)
        df["protocol"] = str(src["protocol"])
        df["split_variant"] = str(src["split_variant"])
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    if out.empty:
        raise RuntimeError("Combined Protocol B best-per-holdout table is empty.")
    front = [
        "dataset",
        "protocol",
        "split_variant",
        "holdout_family",
        "model_family",
        "stage1_weight_mode",
        "apply_loao_stage1",
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
        "run_dir",
    ]
    keep = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
    return out[keep].copy()


def read_confusion_csv(run_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(resolve_repo_path(run_dir), "confusion_matrix_system_test.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)


def extract_failure_modes(best_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rec in best_df.to_dict("records"):
        run_dir = str(rec.get("run_dir", ""))
        cm = read_confusion_csv(run_dir)
        if cm is None or "Unknown" not in cm.index:
            rows.append(
                {
                    "dataset": rec.get("dataset"),
                    "holdout_family": rec.get("holdout_family"),
                    "winner_run_name": rec.get("run_name"),
                    "n_true_unknown": rec.get("n_true_unknown"),
                    "top_sink_family": None,
                    "top_sink_count": None,
                    "top_sink_share": None,
                    "secondary_sink_family": None,
                    "secondary_sink_share": None,
                }
            )
            continue

        unknown_row = cm.loc["Unknown"].copy()
        n_true_unknown = int(unknown_row.sum())
        sink_series = unknown_row.drop(labels=["Unknown"], errors="ignore").sort_values(ascending=False)

        top_sink_family = sink_series.index[0] if len(sink_series) >= 1 else None
        top_sink_count = int(sink_series.iloc[0]) if len(sink_series) >= 1 else None
        secondary_sink_family = sink_series.index[1] if len(sink_series) >= 2 else None
        secondary_sink_count = int(sink_series.iloc[1]) if len(sink_series) >= 2 else None

        rows.append(
            {
                "dataset": rec.get("dataset"),
                "holdout_family": rec.get("holdout_family"),
                "winner_run_name": rec.get("run_name"),
                "n_true_unknown": n_true_unknown,
                "top_sink_family": top_sink_family,
                "top_sink_count": top_sink_count,
                "top_sink_share": (top_sink_count / n_true_unknown) if n_true_unknown and top_sink_count is not None else None,
                "secondary_sink_family": secondary_sink_family,
                "secondary_sink_share": (secondary_sink_count / n_true_unknown) if n_true_unknown and secondary_sink_count is not None else None,
            }
        )
    return pd.DataFrame(rows)


def load_validation_selected_failure_modes() -> Optional[pd.DataFrame]:
    sink_csv = CFG.get("validation_selected_sink_csv")
    if not sink_csv:
        return None
    path = resolve_repo_path(str(sink_csv))
    if not os.path.exists(path):
        return None
    sink = pd.read_csv(path)
    if sink.empty:
        return None
    return pd.DataFrame(
        {
            "dataset": sink.get("dataset"),
            "holdout_family": sink.get("holdout_family"),
            "winner_run_name": sink.get("selected_run_name"),
            "n_true_unknown": sink.get("n_true_unknown"),
            "top_sink_family": sink.get("top_sink_label"),
            "top_sink_count": sink.get("top_sink_count"),
            "top_sink_share": sink.get("top_sink_share"),
            "secondary_sink_family": None,
            "secondary_sink_share": None,
            "confusion_matrix_source": sink.get("confusion_matrix_source"),
        }
    )


def classify_holdout_case(udr: float, fur_all: float) -> str:
    if pd.isna(udr):
        return "unknown"
    if udr >= 0.75 and fur_all <= 0.01:
        return "strong case"
    if udr < 0.10:
        return "hard failure case"
    return "mixed case"


def build_holdout_notes(best_df: pd.DataFrame, failure_df: pd.DataFrame) -> pd.DataFrame:
    work = best_df.copy()
    if "winner_run_name" not in work.columns and "run_name" in work.columns:
        work = work.rename(columns={"run_name": "winner_run_name"})
    merged = work.merge(failure_df, on=["dataset", "holdout_family"], how="left")
    notes = []
    for rec in merged.to_dict("records"):
        case = classify_holdout_case(
            float(rec.get("unknown_detection_rate", np.nan)),
            float(rec.get("false_unknown_rate_all_known", np.nan)),
        )
        sink = rec.get("top_sink_family")
        share = rec.get("top_sink_share")
        if case == "strong case":
            note = "Unknown family is mostly retained as Unknown under the current winner."
        elif case == "hard failure case" and sink:
            note = f"True unknown traffic is absorbed primarily into {sink} ({share:.3f} of unknown samples)." if share is not None else f"True unknown traffic is absorbed primarily into {sink}."
        elif sink:
            note = f"Unknown handling is usable but uneven; the dominant sink is {sink} ({share:.3f} of unknown samples)." if share is not None else f"Unknown handling is usable but uneven; the dominant sink is {sink}."
        else:
            note = "Unknown handling remains mixed and needs cautious interpretation."

        notes.append(
            {
                "dataset": rec.get("dataset"),
                "holdout_family": rec.get("holdout_family"),
                "holdout_case_note": case,
                "narrative_note": note,
            }
        )
    return pd.DataFrame(notes)


def pareto_mask(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros(0, dtype=bool)
    udr = to_num(df["unknown_detection_rate"], 0.0).to_numpy()
    macro = to_num(df["macro_f1"], 0.0).to_numpy()
    fur_all = to_num(df["false_unknown_rate_all_known"], 1.0).to_numpy()
    bfpr = to_num(df["benign_family_fp_rate"], 1.0).to_numpy()
    orr = to_num(df["overall_reject_rate"], 1.0).to_numpy()

    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            better_or_equal = (
                (udr[j] >= udr[i] - 1e-12)
                and (macro[j] >= macro[i] - 1e-12)
                and (fur_all[j] <= fur_all[i] + 1e-12)
                and (bfpr[j] <= bfpr[i] + 1e-12)
                and (orr[j] <= orr[i] + 1e-12)
            )
            strictly_better = (
                (udr[j] > udr[i] + 1e-12)
                or (macro[j] > macro[i] + 1e-12)
                or (fur_all[j] < fur_all[i] - 1e-12)
                or (bfpr[j] < bfpr[i] - 1e-12)
                or (orr[j] < orr[i] - 1e-12)
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        keep[i] = not dominated
    return keep


def followup_disposition(step3_row: pd.Series, step4_row: Optional[pd.Series]) -> Dict[str, object]:
    holdout = str(step3_row["holdout_family"])
    if step4_row is None:
        if float(step3_row["unknown_detection_rate"]) < 0.10:
            return {"holdout_family": holdout, "disposition": "document as hard case", "reason": "No targeted follow-up result was available and the current winner remains a hard open-set failure."}
        return {"holdout_family": holdout, "disposition": "freeze current winner", "reason": "No targeted follow-up result was available, so the existing winner remains the active thesis reference."}

    step3_udr = float(step3_row["unknown_detection_rate"])
    step4_udr = float(step4_row["unknown_detection_rate"])
    step3_macro = float(step3_row["macro_f1"])
    step4_macro = float(step4_row["macro_f1"])
    step3_fur = float(step3_row["false_unknown_rate_all_known"])
    step4_fur = float(step4_row["false_unknown_rate_all_known"])

    if step4_udr < 0.05 and step3_udr < 0.05:
        return {"holdout_family": holdout, "disposition": "document as hard case", "reason": "The targeted follow-up still leaves unknown detection near zero under the active reject constraints."}

    if step4_udr >= step3_udr + 0.05 and step4_fur <= step3_fur + 0.01 and step4_macro >= step3_macro - 0.03:
        return {"holdout_family": holdout, "disposition": "replace winner", "reason": "The targeted follow-up improved unknown detection materially without an unacceptable tradeoff in false-unknown rate or macro-F1."}

    return {"holdout_family": holdout, "disposition": "freeze current winner", "reason": "The targeted follow-up did not produce a materially better operational tradeoff than the current step-3 winner."}


def build_step4_outputs(best_df: pd.DataFrame, out_root: str) -> None:
    path = resolve_repo_path(str(CFG["step4_aggregate_csv"]))
    if not os.path.exists(path):
        return

    follow_df = pd.read_csv(path)
    if follow_df.empty:
        return

    frontier_rows = []
    runs_root = resolve_repo_path(str(CFG["step4_runs_root"]))
    for rec in follow_df.to_dict("records"):
        run_dir = resolve_repo_path(str(rec["run_dir"]))
        frontier_path = os.path.join(run_dir, "frontier_table.csv")
        if not os.path.exists(frontier_path):
            continue
        df = pd.read_csv(frontier_path)
        df["run_name"] = rec["run_name"]
        frontier_rows.append(df)

    if frontier_rows:
        frontier = pd.concat(frontier_rows, ignore_index=True)
        frontier["is_frontier"] = pareto_mask(frontier)
        summary_dir = os.path.join(runs_root, "summary")
        safe_mkdir(summary_dir)
        frontier.to_csv(os.path.join(summary_dir, "targeted_followup_frontier.csv"), index=False)
        frontier.to_csv(os.path.join(out_root, "targeted_followup_frontier.csv"), index=False)

    step3_target = best_df.loc[(best_df["dataset"] == "CICIDS2017") & (best_df["holdout_family"].isin(["DDoS", "Web/App", "Botnet"]))].copy()
    dispositions = []
    for _, step3_row in step3_target.iterrows():
        match = follow_df.loc[follow_df["holdout_family"] == step3_row["holdout_family"]].copy()
        step4_row = match.iloc[0] if not match.empty else None
        dispositions.append(followup_disposition(step3_row, step4_row))
    disp_df = pd.DataFrame(dispositions)
    summary_dir = os.path.join(runs_root, "summary")
    safe_mkdir(summary_dir)
    disp_df.to_csv(os.path.join(summary_dir, "holdout_dispositions.csv"), index=False)
    disp_df.to_csv(os.path.join(out_root, "targeted_followup_dispositions.csv"), index=False)


def main() -> None:
    out_root = resolve_repo_path(str(CFG["out_root"]))
    safe_mkdir(out_root)

    protocol_a = load_protocol_a_summary()
    protocol_b = load_protocol_b_best_rows()
    failure_df = load_validation_selected_failure_modes()
    if failure_df is None:
        failure_df = extract_failure_modes(protocol_b)
    notes_df = build_holdout_notes(protocol_b, failure_df)

    protocol_a.to_csv(os.path.join(out_root, "protocol_a_core_summary.csv"), index=False)
    protocol_b.to_csv(os.path.join(out_root, "protocol_b_best_per_holdout_standardized.csv"), index=False)
    protocol_b.to_csv(os.path.join(out_root, "two_dataset_protocol_b_best_per_holdout_comparison.csv"), index=False)
    failure_df.to_csv(os.path.join(out_root, "protocol_b_failure_modes.csv"), index=False)
    notes_df.to_csv(os.path.join(out_root, "protocol_b_holdout_notes.csv"), index=False)

    build_step4_outputs(protocol_b, out_root)

    snapshot = {
        "protocol_a_summary_csv": str(CFG["protocol_a_summary_csv"]),
        "protocol_b_sources": [src["best_csv"] for src in CFG["protocol_b_sources"]],
        "validation_selected_protocol_b_csv": str(CFG.get("validation_selected_protocol_b_csv")),
        "validation_selected_sink_csv": str(CFG.get("validation_selected_sink_csv")),
        "n_protocol_a_rows": int(len(protocol_a)),
        "n_protocol_b_rows": int(len(protocol_b)),
    }
    with open(os.path.join(out_root, "pack_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Wrote thesis core pack to: {out_root}")


if __name__ == "__main__":
    main()
