#!/usr/bin/env python3
"""Build auditable source tables for the final JISA main-text result figures.

This script performs no training, model selection, threshold selection, or new inference.
It transforms frozen VERA-IDS result artifacts into compact plotting tables for Figures
2-11 of the journal manuscript and records SHA256 hashes of every source artifact used.
Figure 1 is the non-numerical methodology overview and is intentionally handled outside
this numerical plotting pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/12_jisa_finalization/20_main_figure_sources"

PROTOCOL_A_CORE = ROOT / "outputs/summaries/protocol_a_core_summary.csv"
PROTOCOL_A_FLAT = ROOT / "outputs/summaries/protocol_a_flat_vs_two_stage.csv"
REFERENCE_DROP = ROOT / "outputs/summaries/reference_profile_metric_drop.csv"
REFERENCE_PROTOCOL_B = ROOT / "outputs/11_reference_framework_eval/comparison/protocol_b_holdout_stress_table.csv"
SURFACE = ROOT / "outputs/12_jisa_finalization/07_group_safe_comparison/matched_surface_comparison.csv"
KNOWN_HELDOUT = ROOT / "outputs/12_jisa_finalization/11_known_vs_heldout/known_vs_heldout_primary_strict.csv"
FAIL_DEST = ROOT / "outputs/12_jisa_finalization/12_failure_destinations/destination_distribution_per_run.csv"
REJECTOR = ROOT / "outputs/12_jisa_finalization/15_rejector_tradeoff_analysis/primary_3pct_case_metrics.csv"
BUDGET_AUDIT = ROOT / "outputs/12_jisa_finalization/16_rejector_budget_audit/primary_3pct_test_constraint_generalization.csv"

DEFAULT_RECOVERED_OVERLAP_ROOT = ROOT / "q2_reproducibility_audit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frozen source tables for JISA Figures 2-11.")
    parser.add_argument(
        "--recovered-overlap-root",
        default=str(DEFAULT_RECOVERED_OVERLAP_ROOT),
        help=(
            "Directory containing cross_split_exact_duplicate_audit.csv and "
            "split_hash_summary.csv for the recovered CICIDS2017 Protocol-B surface."
        ),
    )
    return parser


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def boolify(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def build_fig2_closed_set() -> pd.DataFrame:
    core = pd.read_csv(require(PROTOCOL_A_CORE)).copy()
    flat = pd.read_csv(require(PROTOCOL_A_FLAT)).copy()
    numeric(core, ["stage2_macro_f1_present", "system_macro_f1_supported_labels"])
    numeric(flat, ["competitive_macro_f1"])

    rows: list[dict[str, object]] = []
    for dataset in ("CICIDS2017", "CICIoT2023"):
        direct = flat.loc[flat["dataset"].astype(str) == dataset, "competitive_macro_f1"]
        if len(direct) != 1:
            raise RuntimeError(f"Expected one direct multiclass row for {dataset}, found {len(direct)}")
        direct_f1 = float(direct.iloc[0])
        for model in ("rf", "xgb"):
            sub = core.loc[
                (core["dataset"].astype(str) == dataset)
                & (core["model_family"].astype(str) == model)
            ].copy()
            strict = sub.loc[sub["policy_variant"].astype(str) == "strict"]
            strict_tau = sub.loc[sub["policy_variant"].astype(str) == "strict_tau"]
            if len(strict) != 1 or len(strict_tau) != 1:
                raise RuntimeError(f"Missing strict/strict_tau row for {dataset}/{model}")
            s = strict.iloc[0]
            st = strict_tau.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "model_family": model,
                    "stage2_macro_f1": float(s["stage2_macro_f1_present"]),
                    "final_strict_macro_f1": float(s["system_macro_f1_supported_labels"]),
                    "final_strict_tau_macro_f1": float(st["system_macro_f1_supported_labels"]),
                    "direct_multiclass_macro_f1": direct_f1,
                }
            )
    return pd.DataFrame(rows)


def build_fig3_reference_closed_set() -> pd.DataFrame:
    df = pd.read_csv(require(REFERENCE_DROP)).copy()
    sub = df[df["full_framework_surface"].astype(str) == "protocol_a_reference_profile"].copy()
    numeric(sub, ["closed_set_macro_f1", "full_framework_macro_f1_supported_labels"])
    cols = [
        "paper",
        "model_profile",
        "dataset",
        "closed_set_task_used",
        "closed_set_macro_f1",
        "full_framework_macro_f1_supported_labels",
    ]
    out = sub[cols].copy()
    out["delta_protocol_a_minus_reference"] = (
        out["full_framework_macro_f1_supported_labels"] - out["closed_set_macro_f1"]
    )
    if len(out) != 3:
        raise RuntimeError(f"Expected 3 Protocol-A reference-profile rows, found {len(out)}")
    return out.sort_values(["paper", "dataset"]).reset_index(drop=True)


def build_fig4_reference_protocol_b() -> pd.DataFrame:
    df = pd.read_csv(require(REFERENCE_PROTOCOL_B)).copy()
    numeric(df, ["unknown_detection_rate", "macro_f1", "accuracy", "overall_reject_rate"])
    required = ["paper", "model_profile", "dataset", "holdout_family", "unknown_detection_rate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Reference Protocol-B table missing columns: {missing}")
    out = df.copy()
    out = out[out["unknown_detection_rate"].notna()].copy()
    if out.empty:
        raise RuntimeError("Reference Protocol-B table contains no UDR values")
    keep = [c for c in [
        "paper",
        "model_profile",
        "dataset",
        "holdout_family",
        "unknown_detection_rate",
        "macro_f1",
        "accuracy",
        "overall_reject_rate",
        "stage1_auc_val",
        "stage2_macro_f1_val",
    ] if c in out.columns]
    return out[keep].sort_values(["paper", "dataset", "holdout_family"]).reset_index(drop=True)


def locate_recovered_overlap_root(requested: Path) -> Path:
    expected = {"cross_split_exact_duplicate_audit.csv", "split_hash_summary.csv"}
    if requested.exists() and expected.issubset({p.name for p in requested.iterdir()}):
        return requested

    candidates: list[Path] = []
    for audit in ROOT.glob("**/cross_split_exact_duplicate_audit.csv"):
        parent = audit.parent
        if (parent / "split_hash_summary.csv").exists():
            candidates.append(parent)
    candidates = sorted(set(candidates), key=lambda p: ("recovery" not in str(p).lower(), len(str(p))))
    if not candidates:
        raise FileNotFoundError(
            "Could not locate the recovered CICIDS2017 duplicate audit. Run the frozen "
            "audit_split_leakage workflow or pass --recovered-overlap-root explicitly."
        )
    return candidates[0]


def build_fig5_overlap(overlap_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    cross_path = require(overlap_root / "cross_split_exact_duplicate_audit.csv")
    split_path = require(overlap_root / "split_hash_summary.csv")
    cross = pd.read_csv(cross_path).copy()
    splits = pd.read_csv(split_path).copy()
    numeric(cross, ["exact_feature_row_hash_overlap"])
    numeric(splits, ["rows"])
    row_counts = dict(zip(splits["split"].astype(str), splits["rows"].astype(float)))

    rows: list[dict[str, object]] = []
    for _, r in cross.iterrows():
        left = str(r["split_left"])
        right = str(r["split_right"])
        if left not in row_counts or right not in row_counts:
            raise RuntimeError(f"Missing split row count for {left}/{right}")
        overlap = int(r["exact_feature_row_hash_overlap"])
        smaller = min(float(row_counts[left]), float(row_counts[right]))
        rows.append(
            {
                "split_left": left,
                "split_right": right,
                "left_rows": int(row_counts[left]),
                "right_rows": int(row_counts[right]),
                "exact_feature_row_hash_overlap": overlap,
                "fraction_of_smaller_split": overlap / smaller if smaller else np.nan,
            }
        )
    return pd.DataFrame(rows), [cross_path, split_path]


def build_fig6_surface() -> pd.DataFrame:
    df = pd.read_csv(require(SURFACE)).copy()
    numeric(df, [
        "historical_unknown_detection_rate",
        "new_unknown_detection_rate",
        "delta_unknown_detection_rate",
        "abs_delta_unknown_detection_rate",
    ])
    cols = [
        "holdout_family",
        "matched_stage1_weight_mode",
        "historical_unknown_detection_rate",
        "new_unknown_detection_rate",
        "delta_unknown_detection_rate",
        "abs_delta_unknown_detection_rate",
    ]
    return df[cols].sort_values("holdout_family").reset_index(drop=True)


def build_fig7_8_known_heldout() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(require(KNOWN_HELDOUT)).copy()
    numeric(df, ["known_recall", "heldout_udr_mean", "heldout_udr_sd"])
    cols = ["dataset", "family", "known_recall", "heldout_udr_mean", "heldout_udr_sd"]
    base = df[cols].copy()
    cicids = base[base["dataset"].astype(str) == "CICIDS2017"].sort_values("family").reset_index(drop=True)
    ciciot = base[base["dataset"].astype(str) == "CICIoT2023"].sort_values("family").reset_index(drop=True)
    if len(cicids) != 6 or len(ciciot) != 6:
        raise RuntimeError(f"Expected 6 matched families per dataset, found {len(cicids)} and {len(ciciot)}")
    return cicids, ciciot


def build_fig9_failure_matrix() -> pd.DataFrame:
    df = pd.read_csv(require(FAIL_DEST)).copy()
    df = df[df["analysis_lane"].astype(str) == "fixed_rf_primary"].copy()
    numeric(df, ["share_conditional_on_nonunknown"])
    out = (
        df.groupby(["dataset", "holdout_family", "destination"], sort=True, as_index=False)
        .agg(
            residual_destination_share_mean=("share_conditional_on_nonunknown", "mean"),
            residual_destination_share_sd=("share_conditional_on_nonunknown", "std"),
            n_runs=("seed", "count"),
        )
    )
    out["residual_destination_share_sd"] = out["residual_destination_share_sd"].fillna(0.0)
    return out.sort_values(["dataset", "holdout_family", "destination"]).reset_index(drop=True)


def build_fig10_11_rejectors() -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = pd.read_csv(require(REJECTOR)).copy()
    audit = pd.read_csv(require(BUDGET_AUDIT)).copy()
    numeric(primary, [
        "unknown_detection_rate",
        "overall_reject_rate",
        "validation_overall_reject_rate",
        "delta_unknown_detection_rate_vs_max_confidence",
    ])

    audit_cols = [
        "dataset",
        "holdout_family",
        "method",
        "test_budget_exceeded",
        "test_any_nonbudget_constraint_exceeded",
        "test_any_selection_constraint_exceeded",
    ]
    missing = [c for c in audit_cols if c not in audit.columns]
    if missing:
        raise RuntimeError(f"Budget audit missing columns: {missing}")
    merged = primary.merge(
        audit[audit_cols],
        on=["dataset", "holdout_family", "method"],
        how="left",
        validate="one_to_one",
    )
    for col in audit_cols[3:]:
        merged[col] = boolify(merged[col].fillna(False))

    fig10 = merged[merged["method"].astype(str) != "max_confidence"].copy()
    fig10_cols = [
        "dataset",
        "holdout_family",
        "method",
        "delta_unknown_detection_rate_vs_max_confidence",
        "unknown_detection_rate",
        "overall_reject_rate",
        "test_any_selection_constraint_exceeded",
    ]
    fig10 = fig10[fig10_cols].sort_values(["dataset", "holdout_family", "method"]).reset_index(drop=True)

    fig11_cols = [
        "dataset",
        "holdout_family",
        "method",
        "validation_overall_reject_rate",
        "overall_reject_rate",
        "test_budget_exceeded",
        "test_any_nonbudget_constraint_exceeded",
        "test_any_selection_constraint_exceeded",
    ]
    fig11 = merged[fig11_cols].sort_values(["dataset", "holdout_family", "method"]).reset_index(drop=True)
    return fig10, fig11


def main() -> None:
    args = build_parser().parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    overlap_root = locate_recovered_overlap_root(Path(args.recovered_overlap_root))

    fig2 = build_fig2_closed_set()
    fig3 = build_fig3_reference_closed_set()
    fig4 = build_fig4_reference_protocol_b()
    fig5, overlap_sources = build_fig5_overlap(overlap_root)
    fig6 = build_fig6_surface()
    fig7, fig8 = build_fig7_8_known_heldout()
    fig9 = build_fig9_failure_matrix()
    fig10, fig11 = build_fig10_11_rejectors()

    outputs = {
        "fig2_closed_set_composition.csv": fig2,
        "fig3_reference_profiles_closed_set.csv": fig3,
        "fig4_reference_profiles_protocol_b.csv": fig4,
        "fig5_recovered_cross_split_overlap.csv": fig5,
        "fig6_cicids_partition_sensitivity.csv": fig6,
        "fig7_known_vs_heldout_cicids2017.csv": fig7,
        "fig8_known_vs_heldout_ciciot2023.csv": fig8,
        "fig9_failure_destination_matrix.csv": fig9,
        "fig10_rejector_delta_udr.csv": fig10,
        "fig11_validation_test_rejection.csv": fig11,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)

    sources = [
        PROTOCOL_A_CORE,
        PROTOCOL_A_FLAT,
        REFERENCE_DROP,
        REFERENCE_PROTOCOL_B,
        SURFACE,
        KNOWN_HELDOUT,
        FAIL_DEST,
        REJECTOR,
        BUDGET_AUDIT,
        *overlap_sources,
    ]
    sources = list(dict.fromkeys(sources))
    manifest = {
        "status": "JISA_MAIN_FIGURE_SOURCES_FROZEN",
        "scientific_reanalysis": False,
        "figure_1_handled_separately": True,
        "recovered_overlap_root": str(overlap_root.relative_to(ROOT)) if overlap_root.is_relative_to(ROOT) else str(overlap_root),
        "source_hashes_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha256(require(p)) for p in sources},
        "figure_sources": {
            "Figure 2": "fig2_closed_set_composition.csv",
            "Figure 3": "fig3_reference_profiles_closed_set.csv",
            "Figure 4": "fig4_reference_profiles_protocol_b.csv",
            "Figure 5": "fig5_recovered_cross_split_overlap.csv",
            "Figure 6": "fig6_cicids_partition_sensitivity.csv",
            "Figure 7": "fig7_known_vs_heldout_cicids2017.csv",
            "Figure 8": "fig8_known_vs_heldout_ciciot2023.csv",
            "Figure 9": "fig9_failure_destination_matrix.csv",
            "Figure 10": "fig10_rejector_delta_udr.csv",
            "Figure 11": "fig11_validation_test_rejection.csv",
        },
        "interpretation_boundaries": {
            "Figure 3": "Published closed-set reference values and framework-compatible reference-profile runs are descriptive contrasts, not byte-identical third-party reproductions.",
            "Figure 4": "Reference-profile Protocol-B results are descriptive stress tests of literature-derived configurations; they do not imply that the cited papers evaluated held-out-family recognition.",
            "Figure 5": "Exact processed-feature hash overlap does not establish repeated physical flows and does not quantify near-duplicates.",
            "Figure 6": "The partition change alters sample assignment as well as exact-representation sharing; do not attribute the family shifts causally to duplicate removal.",
            "Figures 7-8": "The six-family Spearman relationships are descriptive only.",
            "Figure 9": "Failure destinations are model-behavior diagnostics, not semantic equivalence claims.",
            "Figures 10-11": "Operating points were selected on validation only; test exceedances are retained rather than used for reselection.",
        },
    }
    (OUT / "main_figure_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": manifest["status"],
        "source_tables": list(outputs),
        "source_artifacts_hashed": len(sources),
        "next_stage": "render Figures 2-11 with scripts/29_render_jisa_main_figures.py",
    }, indent=2))
    print(f"\nWrote main figure sources to: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
