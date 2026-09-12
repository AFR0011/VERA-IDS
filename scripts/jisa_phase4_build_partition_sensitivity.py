#!/usr/bin/env python3
"""Build the frozen JISA Phase-4 CICIDS2017 partition-sensitivity surface.

No model fitting is performed. The script compares the frozen recovered
contiguous-within-day Protocol-B result surface with the frozen group-safe /
exact-identity-isolated sensitivity candidates. The group-safe summary retains two
RF Stage-1 weighting finalists per holdout, so one winner per holdout is selected
using only the frozen validation hierarchy: Stage-2 validation macro-F1, then
Stage-1 validation AUROC, then lexical run name. Test outcomes never enter this
selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RECOVERED = REPO_ROOT / "outputs" / "summaries" / "protocol_b_best_per_holdout.csv"
GROUP_SAFE = (
    REPO_ROOT
    / "outputs"
    / "12_jisa_finalization"
    / "06_group_safe_protocol_b_rf"
    / "summary"
    / "best_per_holdout.csv"
)
GROUP_SAFE_SPLIT_PROTOCOL = (
    REPO_ROOT
    / "outputs"
    / "12_jisa_finalization"
    / "04_group_safe_surface"
    / "B_group_safe"
    / "CICIDS2017"
    / "SPLIT_PROTOCOL.json"
)
DUPLICATE_AUDIT_SCOPE = (
    REPO_ROOT
    / "outputs"
    / "12_jisa_finalization"
    / "01_duplicate_structure_audit"
    / "audit_scope.json"
)
GROUP_SAFE_SUPPORT = (
    REPO_ROOT
    / "outputs"
    / "12_jisa_finalization"
    / "05_group_safe_support_audit"
    / "CICIDS2017"
    / "eligible_holdouts.csv"
)
OUT = REPO_ROOT / "outputs" / "13_jisa_q1_revision" / "phase4_partition_validity"

EXPECTED_HOLDOUTS = ["Botnet", "BruteForce", "DDoS", "DoS", "Scan/Recon", "Web/App"]
VALIDATION_SELECTION_RULE = (
    "maximum validation Stage-2 macro-F1; tie-break maximum validation Stage-1 AUROC; "
    "final tie-break lexical run name"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_holdout(value: object) -> str:
    text = str(value).strip()
    aliases = {
        "Scan_Recon": "Scan/Recon",
        "Scan-Recon": "Scan/Recon",
        "Web_App": "Web/App",
        "Web-App": "Web/App",
    }
    return aliases.get(text, text)


def _metric_col(df: pd.DataFrame, names: list[str], *, required: bool = False) -> str | None:
    col = next((name for name in names if name in df.columns), None)
    if required and col is None:
        raise RuntimeError(f"Required metric not found. Tried {names}; columns={list(df.columns)}")
    return col


def _validation_metric_columns(df: pd.DataFrame, path: Path) -> tuple[str, str]:
    stage2 = _metric_col(
        df,
        ["stage2_macro_f1_val", "val_stage2_macro_f1", "stage2_val_macro_f1"],
    )
    stage1 = _metric_col(
        df,
        ["stage1_auc_val", "stage1_roc_auc_val", "val_stage1_auc", "stage1_val_auc"],
    )
    if stage2 is None or stage1 is None or "run_name" not in df.columns:
        raise RuntimeError(
            "Group-safe finalists are not uniquely selected and the frozen validation-only "
            "hierarchy cannot be reconstructed from the retained table. Required: a validation "
            "Stage-2 macro-F1 column, a validation Stage-1 AUROC column, and run_name. "
            f"path={path}; columns={list(df.columns)}"
        )
    return stage2, stage1


def _select_group_safe_winners(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one finalist per holdout using validation metrics only."""
    stage2_col, stage1_col = _validation_metric_columns(df, path)
    work = df.copy()
    work["_selection_stage2_val_macro_f1"] = pd.to_numeric(work[stage2_col], errors="raise")
    work["_selection_stage1_val_auc"] = pd.to_numeric(work[stage1_col], errors="raise")
    if work[["_selection_stage2_val_macro_f1", "_selection_stage1_val_auc"]].isna().any().any():
        raise RuntimeError(f"NaN validation selection metric in group-safe finalists: {path}")

    # Exact deterministic implementation of the frozen hierarchy. No test metric appears
    # in the sort keys.
    ordered = work.sort_values(
        [
            "holdout_family",
            "_selection_stage2_val_macro_f1",
            "_selection_stage1_val_auc",
            "run_name",
        ],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).copy()
    ordered["selected_by_phase4_validation_rule"] = False
    winner_idx = ordered.groupby("holdout_family", sort=False).head(1).index
    ordered.loc[winner_idx, "selected_by_phase4_validation_rule"] = True
    selected = ordered.loc[ordered["selected_by_phase4_validation_rule"]].copy()

    counts = selected.groupby("holdout_family", dropna=False).size()
    if not (counts == 1).all() or len(selected) != len(EXPECTED_HOLDOUTS):
        raise RuntimeError(
            f"Validation-only group-safe selection did not yield exactly one row per holdout: {counts.to_dict()}"
        )

    audit_cols = [
        "holdout_family",
        "run_name",
        "model_family",
        "stage1_weight_mode",
        stage2_col,
        stage1_col,
        "_selection_stage2_val_macro_f1",
        "_selection_stage1_val_auc",
        "selected_by_phase4_validation_rule",
    ]
    audit_cols = [c for c in audit_cols if c in ordered.columns]
    audit = ordered[audit_cols].copy()
    audit["selection_rule"] = VALIDATION_SELECTION_RULE
    return selected, audit


def load_selected_surface(
    path: Path,
    *,
    recovered: bool,
    resolve_group_safe_finalists: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not path.exists():
        raise RuntimeError(f"Missing frozen Phase-4 input: {path}")
    df = pd.read_csv(path)
    if "holdout_family" not in df.columns:
        raise RuntimeError(f"Missing holdout_family in {path}; columns={list(df.columns)}")
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == "CICIDS2017"].copy()
    if recovered and "split_variant" in df.columns:
        mask = df["split_variant"].astype(str).str.contains(
            "recovered contiguous-within-day", case=False, na=False
        )
        if mask.any():
            df = df[mask].copy()
    df["holdout_family"] = df["holdout_family"].map(normalize_holdout)

    got_before = sorted(set(df["holdout_family"].astype(str)))
    missing_before = sorted(set(EXPECTED_HOLDOUTS) - set(got_before))
    if missing_before:
        raise RuntimeError(f"Surface {path} is missing expected holdouts: {missing_before}; got={got_before}")

    selection_audit: pd.DataFrame | None = None
    counts = df.groupby("holdout_family", dropna=False).size()
    bad = counts[counts != 1]
    if not bad.empty:
        if not resolve_group_safe_finalists:
            raise RuntimeError(
                f"Ambiguous selected surface {path}: expected one row per holdout; counts={bad.to_dict()}"
            )
        df, selection_audit = _select_group_safe_winners(df, path)

    udr = _metric_col(df, ["unknown_detection_rate", "test_unknown_detection_rate"], required=True)
    macro = _metric_col(df, ["macro_f1", "test_macro_f1_supported", "system_macro_f1_supported_labels"])
    acc = _metric_col(df, ["accuracy", "test_accuracy", "system_accuracy"])

    keep = ["holdout_family", udr]
    for c in [
        macro,
        acc,
        "model_family",
        "stage1_weight_mode",
        "run_name",
        "split_variant",
        "selection_criterion",
        "stage2_macro_f1_val",
        "stage1_auc_val",
        "_selection_stage2_val_macro_f1",
        "_selection_stage1_val_auc",
    ]:
        if c and c in df.columns and c not in keep:
            keep.append(c)
    df = df[keep].copy()
    df = df.rename(columns={udr: "unknown_detection_rate"})
    if macro:
        df = df.rename(columns={macro: "macro_f1"})
    if acc:
        df = df.rename(columns={acc: "accuracy"})

    counts_after = df.groupby("holdout_family", dropna=False).size()
    if not (counts_after == 1).all():
        raise RuntimeError(f"Surface remains ambiguous after selection: {counts_after.to_dict()}")

    df["unknown_detection_rate"] = pd.to_numeric(df["unknown_detection_rate"], errors="raise")
    if not df["unknown_detection_rate"].between(0, 1, inclusive="both").all():
        raise RuntimeError(f"UDR outside [0,1] in {path}")
    return df, selection_audit


def optional_value(row: pd.Series, name: str) -> Any:
    return row[name] if name in row.index else None


def main() -> int:
    provenance_paths = [GROUP_SAFE_SPLIT_PROTOCOL, DUPLICATE_AUDIT_SCOPE, GROUP_SAFE_SUPPORT]
    missing_provenance = [str(p) for p in provenance_paths if not p.exists()]
    if missing_provenance:
        raise RuntimeError(
            "Group-safe result surface exists but required construction/support provenance is missing:\n  "
            + "\n  ".join(missing_provenance)
        )

    recovered, _ = load_selected_surface(RECOVERED, recovered=True)
    group, selection_audit = load_selected_surface(
        GROUP_SAFE,
        recovered=False,
        resolve_group_safe_finalists=True,
    )
    if selection_audit is None:
        # A future uniquely-selected artifact is acceptable, but the current known surface
        # should produce an audit table because it contains two finalists per holdout.
        selection_audit = pd.DataFrame()

    r = recovered.set_index("holdout_family")
    g = group.set_index("holdout_family")
    rows: list[dict[str, Any]] = []
    for holdout in EXPECTED_HOLDOUTS:
        rr = r.loc[holdout]
        gg = g.loc[holdout]
        ur = float(rr["unknown_detection_rate"])
        ug = float(gg["unknown_detection_rate"])
        rec: dict[str, Any] = {
            "holdout_family": holdout,
            "recovered_unknown_detection_rate": ur,
            "group_safe_unknown_detection_rate": ug,
            "delta_group_safe_minus_recovered": ug - ur,
            "absolute_family_shift": abs(ug - ur),
            "recovered_model_family": optional_value(rr, "model_family"),
            "group_safe_model_family": optional_value(gg, "model_family"),
            "recovered_stage1_weight_mode": optional_value(rr, "stage1_weight_mode"),
            "group_safe_stage1_weight_mode": optional_value(gg, "stage1_weight_mode"),
            "group_safe_selected_run_name": optional_value(gg, "run_name"),
            "group_safe_stage2_val_macro_f1_used_for_selection": optional_value(
                gg, "_selection_stage2_val_macro_f1"
            ),
            "group_safe_stage1_val_auc_used_for_selection": optional_value(
                gg, "_selection_stage1_val_auc"
            ),
        }
        if "macro_f1" in rr.index and "macro_f1" in gg.index:
            rec["recovered_macro_f1"] = float(rr["macro_f1"])
            rec["group_safe_macro_f1"] = float(gg["macro_f1"])
            rec["delta_macro_f1_group_safe_minus_recovered"] = float(
                gg["macro_f1"] - rr["macro_f1"]
            )
        if "accuracy" in rr.index and "accuracy" in gg.index:
            rec["recovered_accuracy"] = float(rr["accuracy"])
            rec["group_safe_accuracy"] = float(gg["accuracy"])
            rec["delta_accuracy_group_safe_minus_recovered"] = float(
                gg["accuracy"] - rr["accuracy"]
            )
        rows.append(rec)

    paired = pd.DataFrame(rows)
    mean_r = float(paired["recovered_unknown_detection_rate"].mean())
    mean_g = float(paired["group_safe_unknown_detection_rate"].mean())
    mean_abs = float(paired["absolute_family_shift"].mean())
    max_abs = float(paired["absolute_family_shift"].max())
    max_row = paired.loc[paired["absolute_family_shift"].idxmax()]
    spearman = float(
        paired["recovered_unknown_detection_rate"].corr(
            paired["group_safe_unknown_detection_rate"], method="spearman"
        )
    )

    summary = {
        "n_common_holdouts": int(len(paired)),
        "recovered_mean_unknown_detection_rate": mean_r,
        "group_safe_mean_unknown_detection_rate": mean_g,
        "mean_signed_shift_group_safe_minus_recovered": float(mean_g - mean_r),
        "mean_absolute_family_shift": mean_abs,
        "max_absolute_family_shift": max_abs,
        "max_shift_holdout_family": str(max_row["holdout_family"]),
        "spearman_family_rank_correlation": spearman,
        "group_safe_winner_selection_rule": VALIDATION_SELECTION_RULE,
        "group_safe_test_metrics_used_for_selection": False,
        "interpretation_boundary": (
            "Partition-sensitivity evidence only. Differences are not attributed causally to duplicate removal; "
            "the two constructions encode competing dependence assumptions."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    paired.to_csv(OUT / "cicids_partition_sensitivity_by_holdout.csv", index=False)
    selection_audit.to_csv(OUT / "group_safe_validation_selection_audit.csv", index=False)
    (OUT / "cicids_partition_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "status": "built_from_frozen_results_no_model_rerun",
        "dataset": "CICIDS2017",
        "recovered_partition": "recovered contiguous-within-day Protocol B variant",
        "group_safe_partition": "group-safe / exact-identity-isolated sensitivity variant",
        "group_safe_candidate_selection": {
            "rule": VALIDATION_SELECTION_RULE,
            "test_metrics_used": False,
            "input_rows_expected": 12,
            "selected_rows_expected": 6,
        },
        "inputs": {
            "recovered_selected_results": {"path": str(RECOVERED), "sha256": sha256_file(RECOVERED)},
            "group_safe_finalists": {"path": str(GROUP_SAFE), "sha256": sha256_file(GROUP_SAFE)},
            "group_safe_split_protocol": {
                "path": str(GROUP_SAFE_SPLIT_PROTOCOL),
                "sha256": sha256_file(GROUP_SAFE_SPLIT_PROTOCOL),
            },
            "duplicate_audit_scope": {
                "path": str(DUPLICATE_AUDIT_SCOPE),
                "sha256": sha256_file(DUPLICATE_AUDIT_SCOPE),
            },
            "group_safe_support": {
                "path": str(GROUP_SAFE_SUPPORT),
                "sha256": sha256_file(GROUP_SAFE_SUPPORT),
            },
        },
        "common_holdouts": EXPECTED_HOLDOUTS,
        "test_labels_used_for_new_selection": False,
        "new_model_training": False,
        "causal_duplicate_removal_claim_authorized": False,
    }
    (OUT / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("Group-safe validation-only winners:")
    winner_cols = [
        "holdout_family",
        "stage1_weight_mode",
        "run_name",
        "_selection_stage2_val_macro_f1",
        "_selection_stage1_val_auc",
    ]
    winner_cols = [c for c in winner_cols if c in group.columns]
    print(group[winner_cols].to_string(index=False))
    print("")
    print("CICIDS2017 partition sensitivity by holdout:")
    print(
        paired[
            [
                "holdout_family",
                "recovered_unknown_detection_rate",
                "group_safe_unknown_detection_rate",
                "delta_group_safe_minus_recovered",
                "absolute_family_shift",
            ]
        ].to_string(index=False)
    )
    print("")
    print(f"common holdouts: {len(paired)}/6")
    print(f"recovered mean UDR: {mean_r:.6f}")
    print(f"group-safe mean UDR: {mean_g:.6f}")
    print(f"mean absolute family shift: {mean_abs:.6f}")
    print(f"max absolute family shift: {max_abs:.6f} ({max_row['holdout_family']})")
    print(f"Spearman family-rank correlation: {spearman:.6f}")
    print(f"output: {OUT}")
    print("PASS=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
