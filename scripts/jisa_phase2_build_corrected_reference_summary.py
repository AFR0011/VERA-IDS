#!/usr/bin/env python3
"""Build corrected JISA reference-profile evidence from frozen tracked summaries.

This script performs no model fitting. It separates:
1) metrics actually reported by the cited papers;
2) historical VERA framework-compatible reference values;
3) VERA source-inspired Protocol-A profile results; and
4) VERA primary Protocol-A results.

The local outputs/11_reference_framework_eval/protocol_a summary may contain only
planning rows in some owner workspaces. Therefore the source-inspired profile metrics
are read from outputs/summaries/reference_profile_metric_drop.csv, which is a tracked,
manifested compact evidence surface containing the historical Protocol-A reference-
profile system metrics. No value is inferred from the plan-only summary.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_EVIDENCE = REPO_ROOT / "outputs" / "summaries" / "reference_profile_metric_drop.csv"
PRIMARY_EVIDENCE = REPO_ROOT / "outputs" / "summaries" / "protocol_a_core_summary.csv"
SOURCE_MANIFEST = REPO_ROOT / "outputs" / "summaries" / "SOURCE_MANIFEST.csv"
OUT = REPO_ROOT / "outputs" / "13_jisa_q1_revision" / "phase2_reference_provenance"

SOURCE_ANCHORS: list[dict[str, Any]] = [
    {
        "anchor_id": "adewole2025_xgb__CICIDS2017__multiclass",
        "paper": "adewole2025_xgb",
        "dataset": "CICIDS2017",
        "model_family": "xgb",
        "source_task": "multiclass",
        "source_taxonomy": "paper-defined multiclass CIC-IDS2017",
        "source_accuracy": 0.9988,
        "source_f1": 0.9987,
        "source_location": "Adewole et al. 2025, Table 7",
        "source_url": "https://www.mdpi.com/1424-8220/25/6/1845",
        "main_comparison": True,
    },
    {
        "anchor_id": "adewole2025_xgb__CICIoT2023__binary",
        "paper": "adewole2025_xgb",
        "dataset": "CICIoT2023",
        "model_family": "xgb",
        "source_task": "binary",
        "source_taxonomy": "Benign versus attack",
        "source_accuracy": 0.9854,
        "source_f1": 0.9855,
        "source_location": "Adewole et al. 2025, Table 5",
        "source_url": "https://www.mdpi.com/1424-8220/25/6/1845",
        "main_comparison": True,
    },
    {
        "anchor_id": "neto2023_rf__CICIoT2023__8class",
        "paper": "neto2023_rf",
        "dataset": "CICIoT2023",
        "model_family": "rf",
        "source_task": "8-class multiclass",
        "source_taxonomy": "Benign plus seven attack categories",
        "source_accuracy": 0.994368173,
        "source_f1": 0.71928904,
        "source_location": "Neto et al. 2023, Table 6",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/",
        "main_comparison": True,
    },
    {
        "anchor_id": "neto2023_rf__CICIoT2023__34class",
        "paper": "neto2023_rf",
        "dataset": "CICIoT2023",
        "model_family": "rf",
        "source_task": "34-class multiclass",
        "source_taxonomy": "Benign plus 33 individual attacks",
        "source_accuracy": 0.99164365,
        "source_f1": 0.714021981,
        "source_location": "Neto et al. 2023, Table 6",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/",
        "main_comparison": False,
    },
    {
        "anchor_id": "neto2023_rf__CICIoT2023__binary",
        "paper": "neto2023_rf",
        "dataset": "CICIoT2023",
        "model_family": "rf",
        "source_task": "binary",
        "source_taxonomy": "Benign versus malicious",
        "source_accuracy": 0.99680798,
        "source_f1": 0.965279544,
        "source_location": "Neto et al. 2023, Table 6",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/",
        "main_comparison": False,
    },
]

PROFILE_FOR_PAPER = {
    "adewole2025_xgb": "adewole2025_xgb_profile",
    "neto2023_rf": "neto2023_rf_profile",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}; columns={list(df.columns)}")


def verify_manifest_registration() -> dict[str, str]:
    if not SOURCE_MANIFEST.exists():
        raise RuntimeError(f"Missing tracked source manifest: {SOURCE_MANIFEST}")
    manifest = pd.read_csv(SOURCE_MANIFEST)
    require_columns(manifest, {"public_path", "source_relative_path", "sha256", "bytes"}, "SOURCE_MANIFEST")
    wanted = "outputs/summaries/reference_profile_metric_drop.csv"
    row = manifest[manifest["public_path"].astype(str) == wanted]
    if len(row) != 1:
        raise RuntimeError(f"Expected exactly one SOURCE_MANIFEST row for {wanted}; found {len(row)}")
    r = row.iloc[0]
    return {
        "public_path": str(r["public_path"]),
        "source_relative_path": str(r["source_relative_path"]),
        "recorded_sha256": str(r["sha256"]),
        "recorded_bytes": str(r["bytes"]),
    }


def one_profile_row(df: pd.DataFrame, profile: str, dataset: str) -> pd.Series:
    required = {
        "paper",
        "model_profile",
        "dataset",
        "full_framework_surface",
        "task_or_holdout",
        "closed_set_accuracy",
        "closed_set_macro_f1",
        "full_framework_accuracy",
        "full_framework_macro_f1_supported_labels",
    }
    require_columns(df, required, "reference_profile_metric_drop.csv")
    g = df[
        (df["model_profile"].astype(str) == profile)
        & (df["dataset"].astype(str) == dataset)
        & (df["full_framework_surface"].astype(str) == "protocol_a_reference_profile")
        & (df["task_or_holdout"].astype(str) == "closed_set_system")
    ].copy()
    if len(g) != 1:
        raise RuntimeError(
            f"Expected exactly one frozen Protocol-A source-inspired profile row for {profile}/{dataset}; found {len(g)}"
        )
    row = g.iloc[0]
    for col in [
        "closed_set_accuracy",
        "closed_set_macro_f1",
        "full_framework_accuracy",
        "full_framework_macro_f1_supported_labels",
    ]:
        if pd.isna(pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]):
            raise RuntimeError(f"Non-numeric {col} for {profile}/{dataset}")
    return row


def one_primary_row(df: pd.DataFrame, dataset: str, model_family: str) -> pd.Series:
    required = {
        "dataset",
        "model_family",
        "policy_variant",
        "system_accuracy",
        "system_macro_f1_supported_labels",
    }
    require_columns(df, required, "protocol_a_core_summary.csv")
    g = df[
        (df["dataset"].astype(str) == dataset)
        & (df["model_family"].astype(str) == model_family)
        & (df["policy_variant"].astype(str) == "strict_tau")
    ].copy()
    if len(g) != 1:
        raise RuntimeError(
            f"Expected exactly one strict_tau VERA primary row for {dataset}/{model_family}; found {len(g)}"
        )
    return g.iloc[0]


def main() -> int:
    for path in [PROFILE_EVIDENCE, PRIMARY_EVIDENCE, SOURCE_MANIFEST]:
        if not path.exists():
            raise RuntimeError(f"Missing required frozen evidence file: {path}")

    manifest_row = verify_manifest_registration()
    profile_df = pd.read_csv(PROFILE_EVIDENCE)
    primary_df = pd.read_csv(PRIMARY_EVIDENCE)
    anchors_df = pd.DataFrame(SOURCE_ANCHORS)

    OUT.mkdir(parents=True, exist_ok=True)
    anchors_df.to_csv(OUT / "published_source_anchor_catalog.csv", index=False)

    table_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []

    for anchor in SOURCE_ANCHORS:
        if not bool(anchor["main_comparison"]):
            continue

        paper = str(anchor["paper"])
        dataset = str(anchor["dataset"])
        family = str(anchor["model_family"])
        profile_name = PROFILE_FOR_PAPER[paper]
        profile = one_profile_row(profile_df, profile_name, dataset)
        primary = one_primary_row(primary_df, dataset, family)
        comparison_id = str(anchor["anchor_id"])

        profile_acc = float(profile["full_framework_accuracy"])
        profile_f1 = float(profile["full_framework_macro_f1_supported_labels"])
        primary_acc = float(primary["system_accuracy"])
        primary_f1 = float(primary["system_macro_f1_supported_labels"])

        table_rows.append(
            {
                "comparison_id": comparison_id,
                "paper": paper,
                "dataset": dataset,
                "model_family": family,
                "published_source_task": anchor["source_task"],
                "published_source_taxonomy": anchor["source_taxonomy"],
                "published_source_accuracy": float(anchor["source_accuracy"]),
                "published_source_f1": float(anchor["source_f1"]),
                "published_source_location": anchor["source_location"],
                "source_inspired_profile": profile_name,
                "source_inspired_surface": "VERA Protocol A reference-profile system; frozen supported-label macro-F1",
                "source_inspired_accuracy": profile_acc,
                "source_inspired_macro_f1_supported": profile_f1,
                "vera_primary_model_family": family,
                "vera_primary_surface": "VERA Protocol A strict_tau system; supported-label macro-F1",
                "vera_primary_accuracy": primary_acc,
                "vera_primary_macro_f1_supported": primary_f1,
                "comparability": "published source metric contextual_only_not_matched_replication",
            }
        )

        legacy_rows.append(
            {
                "comparison_id": comparison_id,
                "paper": paper,
                "dataset": dataset,
                "historical_task_label": profile["closed_set_task_used"],
                "historical_framework_compatible_accuracy": float(profile["closed_set_accuracy"]),
                "historical_framework_compatible_macro_f1": float(profile["closed_set_macro_f1"]),
                "provenance_class": "framework_compatible_reference",
                "manuscript_rule": "must_not_be_labelled_as_published_source_metric",
            }
        )

        long_rows.extend(
            [
                {
                    "comparison_id": comparison_id,
                    "paper": paper,
                    "dataset": dataset,
                    "model_family": family,
                    "provenance_class": "published_source_metric",
                    "surface_label": f"Published source ({anchor['source_task']})",
                    "accuracy": float(anchor["source_accuracy"]),
                    "f1_value": float(anchor["source_f1"]),
                    "f1_definition": "paper-reported F1; source averaging/task as reported by paper",
                    "directly_comparable_to_published_source": True,
                },
                {
                    "comparison_id": comparison_id,
                    "paper": paper,
                    "dataset": dataset,
                    "model_family": family,
                    "provenance_class": "source_inspired_profile_result",
                    "surface_label": "VERA source-inspired profile (Protocol A)",
                    "accuracy": profile_acc,
                    "f1_value": profile_f1,
                    "f1_definition": "VERA system supported-label macro-F1",
                    "directly_comparable_to_published_source": False,
                },
                {
                    "comparison_id": comparison_id,
                    "paper": paper,
                    "dataset": dataset,
                    "model_family": family,
                    "provenance_class": "vera_primary_result",
                    "surface_label": f"VERA primary {family.upper()} (Protocol A strict_tau)",
                    "accuracy": primary_acc,
                    "f1_value": primary_f1,
                    "f1_definition": "VERA system supported-label macro-F1",
                    "directly_comparable_to_published_source": False,
                },
            ]
        )

    table = pd.DataFrame(table_rows)
    long_df = pd.DataFrame(long_rows)
    legacy_df = pd.DataFrame(legacy_rows)

    if len(table) != 3 or len(long_df) != 9 or len(legacy_df) != 3:
        raise RuntimeError(
            f"Unexpected corrected evidence shape: table={len(table)}, figure={len(long_df)}, legacy={len(legacy_df)}"
        )

    table.to_csv(OUT / "manuscript_table3_corrected.csv", index=False)
    long_df.to_csv(OUT / "figure3_corrected_long.csv", index=False)
    legacy_df.to_csv(OUT / "historical_framework_compatible_reference_values.csv", index=False)

    manifest = {
        "status": "built_from_frozen_tracked_evidence_no_model_rerun",
        "source_anchor_count": len(SOURCE_ANCHORS),
        "main_comparison_count": int(len(table)),
        "long_evidence_rows": int(len(long_df)),
        "historical_reference_values_preserved_separately": True,
        "inputs": {
            "source_inspired_profile_evidence": {
                "path": str(PROFILE_EVIDENCE),
                "sha256_actual": sha256_file(PROFILE_EVIDENCE),
                "source_manifest_registration": manifest_row,
            },
            "vera_primary_protocol_a_summary": {
                "path": str(PRIMARY_EVIDENCE),
                "sha256_actual": sha256_file(PRIMARY_EVIDENCE),
            },
            "source_manifest": {
                "path": str(SOURCE_MANIFEST),
                "sha256_actual": sha256_file(SOURCE_MANIFEST),
            },
        },
        "comparison_boundary": (
            "Published source metrics are contextual only; source task/taxonomy/split/averaging are not treated as matched to VERA."
        ),
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("source-inspired metric resolution: frozen tracked outputs/summaries/reference_profile_metric_drop.csv")
    print(f"source-inspired metric file: {PROFILE_EVIDENCE}")
    print("Corrected manuscript Table-3 surface:")
    cols = [
        "paper",
        "dataset",
        "published_source_task",
        "published_source_f1",
        "source_inspired_macro_f1_supported",
        "vera_primary_macro_f1_supported",
    ]
    print(table[cols].to_string(index=False))
    print(f"source anchors: {len(SOURCE_ANCHORS)}")
    print(f"main comparison rows: {len(table)}")
    print(f"figure evidence rows: {len(long_df)}")
    print(f"historical framework-compatible rows preserved: {len(legacy_df)}")
    print(f"output: {OUT}")
    print("PASS=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
