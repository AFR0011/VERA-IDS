#!/usr/bin/env python3
"""Build corrected JISA reference-profile evidence from frozen local artifacts.

This script performs no model fitting. It keeps published source metrics,
VERA source-inspired profile results, and VERA primary Protocol-A results as
separate provenance classes and writes a manuscript-facing Table-3 surface plus
long-form Figure-3 data under the JISA revision output namespace.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "config" / "reference_framework_eval.yml"
PRIMARY = REPO_ROOT / "outputs" / "summaries" / "protocol_a_core_summary.csv"
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


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def resolve_repo_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def one_profile_row(df: pd.DataFrame, profile: str, dataset: str) -> pd.Series:
    if "model_profile" not in df.columns or "dataset" not in df.columns:
        raise RuntimeError("Reference-profile summary lacks model_profile/dataset columns")
    g = df[(df["model_profile"].astype(str) == profile) & (df["dataset"].astype(str) == dataset)].copy()
    if g.empty:
        raise RuntimeError(f"Missing local source-inspired profile row: {profile} / {dataset}")

    required = ["system_accuracy", "system_macro_f1_supported_labels"]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise RuntimeError(f"Reference-profile summary missing required columns: {missing}")

    # Multiple bookkeeping rows are allowed only if the manuscript-facing values are identical.
    key_cols = ["system_accuracy", "system_macro_f1_supported_labels"]
    if "system_variant" in g.columns:
        variants = sorted(set(g["system_variant"].astype(str)))
        if variants != ["strict_tau"]:
            raise RuntimeError(f"Unexpected source-inspired system variants for {profile}/{dataset}: {variants}")
        key_cols.append("system_variant")
    unique = g[key_cols].drop_duplicates()
    if len(unique) != 1:
        raise RuntimeError(f"Ambiguous local source-inspired profile rows for {profile}/{dataset}: {len(g)} rows")
    return g.iloc[0]


def one_primary_row(df: pd.DataFrame, dataset: str, model_family: str) -> pd.Series:
    required = {
        "dataset", "model_family", "policy_variant", "system_accuracy",
        "system_macro_f1_supported_labels",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Primary Protocol-A summary missing required columns: {missing}")
    g = df[
        (df["dataset"].astype(str) == dataset)
        & (df["model_family"].astype(str) == model_family)
        & (df["policy_variant"].astype(str) == "strict_tau")
    ].copy()
    if len(g) != 1:
        raise RuntimeError(f"Expected exactly one strict_tau VERA primary row for {dataset}/{model_family}; found {len(g)}")
    return g.iloc[0]


def main() -> int:
    cfg = load_yaml(CONFIG)
    ref_cfg = dict(cfg.get("reference_framework_eval", {}) or {})
    historical_root = resolve_repo_path(str(ref_cfg.get("out_root", "outputs/11_reference_framework_eval")))
    profile_path = historical_root / "protocol_a" / "summary" / "protocol_a_reference_profile_summary.csv"
    if not profile_path.exists():
        raise RuntimeError(f"Missing local reference-profile summary: {profile_path}")
    if not PRIMARY.exists():
        raise RuntimeError(f"Missing tracked VERA primary summary: {PRIMARY}")

    profile_df = pd.read_csv(profile_path)
    primary_df = pd.read_csv(PRIMARY)
    anchors = pd.DataFrame(SOURCE_ANCHORS)
    OUT.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(OUT / "published_source_anchor_catalog.csv", index=False)

    table_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

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

        profile_acc = float(profile["system_accuracy"])
        profile_f1 = float(profile["system_macro_f1_supported_labels"])
        primary_acc = float(primary["system_accuracy"])
        primary_f1 = float(primary["system_macro_f1_supported_labels"])

        table_rows.append({
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
            "source_inspired_surface": "VERA Protocol A strict_tau system; supported-label macro-F1",
            "source_inspired_accuracy": profile_acc,
            "source_inspired_macro_f1_supported": profile_f1,
            "vera_primary_model_family": family,
            "vera_primary_surface": "VERA Protocol A strict_tau system; supported-label macro-F1",
            "vera_primary_accuracy": primary_acc,
            "vera_primary_macro_f1_supported": primary_f1,
            "comparability": "contextual_only_not_matched_replication",
        })

        long_rows.extend([
            {
                "comparison_id": comparison_id,
                "paper": paper,
                "dataset": dataset,
                "model_family": family,
                "provenance_class": "published_source_metric",
                "surface_label": f"Published source ({anchor['source_task']})",
                "accuracy": float(anchor["source_accuracy"]),
                "f1_value": float(anchor["source_f1"]),
                "f1_definition": "paper-reported F1; source averaging as reported by paper",
                "directly_comparable_to_other_classes": False,
            },
            {
                "comparison_id": comparison_id,
                "paper": paper,
                "dataset": dataset,
                "model_family": family,
                "provenance_class": "source_inspired_profile_result",
                "surface_label": "VERA source-inspired profile (Protocol A strict_tau)",
                "accuracy": profile_acc,
                "f1_value": profile_f1,
                "f1_definition": "VERA system supported-label macro-F1",
                "directly_comparable_to_other_classes": False,
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
                "directly_comparable_to_other_classes": False,
            },
        ])

    table = pd.DataFrame(table_rows)
    long_df = pd.DataFrame(long_rows)
    if len(table) != 3 or len(long_df) != 9:
        raise RuntimeError(f"Unexpected corrected evidence shape: table={len(table)}, long={len(long_df)}")

    table.to_csv(OUT / "manuscript_table3_corrected.csv", index=False)
    long_df.to_csv(OUT / "figure3_corrected_long.csv", index=False)

    manifest = {
        "status": "built_from_existing_artifacts_no_model_rerun",
        "historical_reference_config_preserved": True,
        "source_anchor_count": len(SOURCE_ANCHORS),
        "main_comparison_count": int(len(table)),
        "long_evidence_rows": int(len(long_df)),
        "inputs": {
            "reference_config": {"path": str(CONFIG), "sha256": sha256_file(CONFIG)},
            "source_inspired_profile_summary": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
            "vera_primary_protocol_a_summary": {"path": str(PRIMARY), "sha256": sha256_file(PRIMARY)},
        },
        "comparison_boundary": "published source metrics are contextual and not treated as matched replications",
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("Corrected manuscript Table-3 surface:")
    cols = [
        "paper", "dataset", "published_source_task", "published_source_f1",
        "source_inspired_macro_f1_supported", "vera_primary_macro_f1_supported",
    ]
    print(table[cols].to_string(index=False))
    print(f"source anchors: {len(SOURCE_ANCHORS)}")
    print(f"main comparison rows: {len(table)}")
    print(f"figure evidence rows: {len(long_df)}")
    print(f"output: {OUT}")
    print("PASS=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
