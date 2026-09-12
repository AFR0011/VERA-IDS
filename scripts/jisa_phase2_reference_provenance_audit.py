#!/usr/bin/env python3
"""Read-only Phase-2 audit of reference-profile provenance and local compact outputs.

This script does not rerun models and does not rewrite historical reference artifacts.
It identifies legacy values that must not be presented as source-paper metrics, records
verified source anchors frozen in the Phase-2 specification, and inventories compact
reference-profile outputs available locally for rebuilding manuscript Table/Figure 3.

Audit artifacts are written only under .release-audit/jisa_phase2_reference_provenance/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "reference_framework_eval.yml"
AUDIT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase2_reference_provenance"

VERIFIED_SOURCE_ANCHORS = {
    "adewole2025_xgb__CICIDS2017__family": {
        "paper": "adewole2025_xgb",
        "dataset": "CICIDS2017",
        "source_task": "multiclass",
        "source_taxonomy": "paper-defined multiclass CIC-IDS2017",
        "source_accuracy": 0.9988,
        "source_f1": 0.9987,
        "source_url": "https://www.mdpi.com/1424-8220/25/6/1845",
        "source_location": "Table 7",
    },
    "neto2023_rf__CICIoT2023__8class": {
        "paper": "neto2023_rf",
        "dataset": "CICIoT2023",
        "source_task": "8-class multiclass",
        "source_taxonomy": "Benign plus seven attack categories",
        "source_accuracy": 0.994368173,
        "source_f1": 0.71928904,
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/",
        "source_location": "Table 6",
    },
    "neto2023_rf__CICIoT2023__34class": {
        "paper": "neto2023_rf",
        "dataset": "CICIoT2023",
        "source_task": "34-class multiclass",
        "source_taxonomy": "Benign plus 33 individual attacks",
        "source_accuracy": 0.99164365,
        "source_f1": 0.714021981,
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/",
        "source_location": "Table 6",
    },
    "neto2023_rf__CICIoT2023__binary": {
        "paper": "neto2023_rf",
        "dataset": "CICIoT2023",
        "source_task": "binary",
        "source_taxonomy": "Benign versus malicious",
        "source_accuracy": 0.99680798,
        "source_f1": 0.965279544,
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/",
        "source_location": "Table 6",
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def resolve_repo_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def classify_legacy_row(row: dict[str, Any]) -> tuple[str, str]:
    paper = str(row.get("paper", ""))
    dataset = str(row.get("dataset", ""))
    task = str(row.get("task", ""))
    f1 = float(row.get("macro_f1")) if row.get("macro_f1") is not None else float("nan")

    if paper == "adewole2025_xgb" and dataset == "CICIDS2017" and task == "family":
        return (
            "not_source_reported",
            f"legacy_f1={f1:.12g}; verified source multiclass XGB F1=0.9987 (Table 7)",
        )
    if paper == "neto2023_rf" and dataset == "CICIoT2023" and task == "family":
        return (
            "not_source_reported",
            f"legacy_f1={f1:.12g}; verified source RF F1=0.71928904 (8-class) or 0.714021981 (34-class), depending taxonomy",
        )
    return (
        "needs_source_verification",
        "No verified source anchor frozen for this exact legacy row/task; do not relabel as source-reported without source audit.",
    )


def compact_candidates(root: Path) -> list[Path]:
    if not root.exists():
        return []
    wanted_names = {
        "reference_profile_metric_drop.csv",
        "reference_framework_comparison.csv",
        "protocol_a_reference_summary.csv",
        "protocol_b_reference_summary.csv",
        "open_set_reference_summary.csv",
        "sink_aware_reference_summary.csv",
        "aggregate_results.csv",
    }
    found: list[Path] = []
    for path in root.rglob("*.csv"):
        if path.name in wanted_names or "reference" in path.name.lower() or "comparison" in path.name.lower():
            found.append(path)
    return sorted(set(p.resolve() for p in found))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_yaml(args.config.resolve())
    ref = dict(cfg.get("reference_framework_eval", {}) or {})
    legacy = list(ref.get("closed_set_reference_values", []) or [])
    out_root = resolve_repo_path(str(ref.get("out_root", "outputs/11_reference_framework_eval"))).resolve()

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    legacy_rows: list[dict[str, Any]] = []
    for row in legacy:
        rec = dict(row)
        status, note = classify_legacy_row(rec)
        rec["provenance_status"] = status
        rec["provenance_note"] = note
        legacy_rows.append(rec)
    legacy_df = pd.DataFrame(legacy_rows)
    legacy_df.to_csv(AUDIT_ROOT / "legacy_reference_values_audit.csv", index=False)

    anchors_df = pd.DataFrame([{"anchor_id": key, **value} for key, value in VERIFIED_SOURCE_ANCHORS.items()])
    anchors_df.to_csv(AUDIT_ROOT / "verified_source_anchors.csv", index=False)

    compact = compact_candidates(out_root)
    compact_df = pd.DataFrame(
        [
            {
                "path": str(path),
                "relative_to_repo": str(path.relative_to(REPO_ROOT)) if REPO_ROOT in path.parents else "",
                "size_bytes": path.stat().st_size,
            }
            for path in compact
        ]
    )
    compact_df.to_csv(AUDIT_ROOT / "compact_output_inventory.csv", index=False)

    flagged = int((legacy_df.get("provenance_status", pd.Series(dtype=str)) == "not_source_reported").sum()) if not legacy_df.empty else 0
    unresolved = int((legacy_df.get("provenance_status", pd.Series(dtype=str)) == "needs_source_verification").sum()) if not legacy_df.empty else 0
    payload = {
        "config": str(args.config.resolve()),
        "historical_output_root": str(out_root),
        "legacy_rows": int(len(legacy_df)),
        "legacy_rows_confirmed_not_source_reported": flagged,
        "legacy_rows_needing_source_verification": unresolved,
        "verified_source_anchor_count": int(len(anchors_df)),
        "compact_output_files_found": int(len(compact_df)),
        "historical_config_must_remain_preserved": True,
        "table3_figure3_rebuild_required": True,
        "PASS": bool(len(legacy_df) > 0 and flagged >= 2 and len(anchors_df) >= 4),
    }
    (AUDIT_ROOT / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"legacy reference rows: {len(legacy_df)}")
    if not legacy_df.empty:
        cols = [c for c in ["paper", "dataset", "task", "accuracy", "macro_f1", "provenance_status"] if c in legacy_df.columns]
        print(legacy_df[cols].to_string(index=False))
    print(f"confirmed mislabeled-as-source rows: {flagged}")
    print(f"rows still requiring source verification: {unresolved}")
    print(f"verified source anchors: {len(anchors_df)}")
    print(f"compact local reference outputs found: {len(compact_df)}")
    for path in compact[:20]:
        print(f"  {path}")
    if len(compact) > 20:
        print(f"  ... {len(compact) - 20} more")
    print(f"audit: {AUDIT_ROOT / 'audit.json'}")
    print(f"PASS={payload['PASS']}")
    return 0 if payload["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
