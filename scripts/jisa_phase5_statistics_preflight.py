#!/usr/bin/env python3
"""Read-only preflight for JISA Phase-5 statistical refresh.

The script inventories provenance-bearing prepared-data schemas, metadata/mapping
artifacts, and row-level score/prediction files that could support cluster-aware
resampling. It performs no statistical test and no model fitting. Outputs are written
only under `.release-audit/jisa_phase5_statistics_preflight/`.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase5_statistics_preflight"

# Roots most likely to retain joinable row-level evidence or construction metadata.
SEARCH_ROOTS = [
    REPO_ROOT / "outputs" / "02_prepared_data",
    REPO_ROOT / "outputs" / "04_protocol_b_support_audit",
    REPO_ROOT / "outputs" / "05_protocol_b_loao",
    REPO_ROOT / "outputs" / "06_open_set_rejection",
    REPO_ROOT / "outputs" / "08_statistics",
    REPO_ROOT / "outputs" / "10_seed_reliability",
    REPO_ROOT / "outputs" / "12_jisa_finalization",
    REPO_ROOT / "outputs" / "13_jisa_q1_revision",
    REPO_ROOT / "outputs" / "summaries",
]

PROVENANCE_TERMS = [
    "source_file", "sourcefile", "raw_file", "rawfile", "filename", "file_name",
    "source_day", "capture_day", "day", "file_day", "source_unit", "unit_id",
    "capture_id", "session_id", "pcap", "flow_id", "row_id", "sample_id",
    "group_id", "identity_group", "exact_identity", "duplicate_group", "source_group",
]

JOIN_KEY_TERMS = [
    "row_id", "sample_id", "record_id", "flow_id", "source_row", "original_index",
    "index", "id",
]

ROW_LEVEL_NAME_TERMS = [
    "score", "scores", "prediction", "predictions", "prob", "probability",
    "method_grid", "selected_methods", "confidence_intervals_case",
]

MAPPING_NAME_TERMS = [
    "manifest", "mapping", "provenance", "source", "split", "group", "duplicate",
    "identity", "overlap", "partition", "support", "audit",
]

SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".release-audit"}
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".yml", ".yaml", ".txt", ".md"}
MAX_METADATA_BYTES = 50 * 1024 * 1024


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def norm_cols(cols: Iterable[object]) -> list[str]:
    return [str(c).strip() for c in cols]


def term_hits(columns: Iterable[str], terms: list[str]) -> list[str]:
    hits: list[str] = []
    low = {c.lower(): c for c in columns}
    for term in terms:
        for key, original in low.items():
            if term == key or term in key:
                hits.append(original)
    return sorted(set(hits))


def inspect_csv_header(path: Path) -> dict[str, Any] | None:
    try:
        if path.suffix.lower() == ".gz" or path.name.lower().endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
        else:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
    except Exception:
        return None
    cols = norm_cols(header)
    return {
        "path": rel(path),
        "format": "csv",
        "columns": cols,
        "provenance_columns": term_hits(cols, PROVENANCE_TERMS),
        "join_key_columns": term_hits(cols, JOIN_KEY_TERMS),
        "size_bytes": path.stat().st_size,
    }


def inspect_parquet_schema(path: Path) -> dict[str, Any] | None:
    try:
        import pyarrow.parquet as pq
        schema = pq.ParquetFile(path).schema_arrow
        cols = norm_cols(schema.names)
        n_rows = int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as exc:
        return {
            "path": rel(path),
            "format": "parquet",
            "error": f"{type(exc).__name__}: {exc}",
            "columns": [],
            "provenance_columns": [],
            "join_key_columns": [],
            "size_bytes": path.stat().st_size,
        }
    return {
        "path": rel(path),
        "format": "parquet",
        "columns": cols,
        "provenance_columns": term_hits(cols, PROVENANCE_TERMS),
        "join_key_columns": term_hits(cols, JOIN_KEY_TERMS),
        "n_rows": n_rows,
        "size_bytes": path.stat().st_size,
    }


def plausible_row_level(path: Path) -> bool:
    name = path.name.lower()
    return any(term in name for term in ROW_LEVEL_NAME_TERMS)


def plausible_mapping(path: Path) -> bool:
    text = str(path).lower()
    return any(term in text for term in MAPPING_NAME_TERMS)


def walk_files() -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(path)
    return out


def inspect_tabular(files: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prepared: list[dict[str, Any]] = []
    row_level: list[dict[str, Any]] = []
    for path in files:
        low = path.name.lower()
        rec: dict[str, Any] | None = None
        if low.endswith(".parquet"):
            rec = inspect_parquet_schema(path)
        elif low.endswith(".csv") or low.endswith(".csv.gz"):
            if "02_prepared_data" in str(path) or plausible_row_level(path) or plausible_mapping(path):
                rec = inspect_csv_header(path)
        if rec is None:
            continue
        if "02_prepared_data" in str(path) or low.endswith(".parquet"):
            prepared.append(rec)
        if plausible_row_level(path):
            row_level.append(rec)
    return prepared, row_level


def inspect_metadata(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or not plausible_mapping(path):
            continue
        try:
            size = path.stat().st_size
            if size > MAX_METADATA_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")[:2_000_000]
        except Exception:
            continue
        low = text.lower()
        hits = sorted({term for term in PROVENANCE_TERMS if term in low})
        if not hits:
            continue
        rows.append({
            "path": rel(path),
            "size_bytes": size,
            "provenance_term_hits": "|".join(hits),
            "contains_cicids2017": "cicids2017" in low,
            "contains_ciciot2023": "ciciot2023" in low,
        })
    rows.sort(key=lambda r: (-len(str(r["provenance_term_hits"]).split("|")), str(r["path"])))
    return rows


def dataset_guess(path_text: str) -> str:
    low = path_text.lower()
    if "ciciot2023" in low or "ciciot" in low:
        return "CICIoT2023"
    if "cicids2017" in low or "cicids" in low:
        return "CICIDS2017"
    if "unsw" in low:
        return "UNSW-NB15"
    if "nsl" in low:
        return "NSL-KDD"
    return "unknown"


def surface_guess(path_text: str) -> str:
    low = path_text.lower()
    if "phase3_validation_blind" in low:
        return "phase3_validation_blind"
    if "14_rejector_tradeoff_runs" in low or "open_set_rejection" in low:
        return "validation_visible_rejectors"
    if "protocol_b" in low or "protocolb" in low:
        return "protocol_b"
    if "phase4_partition" in low or "group_safe" in low:
        return "partition_validity"
    return "other"


def summarize_row_level(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        p = str(r["path"])
        out.append({
            "path": p,
            "dataset_guess": dataset_guess(p),
            "surface_guess": surface_guess(p),
            "n_columns": len(r.get("columns", [])),
            "provenance_columns": "|".join(r.get("provenance_columns", [])),
            "join_key_columns": "|".join(r.get("join_key_columns", [])),
            "size_bytes": r.get("size_bytes"),
        })
    return out


def main() -> int:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    files = walk_files()
    prepared, row_level = inspect_tabular(files)
    metadata = inspect_metadata(files)
    row_summary = summarize_row_level(row_level)

    pd.DataFrame(prepared).to_json(AUDIT_ROOT / "prepared_schema_inventory.json", orient="records", indent=2)
    pd.DataFrame(row_summary).to_csv(AUDIT_ROOT / "row_level_score_inventory.csv", index=False)
    pd.DataFrame(metadata).to_csv(AUDIT_ROOT / "provenance_metadata_candidates.csv", index=False)

    prepared_with_prov = [r for r in prepared if r.get("provenance_columns")]
    row_with_prov = [r for r in row_level if r.get("provenance_columns")]
    row_with_join = [r for r in row_level if r.get("join_key_columns")]

    datasets_with_prepared_prov = sorted({dataset_guess(str(r["path"])) for r in prepared_with_prov})
    datasets_with_row_prov = sorted({dataset_guess(str(r["path"])) for r in row_with_prov})

    phase3_rows = [r for r in row_summary if r["surface_guess"] == "phase3_validation_blind"]
    phase3_direct_prov = [r for r in phase3_rows if r["provenance_columns"]]
    phase3_joinable = [r for r in phase3_rows if r["join_key_columns"]]

    payload = {
        "files_examined": len(files),
        "prepared_or_parquet_surfaces_inspected": len(prepared),
        "prepared_surfaces_with_provenance_columns": len(prepared_with_prov),
        "row_level_score_surfaces_inspected": len(row_level),
        "row_level_surfaces_with_direct_provenance_columns": len(row_with_prov),
        "row_level_surfaces_with_candidate_join_keys": len(row_with_join),
        "provenance_metadata_candidates": len(metadata),
        "datasets_with_prepared_provenance": datasets_with_prepared_prov,
        "datasets_with_row_level_direct_provenance": datasets_with_row_prov,
        "phase3_row_level_surfaces": len(phase3_rows),
        "phase3_surfaces_with_direct_provenance": len(phase3_direct_prov),
        "phase3_surfaces_with_candidate_join_keys": len(phase3_joinable),
        "new_statistical_tests_performed": False,
        "READY_FOR_PHASE5_DESIGN": bool(prepared_with_prov or metadata),
    }
    (AUDIT_ROOT / "preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"files examined: {len(files)}")
    print(f"prepared/parquet surfaces inspected: {len(prepared)}")
    print(f"prepared surfaces with provenance columns: {len(prepared_with_prov)}")
    for r in prepared_with_prov[:20]:
        print(f"  prepared: {r['path']} provenance={r.get('provenance_columns', [])} join={r.get('join_key_columns', [])}")
    if len(prepared_with_prov) > 20:
        print(f"  ... {len(prepared_with_prov)-20} more")

    print(f"row-level score/prediction surfaces inspected: {len(row_level)}")
    print(f"row-level surfaces with direct provenance columns: {len(row_with_prov)}")
    print(f"row-level surfaces with candidate join keys: {len(row_with_join)}")
    for r in row_summary[:40]:
        if r["provenance_columns"] or r["join_key_columns"] or r["surface_guess"] == "phase3_validation_blind":
            print(
                f"  score: {r['path']} dataset={r['dataset_guess']} surface={r['surface_guess']} "
                f"provenance={r['provenance_columns'] or '-'} join={r['join_key_columns'] or '-'}"
            )

    print(f"provenance metadata candidates: {len(metadata)}")
    for r in metadata[:30]:
        print(f"  metadata: {r['path']} hits={r['provenance_term_hits']}")
    if len(metadata) > 30:
        print(f"  ... {len(metadata)-30} more")

    print(f"datasets with prepared provenance: {datasets_with_prepared_prov}")
    print(f"datasets with row-level direct provenance: {datasets_with_row_prov}")
    print(f"phase3 row-level surfaces: {len(phase3_rows)}")
    print(f"phase3 direct-provenance surfaces: {len(phase3_direct_prov)}")
    print(f"phase3 candidate-join-key surfaces: {len(phase3_joinable)}")
    print(f"audit: {AUDIT_ROOT / 'preflight.json'}")
    print(f"READY_FOR_PHASE5_DESIGN={payload['READY_FOR_PHASE5_DESIGN']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
