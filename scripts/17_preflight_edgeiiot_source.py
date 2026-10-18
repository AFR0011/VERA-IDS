#!/usr/bin/env python3
"""Read-only Edge-IIoTset source manifest and optional curated placeholder audit.

The optional audit counts exact TEXT tokens by Attack_label before any numeric
coercion. Never claim source independence or train/test validity from this tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

RISK_COLUMNS = (
    "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
)
CSV_TYPES = {".csv"}
CAPTURE_TYPES = {".pcap", ".pcapng"}
CURATED_NAMES = {"ML-EdgeIIoT-dataset.csv", "DNN-EdgeIIoT-dataset.csv"}


def inspect_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header:
        raise ValueError(f"Empty header in {path.name}")
    return header


def inspect_source(root: Path) -> tuple[list[dict], list[Path]]:
    if not root.is_dir():
        raise ValueError("Input directory not found")
    records: list[dict] = []
    curated: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError("Source path escapes root")
        suffix = path.suffix.lower()
        rel = path.relative_to(root).as_posix()
        if suffix in CSV_TYPES:
            header = inspect_csv_header(path)
            records.append({
                "relative_path": rel,
                "bytes": path.stat().st_size,
                "kind": "csv",
                "column_count": len(header),
                "headers": header,
                "duplicate_headers": sorted([k for k, n in Counter(header).items() if n > 1]),
                "candidate_identity_time_columns": [
                    c for c in header if re.search(r"time|timestamp|frame\.time|ip\.|port|mac|uid|tcp\.payload|http\.file_data", c, re.I)
                ],
            })
            if path.name in CURATED_NAMES:
                curated.append(path)
        elif suffix in CAPTURE_TYPES:
            records.append({"relative_path": rel, "bytes": path.stat().st_size, "kind": "capture"})
        elif path.name.lower().startswith("readme") or suffix == ".pdf":
            records.append({"relative_path": rel, "bytes": path.stat().st_size, "kind": "documentation"})
    return records, curated


def audit_curated(path: Path) -> dict:
    """Full textual count in four named fields by binary label; no row export."""
    census = {field: {"0": Counter(), "1": Counter(), "other_labels": 0} for field in RISK_COLUMNS}
    targets = Counter()
    types = Counter()
    rows = 0
    invalid_width = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        head = reader.fieldnames or []
        if "Attack_label" not in head or "Attack_type" not in head:
            raise ValueError(f"Curated dataset missing binary/multiclass labels: {path.name}")
        present = [field for field in RISK_COLUMNS if field in head]
        for row in reader:
            rows += 1
            if None in row or any(v is None for v in row.values()):
                invalid_width += 1
            label = str(row.get("Attack_label") or "").strip()
            targets[label] += 1
            types[str(row.get("Attack_type") or "").strip()] += 1
            for field in present:
                if label not in {"0", "1"}:
                    census[field]["other_labels"] += 1
                else:
                    token = row.get(field)
                    if token in {"0", "0.0"}:
                        census[field][label][token] += 1
    return {
        "relative_filename": path.name,
        "rows_scanned": rows,
        "width_errors": invalid_width,
        "binary_label_counts": dict(sorted(targets.items())),
        "attack_type_counts": dict(sorted(types.items())),
        "risk_columns_present": present,
        "risk_columns_missing": [c for c in RISK_COLUMNS if c not in present],
        "exact_text_zero_spellings_by_column_and_label": {
            name: {"normal_label_0": dict(sorted(rec["0"].items())),
                   "attack_label_1": dict(sorted(rec["1"].items())),
                   "other_label_rows": rec["other_labels"]}
            for name, rec in census.items() if name in present
        },
        "warning": "This table counts only exact raw tokens 0 and 0.0; it is not a complete token-purity or label-prediction test.",
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/17_revision_campaign/edgeiiot/source_preflight.json"))
    parser.add_argument("--audit-curated", action="store_true", help="Full curated CSV textual token census; extra source read")
    opts = parser.parse_args(argv)
    try:
        records, curated = inspect_source(opts.input_root)
        if not records:
            raise ValueError("No expected CSV/PCAP/docs found")
        audited = []
        if opts.audit_curated:
            for p in curated:
                print(f"[CURATED] Auditing {p.name}", flush=True)
                audited.append(audit_curated(p))
        result = {
            "schema_version": "edgeiiot-source-preflight-v1",
            "scope": "source_inventory_and_optional_raw_text_artifact_census_only",
            "files_in_manifest": len(records),
            "csv_files": sum(r["kind"] == "csv" for r in records),
            "capture_files": sum(r["kind"] == "capture" for r in records),
            "manifest": records,
            "curated_audits": audited,
            "warnings": [
                "No per-attack file establishes repeated independently captured attack episodes.",
                "Raw column headers and sampled source-file existence cannot establish model validity or source-independent partitions.",
                "Do not coerce 0 and 0.0 before testing the reported serialization artifact.",
                "Curated category counts and token spelling alone are not independent ground-truth validation.",
                "No raw rows, IP addresses, URLs, or payload values are exported to this JSON.",
                "No data preparation, model training, test selection or method decisions occur here.",
            ],
        }
        opts.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = opts.out.with_suffix(opts.out.suffix + ".partial")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(opts.out)
        print(f"[PASS] {len(records)} source artifacts inventoried -> {opts.out}")
        return 0
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
