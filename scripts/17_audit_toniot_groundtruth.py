#!/usr/bin/env python3
"""Read-only full census of TON_IoT Network ground-truth CSV records.

Audits content and time/family coverage without saving raw flows or IP addresses.
Does not match ground-truth rows to processed data or approve evaluation splits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

REQUIRED = {"ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "type"}


def numbered_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    files = []
    for file in root.iterdir():
        match = re.fullmatch(r"GroundTruth_Network_(\d+)\.csv", file.name, re.IGNORECASE)
        if file.is_file() and match:
            files.append((int(match.group(1)), file.name, file))
    return [item[2] for item in sorted(files)]


def checksum(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_timestamp(value: str | None) -> float | None:
    try:
        numeric = float(value) if value and value.strip() else None
    except (ValueError, TypeError):
        return None
    return numeric if numeric is not None and math.isfinite(numeric) else None


def add_time(span: dict, value: float) -> None:
    if not span:
        span.update(min=value, max=value, count=0)
    span["min"] = min(span["min"], value)
    span["max"] = max(span["max"], value)
    span["count"] += 1


def audit_file(path: Path, include_hash: bool) -> dict:
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        if not names or not REQUIRED.issubset(names):
            raise ValueError(f"Missing required ground-truth header(s): {path.name}")
        types: Counter[str] = Counter()
        spans: dict[str, dict] = {}
        overall_span: dict = {}
        empty_by_field: Counter[str] = Counter()
        rows = bad_width = invalid_time = time_reversal = 0
        previous_ts: float | None = None
        for row in reader:
            rows += 1
            if None in row or any(v is None for v in row.values()):
                bad_width += 1
            for name in REQUIRED:
                if not str(row.get(name) or "").strip():
                    empty_by_field[name] += 1
            category = str(row.get("type") or "").strip()
            types[category] += 1
            stamp = parse_timestamp(row.get("ts"))
            if stamp is None:
                invalid_time += 1
            else:
                add_time(overall_span, stamp)
                add_time(spans.setdefault(category, {}), stamp)
                if previous_ts is not None and stamp < previous_ts:
                    time_reversal += 1
                previous_ts = stamp
    return {
        "source_file": path.name,
        "bytes": path.stat().st_size,
        "sha256": checksum(path) if include_hash else None,
        "rows": rows,
        "columns": names,
        "duplicate_headers": sorted(k for k, v in Counter(names).items() if v > 1),
        "type_counts": dict(sorted(types.items())),
        "numeric_timestamp_range": overall_span if overall_span else None,
        "numeric_timestamps_by_type": dict(sorted(spans.items())),
        "invalid_or_missing_timestamps": invalid_time,
        "backwards_timestamp_transitions": time_reversal,
        "malformed_width_rows": bad_width,
        "missing_required_field_counts": dict(sorted(empty_by_field.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundtruth-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/17_revision_campaign/toniot/groundtruth_full_census.json"))
    parser.add_argument("--sha256", action="store_true", help="Also hash complete source files; increases disk reads")
    options = parser.parse_args()
    try:
        files = numbered_files(options.groundtruth_root)
        if not files:
            raise ValueError("No GroundTruth_Network_<number>.csv inputs found")
        records = []
        for index, path in enumerate(files, 1):
            print(f"[{index}/{len(files)}] Ground-truth census {path.name}", flush=True)
            records.append(audit_file(path, options.sha256))
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        print(f"[FAIL] No census saved: {exc}", file=sys.stderr)
        return 1

    types: Counter[str] = Counter()
    for record in records:
        types.update(record["type_counts"])
    repeated_hashes: dict[str, list[str]] = {}
    if options.sha256:
        hash_to_files: dict[str, list[str]] = {}
        for record in records:
            hash_to_files.setdefault(record["sha256"], []).append(record["source_file"])
        repeated_hashes = {
            sha: names for sha, names in hash_to_files.items() if len(names) > 1
        }

    result = {
        "schema_version": "toniot-groundtruth-full-census-v1",
        "scope": "read_only_full_groundtruth_counts_not_flow_join_or_split_approval",
        "groundtruth_files": len(records),
        "total_groundtruth_rows_as_stored": sum(r["rows"] for r in records),
        "aggregate_type_counts_as_stored": dict(sorted(types.items())),
        "groundtruth": records,
        "hashes_complete": options.sha256,
        "duplicate_whole_files_by_sha256": repeated_hashes,
        "warnings": [
            "Ground-truth records have not been matched to processed Network flows.",
            "Duplicate whole-file contents, if any, must not count as independent records.",
            "Timestamp ranges and file numbers alone do not prove session independence.",
            "Raw endpoint addresses and individual rows are not saved to this report.",
            "No taxonomy, feature set, partition or model-training decision is made.",
        ],
    }
    options.out.parent.mkdir(parents=True, exist_ok=True)
    partial = options.out.with_name(options.out.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(options.out)
    print(f"[PASS] Examined {result['total_groundtruth_rows_as_stored']:,} ground-truth rows as stored -> {options.out}")
    print("[CAUTION] This is not an event-to-flow mapping or leakage audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
