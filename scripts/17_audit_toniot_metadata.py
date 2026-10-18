#!/usr/bin/env python3
"""Read-only, bounded-memory full CSV metadata scan for TON_IoT Network.

Reports observed source counts, label/type pairs, valid numeric timestamp
ranges and input schemas. No rows are exported, transformed or partitioned.
No inference about leakage freedom or LOAO eligibility is made here.
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


def input_files(folder: Path, pattern: str) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Directory not found: {folder}")
    paths = [p for p in folder.iterdir() if p.is_file() and re.fullmatch(pattern, p.name, re.I)]
    return sorted(paths, key=lambda p: (int(re.search(r"_(\d+)\.csv$", p.name, re.I).group(1)), p.name))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def observed_time(value: str | None) -> float | None:
    try:
        t = float(value) if value else None
    except (TypeError, ValueError):
        return None
    return t if t is not None and math.isfinite(t) else None


def times_add(times: dict[str, dict], label: str, ts: float) -> None:
    record = times.setdefault(label, {"min": ts, "max": ts, "count": 0})
    record["min"] = min(record["min"], ts)
    record["max"] = max(record["max"], ts)
    record["count"] += 1


def scan_processed(path: Path, hash_files: bool) -> dict:
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames
        if not headers or not {"ts", "label", "type"}.issubset(headers):
            raise ValueError(f"Missing essential fields (ts, label, type): {path.name}")
        labels: Counter[str] = Counter()
        types: Counter[str] = Counter()
        pairs: Counter[tuple[str, str]] = Counter()
        type_times: dict[str, dict] = {}
        all_times: dict[str, dict] = {}
        invalid_ts = malformed = 0
        rows = 0
        for row in reader:
            rows += 1
            if None in row or any(v is None for v in row.values()):
                malformed += 1
            label = str(row.get("label") or "").strip()
            typ = str(row.get("type") or "").strip()
            labels[label] += 1
            types[typ] += 1
            pairs[(label, typ)] += 1
            t = observed_time(row.get("ts"))
            if t is None:
                invalid_ts += 1
            else:
                times_add(all_times, "all", t)
                times_add(type_times, typ, t)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path) if hash_files else None,
        "rows": rows,
        "columns": headers,
        "duplicate_headers": sorted(k for k, v in Counter(headers).items() if v > 1),
        "label_counts": dict(sorted(labels.items())),
        "type_counts": dict(sorted(types.items())),
        "label_type_pairs": [
            {"label": l, "type": t, "count": count}
            for (l, t), count in sorted(pairs.items())
        ],
        "invalid_or_missing_timestamps": invalid_ts,
        "numeric_timestamp_range": all_times.get("all"),
        "numeric_timestamps_by_type": dict(sorted(type_times.items())),
        "malformed_width_rows": malformed,
    }


def scan_groundtruth_header(path: Path, hash_files: bool) -> dict:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        names = next(csv.reader(fh), [])
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "columns": names,
        "sha256": sha256(path) if hash_files else None,
        "note": "Header inventory only; no ground-truth row mapping or timestamp matching",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-root", required=True, type=Path)
    ap.add_argument("--groundtruth-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/17_revision_campaign/toniot/full_metadata_audit.json"))
    ap.add_argument("--sha256", action="store_true", help="Hash every file (additional full-file read)")
    a = ap.parse_args()
    try:
        processed = input_files(a.processed_root, r"Network_dataset_\d+\.csv")
        groundtruth = input_files(a.groundtruth_root, r"GroundTruth_Network_\d+\.csv")
        if not processed or not groundtruth:
            raise ValueError("Missing Network_dataset_*.csv or GroundTruth_Network_*.csv files")
        p_records = []
        for i, path in enumerate(processed, 1):
            print(f"[{i}/{len(processed)}] Counting {path.name}", flush=True)
            p_records.append(scan_processed(path, a.sha256))
        g_records = [scan_groundtruth_header(path, a.sha256) for path in groundtruth]
    except (OSError, UnicodeError, csv.Error, ValueError) as e:
        print(f"[FAIL] No report written: {e}", file=sys.stderr)
        return 1
    types = Counter()
    labels = Counter()
    pairs = Counter()
    for rec in p_records:
        types.update(rec["type_counts"])
        labels.update(rec["label_counts"])
        for pair in rec["label_type_pairs"]:
            pairs[(pair["label"], pair["type"])] += pair["count"]
    duplicate_hashes: dict[str, list[str]] = {}
    if a.sha256:
        hashes: dict[str, list[str]] = {}
        for r in g_records:
            hashes.setdefault(r["sha256"], []).append(r["file"])
        duplicate_hashes = {h: files for h, files in hashes.items() if len(files) > 1}
    result = {
        "schema_version": "toniot-full-metadata-audit-v1",
        "scope": "read_only_metadata_not_partition_or_leakage_approval",
        "complete_processed_scan": True,
        "processed_file_count": len(p_records),
        "groundtruth_file_count": len(g_records),
        "total_processed_rows": sum(r["rows"] for r in p_records),
        "aggregate_type_counts": dict(sorted(types.items())),
        "aggregate_binary_label_counts": dict(sorted(labels.items())),
        "aggregate_label_type_pairs": [
            {"label": l, "type": t, "count": n} for (l, t), n in sorted(pairs.items())
        ],
        "processed": p_records,
        "groundtruth_headers": g_records,
        "duplicate_groundtruth_files_by_sha256": duplicate_hashes,
        "hashes_complete": a.sha256,
        "warnings": [
            "Provider's reported counts are not independently validated by a matching label tally until compared.",
            "File numbers and timestamp ranges do NOT establish source independence or valid partitions.",
            "Groundtruth entries are header-only; source-to-event/row labeling relationships are NOT checked.",
            "Missing/invalid ts, width errors, schema differences and label mismatches need review.",
            "No prediction, threshold, model training, taxonomy approval or test-based selection occurs here.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.out.with_name(a.out.name + ".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(a.out)
    print(f"[PASS] Metadata audit of {result['total_processed_rows']:,} observed rows -> {a.out}")
    print("[CAUTION] Counts and timestamp ranges are NOT a leakage audit or split approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
