#!/usr/bin/env python3
"""Read-only Edge-IIoTset original CSV field-alignment diagnostic.

Reads at most N initial records of each original Attack/Normal CSV using
csv.reader (positional), preserving field widths and classifying field
SHAPES. Exports no raw timestamps, IPs, payloads, or individual flow records.
This does NOT repair time fields, assign sessions, or approve experiments.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import ipaddress
import json
from pathlib import Path
import re
import sys

KNOWN_TYPES = {
    "Backdoor", "DDoS_HTTP", "DDoS_ICMP", "DDoS_TCP", "DDoS_UDP",
    "Fingerprinting", "MITM", "Password", "Port_Scanning", "Ransomware",
    "SQL_injection", "Uploading", "Vulnerability_scanner", "XSS", "Normal",
}
MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
FULL_MONTH_DATE = re.compile(
    rf"^{MONTH}\s+\d{{1,2}},?\s+\d{{4}}\s+\d{{1,2}}:\d{{2}}:\d{{2}}", re.I
)
MONTH_DAY_ONLY = re.compile(rf"^{MONTH}\s+\d{{1,2}},?$", re.I)
YEAR_TIME_FRAGMENT = re.compile(r"^\d{4}\s+\d{1,2}:\d{2}:\d{2}(?:\.\d+)?(?:\s+\S+)?$")
ISO_DATE = re.compile(r"^\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d")
CLOCK_ONLY = re.compile(r"^\d{1,2}:\d\d:\d\d(?:\.\d+)?$")
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")


def shape(value: str) -> str:
    text = value.strip()
    if not text:
        return "empty"
    if FULL_MONTH_DATE.match(text) or ISO_DATE.match(text):
        return "date_time_text"
    if YEAR_TIME_FRAGMENT.fullmatch(text):
        return "year_time_fragment"
    if MONTH_DAY_ONLY.fullmatch(text):
        return "month_day_fragment"
    try:
        ipaddress.ip_address(text)
        return "ip_address"
    except ValueError:
        pass
    if CLOCK_ONLY.fullmatch(text):
        return "time_only"
    if NUMBER.fullmatch(text):
        return "number"
    return "other_text"


def original_files(root: Path) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise ValueError(f"Input root not found: {root}")
    found: list[tuple[str, Path]] = []
    for path in root.rglob("*.csv"):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if "Attack traffic" in parts:
            kind = "attack"
        elif "Normal traffic" in parts:
            kind = "normal"
        else:
            continue
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source path escapes input root")
        found.append((kind, path))
    return sorted(found, key=lambda x: x[1].relative_to(root).as_posix().casefold())


def inspect_file(kind: str, path: Path, root: Path, n: int) -> dict:
    width_counts: Counter[str] = Counter()
    position: dict[str, Counter[str]] = {}
    patterns: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    types: Counter[str] = Counter()
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        csv.field_size_limit(16 * 1024 * 1024)
        reader = csv.reader(stream)
        headers = next(reader, None)
        if not headers or len(set(headers)) != len(headers):
            raise ValueError(f"Empty/duplicate CSV headers in {path.name}")
        required = {"frame.time", "Attack_label", "Attack_type"}
        if not required.issubset(headers):
            raise ValueError(f"Essential column(s) missing in {path.name}")
        first_index = headers.index("frame.time")
        label_index = headers.index("Attack_label")
        type_index = headers.index("Attack_type")
        watched = list(range(min(8, len(headers))))
        if first_index not in watched:
            watched.append(first_index)
        watched = sorted(set(watched))
        position = {f"{i}:{headers[i]}": Counter() for i in watched}
        for row in reader:
            if count >= n:
                break
            count += 1
            width_counts[str(len(row))] += 1
            for i in watched:
                position[f"{i}:{headers[i]}"][shape(row[i] if i < len(row) else "")] += 1
            patterns[" | ".join(shape(row[i] if i < len(row) else "") for i in range(min(4, len(headers))))] += 1
            label = row[label_index].strip() if label_index < len(row) else ""
            attack_type = row[type_index].strip() if type_index < len(row) else ""
            labels[label if label in {"0", "1"} else ("empty" if not label else "other")] += 1
            types[attack_type if attack_type in KNOWN_TYPES else ("empty" if not attack_type else "other")] += 1
    return {
        "source_file": path.relative_to(root).as_posix(),
        "source_kind": kind,
        "expected_header_count": len(headers),
        "frame_time_header_index": first_index,
        "rows_sampled_from_start": count,
        "row_field_count_distribution": dict(sorted(width_counts.items())),
        "rows_with_field_count_mismatch": sum(v for k, v in width_counts.items() if int(k) != len(headers)),
        "first_column_shape_counts": {k: dict(sorted(v.items())) for k, v in position.items()},
        "first_four_position_shape_patterns": dict(sorted(patterns.items())),
        "binary_label_field_counts": dict(sorted(labels.items())),
        "attack_type_field_counts": dict(sorted(types.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path(
        "outputs/17_revision_campaign/edgeiiot/original_field_alignment_probe.json"))
    parser.add_argument("--sample-per-file", type=int, default=256)
    parser.add_argument("--expected-original-files", type=int, default=24)
    args = parser.parse_args()
    if args.sample_per_file < 1 or args.expected_original_files < 0:
        parser.error("Invalid --sample-per-file or --expected-original-files")
    try:
        files = original_files(args.input_root)
        if not files or (args.expected_original_files and len(files) != args.expected_original_files):
            raise ValueError(f"Found {len(files)} original CSVs, expected {args.expected_original_files}")
        groups = Counter(kind for kind, _ in files)
        if args.expected_original_files == 24 and (groups["attack"] != 14 or groups["normal"] != 10):
            raise ValueError("Expected 14 attack and 10 normal original CSVs")
        results = []
        for i, (kind, path) in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Checking CSV field alignment: {path.name}", flush=True)
            results.append(inspect_file(kind, path, args.input_root, args.sample_per_file))
        data = {
            "schema_version": "edgeiiot-original-csv-field-alignment-probe-v1",
            "scope": "sampled_positional_shape_and_width_only_not_timestamp_recovery_or_split_approval",
            "original_file_count": len(results),
            "source_kind_counts": dict(sorted(groups.items())),
            "sample_limit_per_file": args.sample_per_file,
            "sample_rows_total": sum(r["rows_sampled_from_start"] for r in results),
            "sample_rows_with_field_count_mismatch": sum(r["rows_with_field_count_mismatch"] for r in results),
            "source_records": results,
            "warnings": [
                "Only initial sampled rows are inspected; whole-source integrity is NOT established.",
                "Shape categories do not reconstruct missing timestamp segments or detect every value swap.",
                "Aligned CSV widths and apparently valid attack labels cannot exclude column-value corruption.",
                "Original raw record values, timestamps, IPs, payload and row indices are not exported.",
                "This diagnostic does not define independent attacks, source splits, or leakage-free features.",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.out.with_name(args.out.name + ".partial")
        temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp.replace(args.out)
        print(f"[PASS] Sampled {data['sample_rows_total']:,} positional rows -> {args.out}")
        return 0
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        print(f"[FAIL] Positional audit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
