#!/usr/bin/env python3
"""Bounded read-only probe of Edge-IIoTset ORIGINAL CSV timestamp formats.

This probes the FIRST rows of each original CSV to establish parsable frame.time
formats and obvious source-schema anomalies. It is not a whole-source chronology,
a session detection algorithm, or an approved train/validation/test split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}
TEXT_MONTH = re.compile(
    r"^\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(\d{1,2}),\s*(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})"
    r"(?:\.(\d+))?(?:\s+(.+))?\s*$", re.IGNORECASE)
ISO = re.compile(
    r"^\s*(\d{4})-(\d{2})-(\d{2})[T ](\d{1,2}):(\d{2}):(\d{2})"
    r"(?:\.(\d+))?(?:\s*(Z|[+-]\d\d:?\d\d|[A-Za-z].*))?\s*$", re.IGNORECASE)
NUMBER = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")
REQUIRED = {"frame.time", "Attack_label", "Attack_type"}


def timestamp_day(raw: str) -> tuple[str, str | None]:
    """Extract *calendar date as written*, not absolute time or time zone."""
    value = str(raw or "").strip()
    m = TEXT_MONTH.fullmatch(value)
    if m:
        month, day, year, hour, minute, second, _fraction, suffix = m.groups()
        try:
            dt = datetime(int(year), MONTHS[month[:1].upper() + month[1:].lower()],
                          int(day), int(hour), int(minute), int(second))
        except (ValueError, KeyError):
            return "invalid_month_text", None
        return ("month_text_with_zone" if suffix else "month_text", dt.date().isoformat())
    m = ISO.fullmatch(value)
    if m:
        year, month, day, hour, minute, second, _fraction, suffix = m.groups()
        try:
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
        except ValueError:
            return "invalid_iso_text", None
        return ("iso_with_zone" if suffix else "iso_no_zone", dt.date().isoformat())
    if NUMBER.fullmatch(value):
        try:
            seconds = float(value)
            if 946684800 <= seconds < 2082758400:
                day = datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()
                return "plausible_unix_seconds_utc", day
        except (OverflowError, ValueError):
            pass
        return "unresolved_numeric", None
    return ("missing" if not value else "other_unparsed"), None


def original_csvs(root: Path) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise ValueError("Input root is not a directory")
    found: list[tuple[str, Path]] = []
    for path in root.rglob("*.csv"):
        relative = path.relative_to(root)
        parts = relative.parts
        if "Attack traffic" in parts:
            kind = "attack"
        elif "Normal traffic" in parts:
            kind = "normal"
        else:
            continue  # Do not conflate curated CSVs with original capture sources.
        if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid original source path")
        found.append((kind, path))
    return sorted(found, key=lambda pair: pair[1].as_posix().casefold())


def probe_file(kind: str, path: Path, root: Path, limit: int) -> dict:
    formats: Counter[str] = Counter()
    days: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    types: Counter[str] = Counter()
    row_count = 0
    first_dates = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"Missing frame.time or label headers: {path.name}")
        for row in reader:
            if row_count >= limit:
                break
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed sampled row width: {path.name}")
            row_count += 1
            fmt, date = timestamp_day(row["frame.time"])
            formats[fmt] += 1
            if date:
                days[date] += 1
                if len(first_dates) < 5:
                    first_dates.append(date)
            labels[str(row["Attack_label"] or "").strip()] += 1
            types[str(row["Attack_type"] or "").strip()] += 1
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "source_kind": kind,
        "rows_sampled_from_file_start": row_count,
        "timestamp_format_counts": dict(sorted(formats.items())),
        "sample_calendar_day_counts": dict(sorted(days.items())),
        "sample_first_parsed_dates": first_dates,
        "sample_attack_label_counts": dict(sorted(labels.items())),
        "sample_attack_type_counts": dict(sorted(types.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path(
        "outputs/17_revision_campaign/edgeiiot/original_timestamp_probe.json"))
    parser.add_argument("--sample-per-original-file", type=int, default=256)
    parser.add_argument("--expected-original-files", type=int, default=24)
    opts = parser.parse_args()
    if opts.sample_per_original_file < 1 or opts.expected_original_files < 0:
        parser.error("Invalid sample or expected file count")
    try:
        files = original_csvs(opts.input_root)
        if opts.expected_original_files and len(files) != opts.expected_original_files:
            raise ValueError(f"Expected {opts.expected_original_files} original CSVs, found {len(files)}")
        if not files:
            raise ValueError("No original attack/normal CSVs found")
        reports = []
        for i, (kind, path) in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Sampling timestamp form: {path.name}", flush=True)
            reports.append(probe_file(kind, path, opts.input_root, opts.sample_per_original_file))
        by_kind = Counter(r["source_kind"] for r in reports)
        if opts.expected_original_files == 24 and (by_kind["attack"] != 14 or by_kind["normal"] != 10):
            raise ValueError("Unexpected original attack/normal source count")
        result = {
            "schema_version": "edgeiiot-original-timestamp-format-probe-v1",
            "scope": "first_rows_per_original_file_only_no_full_counts_no_episode_or_split_approval",
            "original_csv_files": len(files),
            "source_kind_counts": dict(sorted(by_kind.items())),
            "sample_per_original_file_limit": opts.sample_per_original_file,
            "all_samples": reports,
            "warnings": [
                "Only the first N rows per original CSV are read, NOT the complete file.",
                "Calendar dates are extracted from displayed timestamps; unknown time zones are NOT normalized.",
                "Dates and gaps in this probe do not establish repeated independent attack runs.",
                "Do not infer whole-corpus label counts or class support from these first-row samples.",
                "No source IPs, payloads, raw timestamp strings, or individual rows are exported.",
                "No training, model selection, dataset admission, protocol or feature decisions.",
            ],
        }
        opts.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = opts.out.with_name(opts.out.name + ".partial")
        tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        tmp.replace(opts.out)
        print(f"[PASS] Sampled timestamp strings in {len(files)} original CSVs -> {opts.out}")
        return 0
    except (OSError, csv.Error, UnicodeError, ValueError) as e:
        print(f"[FAIL] Timestamp format probe: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
