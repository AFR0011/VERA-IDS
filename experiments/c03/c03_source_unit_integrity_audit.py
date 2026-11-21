#!/usr/bin/env python3
"""Read-only C03 source-unit assignment/provenance audit, with no row/model access.

Run in VERA-IDS root: python -B <path-to-script>
The audit uses native preparation-side split-unit manifests only. A source CSV
file is NOT necessarily an independent capture, device, attack execution or day.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")
REQ = {"unit_id", "source_file_name", "source_file_path", "row_start", "row_end", "raw_row_count"}


def fail(issues: list[str], message: str) -> None:
    issues.append(message)


def as_int(obj: dict[str, str], key: str, unit_id: str, issues: list[str]) -> int | None:
    try:
        val = int(obj[key])
    except (ValueError, TypeError, KeyError):
        fail(issues, f"unit {unit_id}: invalid {key}={obj.get(key)!r}")
        return None
    return val


def run(root: Path) -> int:
    src = root / "protocol_b_unit_manifest.csv"
    dest = root / "protocol_b_file_manifest.json"
    print("=== C03 RAW SOURCE-UNIT ASSIGNMENT AUDIT (READ ONLY) ===")
    print(f"Dataset metadata folder: {root}")
    print("Only split-plan metadata; no data observations or experimental scores read.")
    if not src.is_file() or not dest.is_file():
        print(f"BLOCKED: expected both {src.name} and {dest.name} in dataset folder")
        print(f"  unit manifest exists: {src.is_file()}; split assignment exists: {dest.is_file()}")
        return 2
    with src.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = REQ - set(reader.fieldnames or [])
        if missing:
            print(f"BLOCKED: missing required unit-manifest fields: {sorted(missing)}")
            return 2
        units = list(reader)
    try:
        with dest.open("r", encoding="utf-8-sig") as f:
            assignments: dict[str, Any] = json.load(f)
    except (ValueError, OSError) as exc:
        print(f"BLOCKED: split-assignment file invalid: {exc}")
        return 2
    if not isinstance(assignments, dict) or any(not isinstance(assignments.get(k), list) for k in SPLITS):
        print("BLOCKED: expected train/val/test arrays in split-assignment JSON")
        return 2
    issues: list[str] = []
    unit_counts = Counter(u["unit_id"] for u in units)
    duplicate_units = [key for key, n in unit_counts.items() if n != 1]
    if duplicate_units:
        fail(issues, f"Duplicate unit_id values in unit table: {len(duplicate_units)}")
    unit_by_id = {u["unit_id"]: u for u in units}
    split_by_unit: dict[str, str] = {}
    for sp in SPLITS:
        assigned = assignments[sp]
        for uid in assigned:
            if not isinstance(uid, str):
                fail(issues, f"Non-string ID in {sp} assignment")
                continue
            if uid not in unit_by_id:
                fail(issues, f"Unknown assigned unit_id {uid!r} in {sp}")
            if uid in split_by_unit:
                fail(issues, f"Repeated assignment of {uid!r}: {split_by_unit[uid]} and {sp}")
            split_by_unit[uid] = sp
    omitted = sorted(set(unit_by_id) - set(split_by_unit))
    if omitted:
        fail(issues, f"Manifest units with no split assignment: {len(omitted)}")

    per_source: dict[str, list[tuple[int, int, str, str]]] = defaultdict(list)
    source_name_to_paths: dict[str, set[str]] = defaultdict(set)
    row_plan: Counter[str] = Counter()
    source_sets: dict[str, set[str]] = {sp: set() for sp in SPLITS}
    modes: Counter[str] = Counter()
    for u in units:
        uid = u["unit_id"]
        st = as_int(u, "row_start", uid, issues)
        en = as_int(u, "row_end", uid, issues)
        total = as_int(u, "raw_row_count", uid, issues)
        if st is None or en is None or total is None:
            continue
        if st < 0 or en <= st or total != en - st:
            fail(issues, f"Invalid source interval/length for unit {uid!r}: [{st},{en}) vs {total}")
            continue
        srcpath = str(u["source_file_path"]).strip().replace("/", "\\").lower()
        if not srcpath:
            fail(issues, f"Empty source path for unit {uid!r}")
            continue
        name = str(u["source_file_name"]).strip().casefold()
        source_name_to_paths[name].add(srcpath)
        sp = split_by_unit.get(uid)
        if sp not in SPLITS:
            continue
        per_source[srcpath].append((st, en, sp, uid))
        row_plan[sp] += total
        source_sets[sp].add(srcpath)
        modes[str(u.get("partition_mode") or "NOT_RECORDED")] += 1
    print("\n=== SOURCE UNITS ===")
    print(f"Manifest units: {len(units)}; distinct raw-source paths: {len(per_source)}")
    print(f"Unit partition modes: {dict(sorted(modes.items()))}")
    for sp in SPLITS:
        print(f"{sp:5}: assigned_units={len(assignments[sp]):4d} distinct_source_files={len(source_sets[sp]):4d} planned_raw_rows={row_plan[sp]:,}")
    print("Planned raw rows precede preprocessing caps/quotas; do NOT compare directly to Parquet rows.")
    ambiguous_names = {name: paths for name, paths in source_name_to_paths.items() if len(paths) > 1}
    print(f"Duplicate source basenames resolving to different source paths: {len(ambiguous_names)}")

    print("\n=== SOURCE-FILE CROSS-SPLIT MEMBERSHIP ===")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = source_sets[a] & source_sets[b]
        print(f"{a} × {b}: {len(shared)} shared raw-source paths")
        for name in sorted(shared)[:5]:
            print(f"  example: {Path(name).name}")
    all_three = set.intersection(*(source_sets[sp] for sp in SPLITS))
    print(f"Sources used in all three splits: {len(all_three)}")
    print("Shared source files across splits imply a potential source-group dependency; distinct files do not prove independent captures.")

    print("\n=== SOURCE INTERVAL CONSISTENCY ===")
    overlaps = 0
    cross_split_overlaps = 0
    gaps = 0
    nonzero_start = 0
    multiunit_sources = 0
    for srcpath, ivs in per_source.items():
        ivs.sort(key=lambda v: (v[0], v[1]))
        if len(ivs) > 1:
            multiunit_sources += 1
        if ivs[0][0] > 0:
            nonzero_start += 1
        previous = ivs[0]
        for current in ivs[1:]:
            if current[0] < previous[1]:
                overlaps += 1
                if current[2] != previous[2]:
                    cross_split_overlaps += 1
            elif current[0] > previous[1]:
                gaps += 1
            if current[1] > previous[1]:
                previous = current
    print(f"Raw files segmented into >1 planning unit: {multiunit_sources}")
    print(f"Overlapping source-row intervals: {overlaps} (cross-split: {cross_split_overlaps})")
    print(f"Gaps between planned intervals: {gaps}; source paths whose first interval begins >0: {nonzero_start}")
    if overlaps:
        fail(issues, f"Overlapping source-row intervals detected: {overlaps}")
    if gaps or nonzero_start:
        print("WARNING: partial raw-source interval coverage; investigate dropped/unmanifested planning units.")
    if ambiguous_names:
        print("WARNING: ambiguous basenames; source-file identity must use full source path, not basename.")
    if issues:
        print("\nRESULT: INTEGRITY FAIL / INCOMPLETE PLAN")
        for i in issues[:15]:
            print(" -", i)
        if len(issues) > 15:
            print(f" - ... {len(issues)-15} additional issues")
        return 1
    print("\nRESULT: PASS source-unit assignment/interval integrity for recorded planning metadata")
    if any(source_sets[a] & source_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        print("SOURCE-LEVEL CAUTION: raw source file membership crosses split boundaries.")
    print("NOT PROVED: independent attacks/devices/days, unique flows, label-rule validity, numerical proxy absence, near duplicates, physical capture identity.")
    print("No file writes; no model fitting, testing, or score inspection.")
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("processed_V5/B_day_file/CICIoT2023"))
    args = parser.parse_args()
    raise SystemExit(run(args.dataset_dir))
