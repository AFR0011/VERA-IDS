#!/usr/bin/env python3
"""C03 read-only legacy Protocol-B *raw-file assignment* integrity audit.

Run from repository root:
    python -B c03_source_file_assignment_audit_v3.py

Works with legacy protocol_b_file_family_support.csv having columns:
  dataset,file_order,file_name,file_path,family,count
and protocol_b_file_manifest.json with split lists of raw filenames.

This is strictly file-level **metadata** integrity, not an interval, capture,
flow overlap, near-duplicate, feature proxy or historical raw-byte audit.
No raw CSV, processed Parquet, test score or model is opened, and nothing written.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

SPLITS = ("train", "val", "test")
NEEDED = {"file_name", "file_path", "family", "count"}


def win_norm(s: str) -> str:
    return str(PureWindowsPath(s.strip().replace("/", "\\"))).casefold()


def filename(s: str) -> str:
    return PureWindowsPath(s.strip().replace("/", "\\")).name.casefold()


def review(root: Path) -> int:
    print("=== C03 LEGACY RAW-SOURCE FILE ASSIGNMENT AUDIT v3 (READ ONLY) ===")
    print(f"Metadata folder: {root}")
    planpath = root / "protocol_b_file_manifest.json"
    supportpath = root / "protocol_b_file_family_support.csv"
    if not planpath.is_file() or not supportpath.is_file():
        print("BLOCKED: missing source metadata:")
        for p in (planpath, supportpath): print(f"  {p.name}: {'present' if p.is_file() else 'missing'}")
        return 2

    errors, warnings = [], []
    try:
        with planpath.open(encoding="utf-8-sig") as f: plan = json.load(f)
        if not isinstance(plan, dict) or any(not isinstance(plan.get(sp), list) for sp in SPLITS):
            print("BLOCKED: file manifest requires train/val/test arrays")
            return 2
        assignments: dict[str, str] = {}
        assigned_raw: dict[str, str] = {}
        for sp in SPLITS:
            for orig in plan[sp]:
                if not isinstance(orig, str) or not orig.strip():
                    errors.append(f"Invalid source file ID in {sp}: {orig!r}")
                    continue
                key = orig.strip().casefold()
                if "\\" in orig or "/" in orig:
                    errors.append(f"Expected a raw filename, got path-like ID in {sp}: {orig}")
                if key in assignments:
                    errors.append(f"File ID assigned repeatedly: {orig!r} ({assignments[key]}, {sp})")
                assignments[key] = sp
                assigned_raw[key] = orig

        with supportpath.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            if not NEEDED.issubset(fields):
                print(f"BLOCKED: support CSV missing required fields {sorted(NEEDED-fields)}")
                return 2
            records = list(reader)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        print(f"BLOCKED: metadata unreadable: {exc}")
        return 2

    print("\n=== SPLIT PLAN ===")
    for sp in SPLITS: print(f"{sp:5}: {len(plan[sp])} assigned raw filenames")
    print(f"Total distinct (case-insensitive) assigned filenames: {len(assignments)}")

    identities: dict[str, dict] = {}
    families_seen: set[tuple[str, str]] = set()
    orders: dict[int, str] = {}
    rows_counted = Counter()
    datasets = set()
    for lineno, r in enumerate(records, 2):
        nm = (r.get("file_name") or "").strip()
        rawpath = (r.get("file_path") or "").strip()
        fam = (r.get("family") or "").strip()
        ds = (r.get("dataset") or "").strip()
        if ds: datasets.add(ds)
        if not nm or not rawpath or not fam:
            errors.append(f"Support CSV line {lineno}: blank file_name, file_path, or family")
            continue
        key = nm.casefold()
        normpath = win_norm(rawpath)
        if filename(rawpath) != key:
            errors.append(f"Support line {lineno}: path basename differs from file_name {nm!r}")
        try:
            n = int(str(r.get("count", "")).strip())
            if n < 0: raise ValueError()
        except ValueError:
            errors.append(f"Support line {lineno}: nonnegative integer family count required")
            continue
        if (key, fam) in families_seen:
            errors.append(f"Support CSV repeated family record for {nm!r} / {fam!r}")
        families_seen.add((key, fam))
        try:
            order = int(str(r.get("file_order", "")).strip()) if "file_order" in fields else None
            if order is not None and order < 0: raise ValueError()
        except ValueError:
            errors.append(f"Support line {lineno}: invalid file_order")
            order = None
        old = identities.get(key)
        if old:
            if old["path"] != normpath:
                errors.append(f"Same file name references different raw source paths: {nm!r}")
            if old["order"] != order:
                errors.append(f"Inconsistent file_order for {nm!r}")
        else:
            identities[key] = {"path": normpath, "original": rawpath, "order": order}
        if order is not None:
            if order in orders and orders[order] != key:
                errors.append(f"Two distinct raw files share file_order={order}: {nm!r}")
            orders[order] = key
        rows_counted[key] += n

    print("\n=== SUPPORT METADATA ===")
    print(f"Support rows: {len(records)}; distinct recorded raw filenames: {len(identities)}")
    print(f"Recorded dataset names: {sorted(datasets)}")
    if datasets and datasets != {"CICIoT2023"}:
        errors.append(f"Unexpected dataset names in support CSV: {sorted(datasets)}")
    missing = set(assignments) - set(identities)
    extra = set(identities) - set(assignments)
    print(f"Split IDs missing from support metadata: {len(missing)}")
    print(f"Support filenames absent from split plan: {len(extra)}")
    if missing: errors.append(f"{len(missing)} split-assigned source files not in support CSV; examples: {sorted(missing)[:3]}")
    if extra: errors.append(f"{len(extra)} support-source files unassigned; examples: {sorted(extra)[:3]}")
    all_paths = defaultdict(set)
    source_paths_by_split = defaultdict(set)
    basename_by_path = defaultdict(set)
    for key, rec in identities.items():
        path = rec["path"]
        all_paths[path].add(key)
        basename_by_path[key].add(path)
        if key in assignments:
            source_paths_by_split[assignments[key]].add(path)
    aliases = {p: names for p, names in all_paths.items() if len(names) > 1}
    if aliases: errors.append(f"{len(aliases)} normalized raw paths associated with multiple file names")
    print(f"Distinct normalized original raw paths: {len(all_paths)}")
    print(f"Multiple filenames pointing at the same path: {len(aliases)}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = source_paths_by_split[a] & source_paths_by_split[b]
        print(f"Cross-split identical raw path {a}/{b}: {len(overlap)}")
        if overlap: errors.append(f"Same normalized raw path assigned to {a} and {b}: {sorted(overlap)[:2]}")

    suffix_groups = Counter()
    for key in identities:
        m = re.fullmatch(r"part-\d+-(.+)-c\d+\.csv", key)
        if m: suffix_groups[m.group(1)] += 1
    print(f"Shared part-file middle ID groups (naming heuristic only): {dict(suffix_groups.most_common(4))}")
    if suffix_groups:
        warnings.append("part-* filenames appear to be dataset shards, not necessarily independent captures or days")
    if "file_order" not in fields:
        warnings.append("no file_order column (not needed for path-level assignment check)")
    if "file_path" in fields and identities:
        print("Recorded paths are metadata references; raw input CSV bytes/existence are NOT checked.")
        warnings.append("Raw source paths may refer to an earlier computer; source bytes/identities not independently verified")

    protocolpath = root / "SPLIT_PROTOCOL.json"
    if protocolpath.is_file():
        try:
            with protocolpath.open(encoding="utf-8-sig") as f: protocol = json.load(f)
            print(f"Declared split protocol: {protocol.get('protocol')}; mode: {protocol.get('protocol_b_split_mode')}")
            if protocol.get("protocol") != "B_day_file": errors.append("Split protocol isn't B_day_file")
            if protocol.get("protocol_b_partition_mode") is None:
                warnings.append("SPLIT_PROTOCOL does not document a raw-file partition mode")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"SPLIT_PROTOCOL unreadable: {exc}")

    if errors:
        print("\nRESULT: FAIL metadata-based source-file assignment integrity")
        for err in errors[:18]: print("  ERROR:", err)
        if len(errors) > 18: print(f"  ... plus {len(errors)-18} more errors")
        return 1
    print("\nRESULT: PASS metadata-consistent, disjoint raw CSV *file-path assignments*")
    print("CAUTION: There are no raw row-interval records in this legacy schema, so interval coverage is NOT verified.")
    print("NOT ESTABLISHED: independent devices/capture days/attacks, duplicate or near-duplicate flows across shards,")
    print("raw CSV byte equality, label-proxy absence, test-data isolation, or genuinely independent test units.")
    for w in dict.fromkeys(warnings): print("  NOTE:", w)
    print("No raw/processed observations, test data, score files, or models accessed; no files written.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("processed_V5/B_day_file/CICIoT2023"))
    args = parser.parse_args()
    sys.exit(review(args.dataset_dir))
