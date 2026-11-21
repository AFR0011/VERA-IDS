#!/usr/bin/env python3
"""C03 Protocol-B source-unit integrity audit v2: metadata-only, read-only.

Run from VERA-IDS root:
    python -B c03_source_unit_integrity_audit_v2.py

Uses the canonical protocol_b_unit_manifest.csv when available, or reconstructs
unit metadata from protocol_b_file_family_support.csv (one row per unit/family).
Only metadata and source-assignment JSON are opened. No Parquet, model, test
scores, or observations are read; no files are written.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
STATIC = ("unit_id", "source_file_name", "source_file_path", "row_start", "row_end")
VALID_FIELDS = set(STATIC)
INTERVAL_RE = re.compile(r"^(.*)::part(\d+)of(\d+)$", re.I)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        r = csv.DictReader(fh)
        fields = set(r.fieldnames or ())
        rows = list(r)
    return fields, rows


def pnorm(s: str) -> str:
    return str(s).strip().replace("/", "\\").casefold()


def check_assignment(path: Path):
    with path.open("r", encoding="utf-8-sig") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or any(not isinstance(data.get(s), list) for s in SPLITS):
        raise ValueError("split JSON must contain train, val and test lists")
    counts = Counter()
    split_by_uid = {}
    issues = []
    for sp in SPLITS:
        for uid in data[sp]:
            if not isinstance(uid, str) or not uid.strip():
                issues.append(f"Invalid unit ID in {sp}: {uid!r}")
                continue
            counts[uid] += 1
            if uid in split_by_uid:
                issues.append(f"Unit assigned more than once: {uid!r} ({split_by_uid[uid]}, {sp})")
            split_by_uid[uid] = sp
    return data, split_by_uid, issues


def aggregate_support(rows):
    """Collapse per-family support rows into source-unit records, checking metadata."""
    result = {}
    issues = []
    by_family = set()
    for ix, row in enumerate(rows, start=2):
        uid = row.get("unit_id", "").strip()
        if not uid:
            issues.append(f"Empty unit_id on support-file line {ix}")
            continue
        fam = row.get("family", "").strip()
        if fam:
            if (uid, fam) in by_family:
                issues.append(f"Repeated unit/family support row: {uid!r} / {fam!r}")
            by_family.add((uid, fam))
        core = {k: str(row.get(k, "")).strip() for k in STATIC}
        other = result.get(uid)
        if other is not None:
            for k in STATIC:
                if core[k] != other[k]:
                    issues.append(f"Conflicting support metadata for unit={uid!r} field={k}")
        else:
            result[uid] = core
    return result, issues


def source_id_fallback(split_by_uid):
    """Useful diagnostic when the only available record is assignment IDs."""
    print("\n=== ASSIGNMENT-ONLY DIAGNOSTICS (NOT AN INDEPENDENCE AUDIT) ===")
    groups = defaultdict(set)
    syntaxes = Counter()
    for uid, sp in split_by_uid.items():
        match = INTERVAL_RE.fullmatch(uid)
        if match:
            base = match.group(1)
            syntaxes["contiguous_subfile_style"] += 1
        else:
            base = uid
            syntaxes["other_or_whole_file_style"] += 1
        groups[pnorm(base)].add(sp)
    print(f"Unit ID patterns: {dict(syntaxes)}")
    print(f"Distinct inferred source basenames: {len(groups)}")
    shared = {s: ss for s, ss in groups.items() if len(ss) > 1}
    print(f"Inferred source basenames assigned across splits: {len(shared)}")
    for source, spl in list(sorted(shared.items()))[:10]:
        print(f"  {source}: {','.join(sorted(spl))}")
    print("Only IDs are available: source PATHS, source-row intervals, and basename ambiguity cannot be verified.")


def find_records(root: Path, assigned_ids: set[str]):
    """Require exact ID coverage; never silently combine a different split plan."""
    unit = root / "protocol_b_unit_manifest.csv"
    support = root / "protocol_b_file_family_support.csv"
    other = root / "protocol_b_wholefile_unit_manifest.csv"
    other_support = root / "protocol_b_wholefile_file_family_support.csv"
    candidates = [unit, support, other, other_support]
    print("\n=== METADATA DISCOVERY ===")
    found = []
    for path in candidates:
        print(f"{path.name}: {'present' if path.is_file() else 'missing'}")
        if not path.is_file():
            continue
        fields, rows = read_csv(path)
        if not VALID_FIELDS.issubset(fields):
            print(f"  fields insufficient: missing={sorted(VALID_FIELDS - fields)}")
            continue
        if "unit_manifest" in path.name:
            by_uid = {}
            duplicate = []
            for row in rows:
                uid = str(row["unit_id"]).strip()
                if uid in by_uid:
                    duplicate.append(uid)
                by_uid[uid] = row
            issues = [f"Repeated unit_id in unit manifest: {x!r}" for x in duplicate]
        else:
            by_uid, issues = aggregate_support(rows)
        ids = set(by_uid)
        missing = assigned_ids - ids
        extra = ids - assigned_ids
        print(f"  rows={len(rows)} distinct_units={len(ids)} matching_plan={'yes' if not missing and not extra else 'NO'}"
              f" missing={len(missing)} extra={len(extra)} metadata_issues={len(issues)}")
        if not missing and not extra:
            found.append((path, by_uid, issues, "support" if "family_support" in path.name else "unit"))
    if not found:
        return None
    # Prefer exact canonical plan files to historical whole-file counterfactual reports.
    return found[0]


def inspect(root: Path) -> int:
    print("=== C03 SOURCE-UNIT INTEGRITY AUDIT v2 (READ ONLY) ===")
    print(f"Dataset metadata directory: {root}")
    manifest = root / "protocol_b_file_manifest.json"
    if not manifest.is_file():
        print("BLOCKED: protocol_b_file_manifest.json missing")
        return 2
    try:
        assigned, split_by_uid, issues = check_assignment(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"BLOCKED: cannot parse split assignments: {e}")
        return 2
    print("\n=== SPLIT ASSIGNMENTS ===")
    for sp in SPLITS:
        print(f"{sp:5}: assigned units={len(assigned[sp])}")
    if issues:
        print("FAIL: invalid split assignments")
        for issue in issues[:12]:
            print(" -", issue)
        return 1
    selected = find_records(root, set(split_by_uid))
    if selected is None:
        source_id_fallback(split_by_uid)
        print("\nRESULT: BLOCKED / INCONCLUSIVE — no matching source-unit interval metadata found")
        print("The split assignment exists but a unit interval/actual raw path cannot be checked.")
        print("No observation/test rows or model scores read; no files changed.")
        return 2
    path, units, meta_issues, mode = selected
    issues.extend(meta_issues)
    if "wholefile" in path.name:
        print("WARNING: recovered matching unit IDs from a whole-file baseline report; plan provenance must be confirmed.")
    print(f"\nUSING: {path.name} ({'reconstructed from per-family rows' if mode=='support' else 'unit records'})")
    if mode == "support":
        print("Raw row counts inferred from row_end-row_start; independently stored raw_row_count not available.")
    paths = {sp: set() for sp in SPLITS}
    basenames = defaultdict(set)
    intervals = defaultdict(list)
    modes = Counter()
    planned = Counter()
    for uid, row in units.items():
        rawpath = pnorm(row.get("source_file_path", ""))
        name = str(row.get("source_file_name", "")).strip().casefold()
        if not rawpath or not name:
            issues.append(f"Missing raw source path/name for unit {uid!r}")
            continue
        basename = rawpath.rsplit("\\", 1)[-1]
        if basename != name:
            issues.append(f"Mismatched source name and path basename for unit {uid!r}: {name!r} != {basename!r}")
        try:
            st, en = int(row["row_start"]), int(row["row_end"])
        except (KeyError, ValueError, TypeError):
            issues.append(f"Invalid numeric interval for unit {uid!r}")
            continue
        if not (0 <= st < en):
            issues.append(f"Invalid interval for unit {uid!r}: [{st},{en})")
            continue
        if "raw_row_count" in row:
            try:
                cnt = int(row["raw_row_count"])
                if cnt != en-st:
                    issues.append(f"Wrong raw_row_count for {uid!r}: expected {en-st}, found {cnt}")
            except (ValueError, TypeError):
                issues.append(f"Invalid raw_row_count for {uid!r}")
        sp = split_by_uid[uid]
        paths[sp].add(rawpath)
        basenames[name].add(rawpath)
        intervals[rawpath].append((st, en, sp, uid))
        planned[sp] += en-st
        modes[str(row.get("partition_mode", "unrecorded"))] += 1
    print("\n=== RAW SOURCE MEMBERSHIP ===")
    print(f"Partition modes: {dict(modes)}")
    for sp in SPLITS:
        print(f"{sp:5}: raw_source_paths={len(paths[sp])}, planned_raw_rows={planned[sp]:,}")
    ambiguous = [nm for nm, pp in basenames.items() if len(pp)>1]
    print(f"Ambiguous basenames resolving to different paths: {len(ambiguous)}")
    for a,b in (("train","val"),("train","test"),("val","test")):
        shared=paths[a]&paths[b]
        print(f"Shared raw source paths {a}/{b}: {len(shared)}")
        for s in sorted(shared)[:5]:
            print("  ",s)
    overlap=gaps=nonzero=multi=0
    cross_overlap=0
    for rawpath, iv in intervals.items():
        iv.sort(key=lambda x: (x[0],x[1]))
        multi += len(iv)>1
        nonzero += iv[0][0]!=0
        prev=iv[0]
        for curr in iv[1:]:
            if curr[0]<prev[1]:
                overlap+=1
                if curr[2]!=prev[2]: cross_overlap+=1
            elif curr[0]>prev[1]:
                gaps+=1
            if curr[1]>prev[1]: prev=curr
    print("\n=== SOURCE-INTERVAL INTEGRITY ===")
    print(f"Multiple units from same raw path: {multi}")
    print(f"Overlapping raw row intervals: {overlap} (cross-split={cross_overlap})")
    print(f"Source-interval gaps: {gaps}; nonzero first offsets: {nonzero}")
    if overlap:
        issues.append("Raw source-row intervals overlap")
    if ambiguous:
        issues.append("Identical basenames refer to different source paths")
    if gaps or nonzero:
        print("CAUTION: partial coverage or unrecorded units; cannot infer complete coverage.")
    if issues:
        print("\nRESULT: FAIL source-plan integrity")
        for issue in issues[:15]: print(" -",issue)
        if len(issues)>15: print(f" ... and {len(issues)-15} more")
        return 1
    print("\nRESULT: PASS manifest-consistent raw-source unit assignment and recorded interval nonoverlap")
    if any(paths[a]&paths[b] for a,b in (("train","val"),("train","test"),("val","test"))):
        print("SOURCE-DEPENDENCE FLAG: at least one raw source path is used across distinct splits.")
    print("LIMIT: metadata integrity only, not raw-file hashes, independent captures, flow uniqueness, proxy-free values, or test isolation.")
    print("No observation/test rows, fitted models, or score files read; no file writes.")
    return 0


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--dataset-dir", type=Path, default=Path("processed_V5/B_day_file/CICIoT2023"))
    args = arg.parse_args()
    raise SystemExit(inspect(args.dataset_dir))
