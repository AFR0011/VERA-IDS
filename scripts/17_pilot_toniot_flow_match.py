#!/usr/bin/env python3
"""Read-only sampled flow-key matching between TON_IoT ground truth and processed Network.

Stream all ground-truth rows into deterministic per-file/per-class reservoirs,
then scan all processed files for exact six-field flow identity matches.
Only aggregate counts are exported. This is NOT independent label verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re
import sys

IDENTITY_FIELDS = ("ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto")
SEED = 1729


def select_inputs(directory: Path, prefix: str) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    regex = re.compile(re.escape(prefix) + r"_(\d+)\.csv$", re.IGNORECASE)
    matched = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        m = regex.fullmatch(path.name)
        if m:
            matched.append((int(m.group(1)), path.name, path))
    return [entry[-1] for entry in sorted(matched)]


def identity(row: dict) -> tuple[str, ...]:
    return tuple(str(row.get(field) or "").strip() for field in IDENTITY_FIELDS)


def make_reservoir() -> dict:
    return {"seen": 0, "keys": [], "rng": random.Random(SEED)}


def accumulate_reservoir(root: Path, cap: int) -> tuple[dict, int]:
    buckets: dict[tuple[str, str], dict] = {}
    processed_rows = 0
    files = select_inputs(root, "GroundTruth_Network")
    if not files:
        raise ValueError("No GroundTruth_Network_<number>.csv files")
    for index, path in enumerate(files, 1):
        print(f"[GT {index}/{len(files)}] Sampling {path.name}", flush=True)
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not set((*IDENTITY_FIELDS, "type")).issubset(reader.fieldnames):
                raise ValueError(f"Missing GT identity/label headers in {path.name}")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f"Malformed GT row in {path.name}")
                processed_rows += 1
                typ = str(row.get("type") or "").strip()
                bucket = buckets.setdefault((path.name, typ), make_reservoir())
                bucket["seen"] += 1
                key = identity(row)
                chosen = bucket["keys"]
                if len(chosen) < cap:
                    chosen.append(key)
                else:
                    index = bucket["rng"].randrange(bucket["seen"])
                    if index < cap:
                        chosen[index] = key
    return buckets, processed_rows


def count_processed_matches(root: Path, search_keys: set[tuple[str, ...]]) -> tuple[dict, int]:
    found: dict[tuple[str, ...], Counter[tuple[str, str]]] = {}
    total_rows = 0
    files = select_inputs(root, "Network_dataset")
    if not files:
        raise ValueError("No Network_dataset_<number>.csv files")
    for index, path in enumerate(files, 1):
        print(f"[PROC {index}/{len(files)}] Looking up sampled flow keys in {path.name}", flush=True)
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or not set((*IDENTITY_FIELDS, "label", "type")).issubset(reader.fieldnames):
                raise ValueError(f"Missing processed identity/label headers in {path.name}")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f"Malformed processed row in {path.name}")
                total_rows += 1
                key = identity(row)
                if key in search_keys:
                    found.setdefault(key, Counter())[
                        (str(row.get("label") or "").strip(), str(row.get("type") or "").strip())
                    ] += 1
    return found, total_rows


def summarise(buckets: dict, matches: dict) -> list[dict]:
    reports = []
    for (filename, typ), bucket in sorted(buckets.items()):
        keys = bucket["keys"]
        exact = identity_only = absent = 0
        for key in keys:
            counts = matches.get(key)
            if counts is None:
                absent += 1
            elif counts.get(("1", typ), 0) > 0:
                exact += 1
            else:
                identity_only += 1
        reports.append({
            "groundtruth_file": filename,
            "type": typ,
            "groundtruth_rows_as_stored": bucket["seen"],
            "sampled_rows": len(keys),
            "sampled_unique_six_field_keys": len(set(keys)),
            "exact_flow_and_attack_label_matches": exact,
            "identity_matches_without_same_attack_label": identity_only,
            "unmatched_exact_six_field_keys": absent,
        })
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", required=True, type=Path)
    parser.add_argument("--groundtruth-root", required=True, type=Path)
    parser.add_argument("--sample-per-file-type", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path(
        "outputs/17_revision_campaign/toniot/groundtruth_flow_match_pilot.json"))
    args = parser.parse_args()
    if args.sample_per_file_type < 1:
        parser.error("--sample-per-file-type must be >= 1")
    try:
        buckets, gt_count = accumulate_reservoir(args.groundtruth_root, args.sample_per_file_type)
        query_keys = {key for bucket in buckets.values() for key in bucket["keys"]}
        observed, processed_count = count_processed_matches(args.processed_root, query_keys)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        print(f"[FAIL] No report saved: {exc}", file=sys.stderr)
        return 1

    reports = summarise(buckets, observed)
    totals = {
        "sampled_rows_as_stored": sum(r["sampled_rows"] for r in reports),
        "exact_flow_and_attack_label_matches": sum(r["exact_flow_and_attack_label_matches"] for r in reports),
        "identity_matches_without_same_attack_label": sum(
            r["identity_matches_without_same_attack_label"] for r in reports),
        "unmatched_exact_six_field_keys": sum(r["unmatched_exact_six_field_keys"] for r in reports),
    }
    result = {
        "schema_version": "toniot-groundtruth-flow-pilot-v1",
        "scope": "deterministic_sample_exact_textual_identity_not_independent_groundtruth_validation",
        "sample_per_file_type": args.sample_per_file_type,
        "reservoir_seed": SEED,
        "groundtruth_rows_scanned_as_stored": gt_count,
        "processed_rows_scanned": processed_count,
        "unique_sampled_six_field_keys": len(query_keys),
        "totals_as_stored_including_duplicate_gt_files": totals,
        "groundtruth_file_type_groups": reports,
        "warnings": [
            "Matches compare raw text of ts/src_ip/src_port/dst_ip/dst_port/proto plus processed label/type.",
            "Ground-truth #15 and #16 are confirmed duplicate whole files; their sampled counts are both included in as-stored totals.",
            "A match shows two collections share a flow key/label, not that labeling was independently verified.",
            "Unmatched exact-text keys may reflect formatting, transformation, missing flows or source errors; no cause is inferred.",
            "Per-class and per-file sampling is deterministic reservoir sampling, not a full join.",
            "No source rows, addresses, hashed identities, or flow examples are written.",
            "No source/time independent partitions, feature exclusions, or model selection are approved.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".partial")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(f"[PASS] Checked {len(query_keys)} sampled identities against {processed_count:,} processed rows -> {args.out}")
    print("[CAUTION] Direct consistency pilot, not independent labeling validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
