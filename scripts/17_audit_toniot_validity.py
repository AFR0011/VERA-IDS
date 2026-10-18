#!/usr/bin/env python3
"""Read-only TON_IoT Network temporal/identity/episode-proxy feasibility audit.

Full-stream counts for fixed diagnostic time bins and documented attack-origin
addresses; bounded, deterministic per-class sampling for address associations.
No model, partition, labels, or source files are modified. The result cannot
identify independent attack campaigns or establish absence of proxy leakage.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

REQUIRED = {"ts", "label", "type", "src_ip", "dst_ip", "src_port", "dst_port", "proto"}
GAP_SECONDS = (300, 1800, 3600)
SEED = 1729


def source_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Processed source directory does not exist: {root}")
    found = []
    for p in root.iterdir():
        m = re.fullmatch(r"Network_dataset_(\d+)\.csv", p.name, re.I)
        if p.is_file() and m:
            found.append((int(m.group(1)), p.name, p))
    if not found:
        raise ValueError(f"No Network_dataset_*.csv found in {root}")
    return [p for _, _, p in sorted(found)]


def attack_source_marker(ip: str) -> bool:
    """Address range described in official dataset README, not inferred source identities."""
    if not ip.startswith("192.168.159."):
        return False
    suffix = ip[len("192.168.159."):]
    return suffix.isdigit() and 30 <= int(suffix) <= 39


def valid_ts(ts: str | None) -> float:
    try:
        value = float(ts) if ts is not None else float("nan")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Unparseable timestamp {ts!r}") from e
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite timestamp {ts!r}")
    return value


def init_family(category: str) -> dict:
    digest = hashlib.sha256(category.encode("utf-8")).digest()
    return {
        "rows": 0,
        "first": None, "last": None,
        "time_min": None, "time_max": None,
        "backwards_transitions": 0,
        "candidate_gaps": {str(s): 0 for s in GAP_SECONDS},
        "attack_range_source_rows": 0,
        "attack_range_destination_rows": 0,
        "sample": [],
        "rng": random.Random(SEED ^ int.from_bytes(digest[:8], "big")),
    }


def scan(root: Path, bin_seconds: int, sample_per_type: int) -> dict:
    files = source_files(root)
    buckets: dict[int, Counter[str]] = defaultdict(Counter)
    families: dict[str, dict] = {}
    rows = 0
    widths = {}
    for file_idx, path in enumerate(files, 1):
        print(f"[{file_idx}/{len(files)}] Validity audit {path.name}", flush=True)
        csv.field_size_limit(16 * 1024 * 1024)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
                raise ValueError(f"Missing essential headers: {path.name}")
            widths[path.name] = len(reader.fieldnames)
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV width: {path.name}")
                t = valid_ts(row.get("ts"))
                category = str(row["type"] or "").strip()
                label = str(row["label"] or "").strip()
                if not category or (category == "normal" and label != "0") or (category != "normal" and label != "1"):
                    raise ValueError(f"Inconsistent/empty type-label mapping in {path.name}")
                src_ip = str(row["src_ip"] or "").strip()
                dst_ip = str(row["dst_ip"] or "").strip()
                data = families.get(category)
                if data is None:
                    data = init_family(category)
                    families[category] = data
                data["rows"] += 1
                rows += 1
                data["time_min"] = t if data["time_min"] is None else min(data["time_min"], t)
                data["time_max"] = t if data["time_max"] is None else max(data["time_max"], t)
                if data["last"] is not None:
                    gap = t - data["last"]
                    if gap < 0:
                        data["backwards_transitions"] += 1
                    else:
                        for seconds in GAP_SECONDS:
                            if gap >= seconds:
                                data["candidate_gaps"][str(seconds)] += 1
                else:
                    data["first"] = t
                data["last"] = t
                buckets[math.floor(t / bin_seconds)][category] += 1
                data["attack_range_source_rows"] += int(attack_source_marker(src_ip))
                data["attack_range_destination_rows"] += int(attack_source_marker(dst_ip))
                sample = data["sample"]
                key = None
                if len(sample) < sample_per_type:
                    key = (src_ip, dst_ip, str(row["src_port"] or "").strip(),
                           str(row["dst_port"] or "").strip(), str(row["proto"] or "").strip())
                    sample.append(key)
                else:
                    chosen = data["rng"].randrange(data["rows"])
                    if chosen < sample_per_type:
                        key = (src_ip, dst_ip, str(row["src_port"] or "").strip(),
                               str(row["dst_port"] or "").strip(), str(row["proto"] or "").strip())
                        sample[chosen] = key
    return {"files": [x.name for x in files], "rows": rows, "families": families,
            "buckets": buckets, "schema_widths": widths}


def sample_summary(families: dict[str, dict]) -> dict:
    # Equal-cap per-family reservoirs: not prevalence-weighted; never export identifiers.
    samples = {typ: item["sample"] for typ, item in families.items()}
    source_sets = {typ: {r[0] for r in sample} for typ, sample in samples.items()}
    dest_sets = {typ: {r[1] for r in sample} for typ, sample in samples.items()}
    benign_src = source_sets.get("normal", set())
    attack_types = [x for x in samples if x != "normal"]
    summary = {}
    for category in sorted(samples):
        records = samples[category]
        other_src = set().union(*(source_sets[t] for t in attack_types if t != category)) if attack_types else set()
        src = source_sets[category]
        dst = dest_sets[category]
        summary[category] = {
            "sample_rows": len(records),
            "distinct_source_addresses_in_sample": len(src),
            "distinct_destination_addresses_in_sample": len(dst),
            "distinct_source_destination_pairs_in_sample": len({r[:2] for r in records}),
            "distinct_five_tuple_identities_in_sample": len(set(records)),
            "distinct_source_ports_in_sample": len({r[2] for r in records}),
            "distinct_destination_ports_in_sample": len({r[3] for r in records}),
            "sample_rows_src_also_in_benign_sample": (
                sum(s in benign_src for s, *_ in records) if category != "normal" else None),
            "sample_rows_src_also_in_other_attack_family_samples": (
                sum(s in other_src for s, *_ in records) if category != "normal" else None),
            "source_address_overlap_with_benign_sample": (
                len(src & benign_src) if category != "normal" else None),
            "source_address_overlap_with_other_attack_family_samples": (
                len(src & other_src) if category != "normal" else None),
        }
    return summary


def report(source: dict, bin_seconds: int, sample_per_type: int) -> dict:
    families = source["families"]
    hourly = source["buckets"]
    family_bins = {x: [] for x in families}
    for index, counts in sorted(hourly.items()):
        for category, count in counts.items():
            if count:
                family_bins[category].append(index)
    spans = [
        {"type": typ, "min": data["time_min"], "max": data["time_max"]}
        for typ, data in families.items() if typ != "normal"
    ]
    spans.sort(key=lambda item: (item["min"], item["type"]))
    strictly_disjoint = bool(spans) and all(
        spans[i]["max"] < spans[i + 1]["min"] for i in range(len(spans) - 1))
    results = {}
    for category, data in sorted(families.items()):
        indices = family_bins[category]
        mixed_benign_bins = sum(1 for i in indices if hourly[i].get("normal", 0) and category != "normal")
        results[category] = {
            "rows": data["rows"],
            "timestamp_min": data["time_min"],
            "timestamp_max": data["time_max"],
            "diagnostic_bin_count": len(indices),
            "bins_with_benign": mixed_benign_bins if category != "normal" else None,
            "backwards_timestamp_transitions_within_type": data["backwards_transitions"],
            "time_gaps_at_least_seconds": dict(data["candidate_gaps"]),
            "rows_with_documented_attack_src_ip": data["attack_range_source_rows"],
            "rows_with_documented_attack_dst_ip": data["attack_range_destination_rows"],
            "documented_attack_src_ip_fraction": round(data["attack_range_source_rows"] / data["rows"], 8),
            "documented_attack_dst_ip_fraction": round(data["attack_range_destination_rows"] / data["rows"], 8),
        }
    bins = [
        {"bin_start_unix_seconds": index * bin_seconds, "counts_by_type": dict(sorted(counts.items()))}
        for index, counts in sorted(hourly.items())
    ]
    return {
        "schema_version": "toniot-temporal-identity-episode-proxy-audit-v1",
        "scope": "diagnostic_temporal_identity_proxy_not_independent_episode_proof_or_split_approval",
        "source_file_count": len(source["files"]),
        "source_file_names": source["files"],
        "processed_rows_scanned": source["rows"],
        "header_columns_per_file": source["schema_widths"],
        "diagnostic_time_bin_seconds": bin_seconds,
        "source_address_range_basis": "Provider ReadMe: attack-source IPs 192.168.159.30 through 192.168.159.39",
        "balanced_reservoir_seed": SEED,
        "max_reservoir_rows_per_type": sample_per_type,
        "attack_family_intervals_strictly_disjoint": strictly_disjoint,
        "attack_intervals_sorted": spans,
        "per_type": results,
        "address_sample_associations": sample_summary(families),
        "diagnostic_time_bins": bins,
        "warnings": [
            "Time bins and timestamp gaps are diagnostic scales, NOT chosen train/validation/test partitions or verified attack episodes.",
            "Source and destination addresses and endpoint 5-tuples are assessed as association and label-leakage proxies; association is not causal proof.",
            "The documented attack-origin IP indicator is evaluated over ALL processed rows, not just sampled rows.",
            "Per-type reservoir is balanced, so sample overlap counts are NOT representative of corpus class prevalence.",
            "A shared IP in different samples does not imply the same capture session, and different IPs do not prove independence.",
            "No observed raw IPs, source-destination pairs, or individual flows were exported.",
            "No full-feature or duplicate-flow comparison is performed.",
            "The source contains no authenticated attack-operation/session manifests; independent attack episodes cannot be certified here.",
            "TON_IoT remains unapproved for primary Protocol B LOAO without user decision and independent support checks.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/17_revision_campaign/toniot/targeted_validity_audit.json"))
    parser.add_argument("--bin-seconds", type=int, default=3600)
    parser.add_argument("--sample-per-type", type=int, default=4096)
    parser.add_argument("--expected-rows", type=int, default=22339021)
    parser.add_argument("--expected-files", type=int, default=23)
    args = parser.parse_args(argv)
    if args.bin_seconds < 1 or args.sample_per_type < 1 or args.expected_rows < 0 or args.expected_files < 0:
        parser.error("Positive bin-seconds, sample-per-type and nonnegative expected counts required")
    try:
        source = scan(args.processed_root, args.bin_seconds, args.sample_per_type)
        if args.expected_files and len(source["files"]) != args.expected_files:
            raise ValueError(f"Expected {args.expected_files} files, observed {len(source['files'])}; no report written")
        if args.expected_rows and source["rows"] != args.expected_rows:
            raise ValueError(f"Expected {args.expected_rows} processed rows, observed {source['rows']}; no report written")
        payload = report(source, args.bin_seconds, args.sample_per_type)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.out.with_name(args.out.name + ".partial")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp.replace(args.out)
        print(f"[PASS] Audited {source['rows']:,} processed rows and {len(source['files'])} files -> {args.out}")
        print("[CAUTION] Diagnostic bins, address associations and gaps do NOT prove independent attack episodes")
        return 0
    except (OSError, csv.Error, UnicodeError, ValueError) as error:
        print(f"[FAIL] Validity audit stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
