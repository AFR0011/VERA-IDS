#!/usr/bin/env python3
"""Read-only IoT-23 v1 capture/Zeek-label feasibility audit.

PRE-FLIGHT (default): first N flow rows of each native conn.log.labeled.
CENSUS (explicit): stream every flow row in every source file.

No payloads, identifiers, flow rows, addresses, binaries or learned features
are read into a research output. Native labels are not mapped to VERA families.
Capture names are candidate source units, NOT certified independent runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import re
import sys

SCENARIO = re.compile(r"^CTU-(IoT-Malware|Honeypot)-Capture-[A-Za-z0-9-]+$")
LOG_NAMES = {"conn.log.labeled", "conn.log.labeled.gz"}
REQUIRED = {"ts", "uid", "id.orig_h", "id.resp_h", "label", "detailed-label"}
EXPECTED = {"malware": 20, "benign_device": 3}


def within(root: Path, path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def source_inventory(root: Path) -> list[dict]:
    if not root.is_dir():
        raise ValueError("Extracted IoT-23 source directory does not exist")
    findings = []
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.name not in LOG_NAMES:
            continue
        if candidate.is_symlink() or not within(root, candidate):
            raise ValueError("Symlink or source file escapes input root")
        matches = [p for p in candidate.parents if SCENARIO.fullmatch(p.name)]
        if len(matches) != 1 or not within(root, matches[0]):
            raise ValueError("Labeled log is outside an unambiguous CTU scenario")
        scenario_dir = matches[0]
        scenario_type = "malware" if "IoT-Malware" in scenario_dir.name else "benign_device"
        findings.append({
            "capture_id": scenario_dir.name,
            "source_kind": scenario_type,
            "log_path": candidate,
            "log_relative": candidate.relative_to(root).as_posix(),
            "log_bytes": candidate.stat().st_size,
            "readme_present": (scenario_dir / "README.md").is_file(),
            "root_pcap_file_count": sum(p.is_file() for p in scenario_dir.glob("*.pcap")),
        })
    findings.sort(key=lambda a: a["capture_id"])
    seen = set()
    for item in findings:
        if item["capture_id"] in seen:
            raise ValueError("Multiple labeled flow files for one scenario: " + item["capture_id"])
        seen.add(item["capture_id"])
    return findings


def read_labeled(path: Path, limit: int, progress: int) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    counts = Counter()
    detailed = Counter()
    joint = Counter()
    days = Counter()
    rows = malformed = missing_timestamps = invalid_timestamps = 0
    first_errors = []
    fields = None
    metadata_head = []
    time_min = time_max = None
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if line.startswith("#"):
                if line.startswith("#fields"):
                    new_fields = line.split()[1:]
                    if fields is not None and fields != new_fields:
                        raise ValueError("Conflicting #fields declarations in " + path.name)
                    fields = new_fields
                elif line.startswith("#separator") or line.startswith("#set_separator") or line.startswith("#empty_field"):
                    if len(metadata_head) < 6:
                        metadata_head.append(line.split(maxsplit=1)[0])
                continue
            if fields is None:
                raise ValueError("Zeek #fields header missing or after data rows: " + path.name)
            if limit and rows >= limit:
                break
            rows += 1
            parts = line.split()
            if len(parts) != len(fields):
                malformed += 1
                if len(first_errors) < 10:
                    first_errors.append({
                        "source_line_number": line_number,
                        "expected_fields": len(fields),
                        "observed_fields": len(parts),
                    })
                continue
            if not REQUIRED.issubset(fields):
                raise ValueError("Missing required native label/identity metadata fields")
            positions = {name: fields.index(name) for name in ("ts", "label", "detailed-label")}
            a = parts[positions["label"]]
            b = parts[positions["detailed-label"]]
            counts[a] += 1
            detailed[b] += 1
            joint[(a, b)] += 1
            try:
                ts = float(parts[positions["ts"]])
                if not math.isfinite(ts) or not (946684800 <= ts < 2082758400):
                    raise ValueError("Implausible timestamp")
                time_min = ts if time_min is None else min(time_min, ts)
                time_max = ts if time_max is None else max(time_max, ts)
                if limit == 0:
                    # Full-population day occupancy only. No event/session grouping.
                    day = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
                    days[day] += 1
            except ValueError:
                invalid_timestamps += 1
            if progress and rows % progress == 0:
                print(f"    {path.name}: {rows:,} Zeek rows inspected", flush=True)
    if fields is None:
        raise ValueError("No #fields header in source flow file")
    if not REQUIRED.issubset(fields):
        raise ValueError("Required fields absent: " + ",".join(sorted(REQUIRED - set(fields))))
    def utc(ts: float | None) -> str | None:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts is not None else None
    return {
        "declared_columns": fields,
        "has_required_native_columns": True,
        "rows_inspected": rows,
        "rows_malformed_width": malformed,
        "first_malformed_line_descriptors": first_errors,
        "rows_invalid_or_implausible_timestamp": invalid_timestamps,
        "first_zeek_control_tokens": metadata_head,
        "native_binary_label_counts": dict(sorted(counts.items())),
        "native_detailed_label_counts": dict(sorted(detailed.items())),
        "native_label_pair_counts": [
            {"label": a, "detailed_label": b, "count": v}
            for (a, b), v in sorted(joint.items())
        ],
        "inspected_ts_min_utc": utc(time_min),
        "inspected_ts_max_utc": utc(time_max),
        "utc_day_counts": dict(sorted(days.items())) if limit == 0 else {},
    }


def run(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True,
                   help="Extracted v1 archive root; nested opt/Malware-Project layout is supported")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("preflight", "census"), default="preflight")
    p.add_argument("--sample-rows-per-capture", type=int, default=512)
    p.add_argument("--progress-every", type=int, default=2_000_000)
    p.add_argument("--expected-captures", type=int, default=23)
    args = p.parse_args(argv)
    if args.sample_rows_per_capture < 1 or args.progress_every < 0 or args.expected_captures < 0:
        p.error("Sample, progress and expected capture counts must be valid")
    try:
        items = source_inventory(args.input_root)
        if not items:
            raise ValueError("No conn.log.labeled under recognized CTU source capture directories")
        limit = args.sample_rows_per_capture if args.phase == "preflight" else 0
        results = []
        for idx, item in enumerate(items, 1):
            print(f"[{idx}/{len(items)}] {args.phase}: {item['capture_id']}", flush=True)
            parsed = read_labeled(item["log_path"], limit, args.progress_every if limit == 0 else 0)
            results.append({
                key: value for key, value in item.items() if key != "log_path"
            } | parsed)
        kinds = Counter(item["source_kind"] for item in results)
        problems = []
        if args.expected_captures and len(results) != args.expected_captures:
            problems.append("Unexpected number of original capture scenarios")
        if args.expected_captures == 23 and dict(kinds) != EXPECTED:
            problems.append("Original 20-malware / three-benign capture mix not observed")
        if any(not row["readme_present"] for row in results):
            problems.append("At least one native capture README.md is missing")
        if any(row["rows_malformed_width"] or row["rows_invalid_or_implausible_timestamp"]
               for row in results):
            problems.append("Malformed source rows or invalid timestamps observed")
        schemas = defaultdict(list)
        for row in results:
            schemas[",".join(row["declared_columns"])].append(row["capture_id"])
        distinct_labels = sorted({k for row in results
                                  for k in row["native_detailed_label_counts"]})
        support = []
        for label in distinct_labels:
            holders = [
                {"capture_id": row["capture_id"],
                 "source_kind": row["source_kind"],
                 "inspected_rows": row["native_detailed_label_counts"].get(label, 0)}
                for row in results if row["native_detailed_label_counts"].get(label, 0)
            ]
            support.append({"native_exact_detailed_label": label,
                            "captures_with_observed_rows": len(holders),
                            "capture_counts": holders})
        output = {
            "schema_version": "iot23-native-source-label-audit-v1",
            "mode": args.phase,
            "complete_source_row_census": args.phase == "census" and not problems,
            "rows_per_capture_limit": limit,
            "capture_scenarios_found": len(results),
            "source_kinds": dict(sorted(kinds.items())),
            "total_rows_inspected": sum(x["rows_inspected"] for x in results),
            "schema_variants": [
                {"capture_ids": names, "columns": columns.split(",")}
                for columns, names in sorted(schemas.items())
            ],
            "capture_records": results,
            "native_exact_detailed_label_support": support,
            "problems": problems,
            "warnings": [
                "Preflight native-label counts are PREFIX SAMPLES, NOT population counts or Protocol B support.",
                "Census labels are publisher rule-based flow labels, not independently verified attack-run identities.",
                "Native detailed labels may encode behaviors, malware names, or compound categories; no mapping adopted.",
                "Do not infer independent capture/run repetitions from source names alone; verify provenance and support.",
                "No train/validation/test partitions, feature exclusions, thresholds, models or source changes are made.",
                "No raw IPs, UIDs, flow rows or packet payloads exported.",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".partial")
        temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.out)
        status = "[PARTIAL]" if problems else "[PASS]"
        print(f"{status} {len(results)} scenarios, {output['total_rows_inspected']:,} inspected rows -> {args.out}")
        if problems:
            print("Issues: " + "; ".join(problems), file=sys.stderr)
            return 2
        return 0
    except (OSError, ValueError, TypeError, UnicodeError, gzip.BadGzipFile) as e:
        print(f"[FAIL] IoT-23 source audit: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
