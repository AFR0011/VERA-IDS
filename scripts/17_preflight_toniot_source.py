#!/usr/bin/env python3
"""Read-only inventory of candidate TON_IoT network CSVs.

This is a feasibility inventory, NOT an approved taxonomy, leakage audit,
partition implementation, or Protocol-B support audit. It never saves rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

LABEL_NAMES = {
    "label", "type", "attack", "attack_type", "attack_category",
    "category", "class", "attack_cat", "subtype", "sub_type",
}
TIME_NAMES = {"ts", "timestamp", "time", "start_time", "end_time", "stime", "ltime"}
IDENTIFIER_HINTS = {"src_ip", "dst_ip", "src_port", "dst_port", "uid", "id", "flow_id", "ip", "port"}


def classify_column(name: str) -> list[str]:
    """Return candidates only; no feature exclusion is decided here."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    result = []
    if key in LABEL_NAMES:
        result.append("label_candidate")
    if key in TIME_NAMES or key.endswith("_timestamp"):
        result.append("time_candidate")
    if key in IDENTIFIER_HINTS or key.endswith("_ip") or key.endswith("_port"):
        result.append("identity_candidate")
    return result


def csv_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.name.lower().endswith((".csv", ".csv.gz")) else []
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name.lower().endswith((".csv", ".csv.gz"))),
        key=lambda p: p.as_posix().lower(),
    )


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def examine(path: Path, root: Path, limit: int, with_hashes: bool) -> dict:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    relative = path.name if root.is_file() else path.relative_to(root).as_posix()
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        if not names:
            raise ValueError(f"Missing CSV header: {relative}")
        duplicate_names = sorted({name for name, count in Counter(names).items() if count > 1})
        candidates = {name: classify_column(name) for name in names if classify_column(name)}
        label_columns = [name for name, kinds in candidates.items() if "label_candidate" in kinds]
        label_counts = {name: Counter() for name in label_columns}
        rows = 0
        truncated = False
        for row in reader:
            if limit and rows >= limit:
                truncated = True
                break
            rows += 1
            for name in label_columns:
                label_counts[name][str(row.get(name, ""))] += 1
    record = {
        "source_file": relative,
        "size_bytes": path.stat().st_size,
        "n_columns": len(names),
        "columns": names,
        "duplicate_headers": duplicate_names,
        "column_candidates": candidates,
        "observed_rows": rows,
        "scan_complete": not truncated,
        "label_value_counts": {
            name: dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:50])
            for name, counts in label_counts.items()
        },
        "label_counts_truncated": {name: len(counts) > 50 for name, counts in label_counts.items()},
        "sha256": checksum(path) if with_hashes else None,
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="Directory containing official processed network CSVs, or one CSV file")
    parser.add_argument("--out", type=Path, default=Path("outputs/17_revision_campaign/toniot/source_preflight.json"))
    parser.add_argument("--max-rows-per-file", type=int, default=10000, help="0 means count every row; otherwise bounded sample")
    parser.add_argument("--sha256", action="store_true", help="Hash every entire source file (potentially slow)")
    parser.add_argument("--dry-run", action="store_true", help="List candidate input files without reading rows or writing results")
    args = parser.parse_args(argv)
    if args.max_rows_per_file < 0:
        parser.error("--max-rows-per-file must be nonnegative")
    root = args.input_root.expanduser().resolve()
    if not root.exists():
        parser.error(f"Input does not exist: {root}")
    candidates = csv_files(root)
    if not candidates:
        parser.error(f"No CSV or CSV.GZ inputs found under {root}")
    if args.dry_run:
        print(f"[DRY-RUN] {len(candidates)} candidate CSV file(s)")
        for item in candidates:
            print(f"  {item}")
        return 0
    try:
        records = [examine(item, root, args.max_rows_per_file, args.sha256) for item in candidates]
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"[FAIL] Preflight stopped: {error}", file=sys.stderr)
        return 1
    result = {
        "schema_version": "toniot-source-preflight-v1",
        "scope": "read_only_inventory_not_a_leakage_or_support_audit",
        "input_root": str(root),
        "file_count": len(records),
        "max_rows_per_file": args.max_rows_per_file,
        "file_hashes_complete": args.sha256,
        "row_count_complete": all(record["scan_complete"] for record in records),
        "files": records,
        "warnings": [
            "Label candidates and identities are heuristics, not approved taxonomy or feature exclusions.",
            "Counts in files marked scan_complete=false describe only sampled rows.",
            "File names are not proof of independent acquisition or valid source/time partitions.",
            "This inventory does not establish LOAO support, no-leakage, or admissible splits.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[PASS] Inventoried {len(records)} CSV file(s) -> {args.out}")
    print("[CAUTION] Not a dataset approval or leakage/support audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
