#!/usr/bin/env python3
"""Run resumable repeated-seed reliability experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src.paths import load_config  # noqa: E402
from ids_eval_framework.src.seed_reliability import (  # noqa: E402
    parse_csv_arg,
    parse_seed_args,
    run_seed_reliability,
)
from ids_eval_framework.src.synthetic_workflows import run_synthetic_cli  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated-seed reliability lanes.")
    parser.add_argument(
        "--config",
        default="config/seed_reliability.yml",
        help="Seed-reliability YAML config path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and planned native calls without training or writing state files.",
    )
    parser.add_argument("--synthetic", action="store_true", help="Run a bounded deterministic synthetic workflow.")
    parser.add_argument(
        "--lanes",
        nargs="*",
        help="Optional lane names, comma-separated or space-separated.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        help="Optional seed values, comma-separated or space-separated.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Rerun items that have error.json and no run_complete.json.",
    )
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help="Remove stale per-item lock files before running. Use only after confirming no run is active.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild seed reliability summary CSVs from existing completed artifacts only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.synthetic:
        run_synthetic_cli("seed_reliability")
        return
    run_seed_reliability(
        load_config(args.config),
        lanes=parse_csv_arg(args.lanes),
        seeds=parse_seed_args(args.seeds),
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        force_unlock=args.force_unlock,
        summary_only=args.summary_only,
    )


if __name__ == "__main__":
    main()
