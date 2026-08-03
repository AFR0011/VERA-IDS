#!/usr/bin/env python3
"""Run dataset audit and preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src import data_preparation, dataset_audit  # noqa: E402
from ids_eval_framework.src.paths import load_config  # noqa: E402
from ids_eval_framework.src.synthetic_workflows import run_synthetic_cli  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit raw/processed datasets and build prepared splits.")
    parser.add_argument("--config", default="config/datasets.yml", help="Framework YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Load entrypoints and print the planned native workflow.")
    parser.add_argument("--synthetic", action="store_true", help="Run a bounded deterministic synthetic workflow.")
    parser.add_argument("--skip-audit", action="store_true", help="Skip dataset/schema/leakage audit.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.synthetic:
        run_synthetic_cli("audit_and_prepare")
        return
    config = load_config(args.config)
    if not args.skip_audit:
        dataset_audit.run_dataset_audit(config, dry_run=args.dry_run)
    if not args.skip_prepare:
        data_preparation.run_preparation(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
