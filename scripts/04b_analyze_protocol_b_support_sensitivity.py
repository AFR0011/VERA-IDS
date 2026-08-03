#!/usr/bin/env python3
"""Analyze Protocol B support-threshold sensitivity without model retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src.paths import load_config  # noqa: E402
from ids_eval_framework.src.support_sensitivity import run_support_sensitivity  # noqa: E402
from ids_eval_framework.src.synthetic_workflows import run_synthetic_cli  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate support thresholds against existing split counts."
    )
    parser.add_argument(
        "--config",
        default="config/protocol_b_loao.yml",
        help="Framework YAML config path.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned inputs and outputs.")
    parser.add_argument("--synthetic", action="store_true", help="Run a bounded deterministic synthetic workflow.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.synthetic:
        run_synthetic_cli("support_sensitivity")
        return
    run_support_sensitivity(load_config(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
