#!/usr/bin/env python3
"""Run Protocol B support audit and manifest generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src.paths import load_config  # noqa: E402
from ids_eval_framework.src.support_audit import run_support_audit  # noqa: E402
from ids_eval_framework.src.synthetic_workflows import run_synthetic_cli  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Protocol B/LOAO support eligibility.")
    parser.add_argument("--config", default="config/protocol_b_loao.yml", help="Framework YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned support-audit call.")
    parser.add_argument("--synthetic", action="store_true", help="Run a bounded deterministic synthetic workflow.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.synthetic:
        run_synthetic_cli("support_audit")
        return
    run_support_audit(load_config(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
