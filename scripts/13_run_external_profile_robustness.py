#!/usr/bin/env python3
"""Run repeated external classifier-profile robustness checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src.external_profile_robustness import run_external_profile_robustness  # noqa: E402
from ids_eval_framework.src.paths import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run literature-derived classifier profiles through repeated common VERA conditions."
    )
    parser.add_argument("--config", default="config/external_profile_robustness.yml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--skip-protocol-a", action="store_true")
    parser.add_argument("--skip-protocol-b", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run_external_profile_robustness(
        load_config(args.config),
        dry_run=args.dry_run,
        smoke=args.smoke,
        profiles=args.profile,
        seeds=args.seed,
        skip_protocol_a=args.skip_protocol_a,
        skip_protocol_b=args.skip_protocol_b,
    )
    print(f"External-profile robustness output root: {out}")


if __name__ == "__main__":
    main()
