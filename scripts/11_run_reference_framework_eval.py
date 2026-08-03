#!/usr/bin/env python3
"""Run reference-paper model profiles through the full framework surfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src.paths import load_config  # noqa: E402
from ids_eval_framework.src.reference_framework_eval import run_reference_framework_eval  # noqa: E402
from ids_eval_framework.src.synthetic_workflows import run_synthetic_cli  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Adewole/Neto reference profiles through framework evaluation.")
    parser.add_argument("--config", default="config/reference_framework_eval.yml", help="Framework YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print/validate planned calls without training.")
    parser.add_argument("--synthetic", action="store_true", help="Run a bounded deterministic synthetic workflow.")
    parser.add_argument("--smoke", action="store_true", help="Use small row caps and smoke output root.")
    parser.add_argument("--profile", action="append", help="Filter profile: adewole2025_xgb_profile or neto2023_rf_profile.")
    parser.add_argument("--dataset", action="append", help="Filter Protocol A datasets.")
    parser.add_argument("--no-resume", action="store_true", help="Re-run Protocol A profiles even if complete compatible runs exist.")
    parser.add_argument("--skip-protocol-a", action="store_true")
    parser.add_argument("--skip-protocol-b", action="store_true")
    parser.add_argument("--skip-open-set", action="store_true")
    parser.add_argument("--skip-sink-aware", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.synthetic:
        run_synthetic_cli("reference_profile")
        return
    run_reference_framework_eval(
        load_config(args.config),
        dry_run=args.dry_run,
        smoke=args.smoke,
        resume=not args.no_resume,
        profiles=args.profile,
        datasets=args.dataset,
        skip_protocol_a=args.skip_protocol_a,
        skip_protocol_b=args.skip_protocol_b,
        skip_open_set=args.skip_open_set,
        skip_sink_aware=args.skip_sink_aware,
        skip_comparison=args.skip_comparison,
    )


if __name__ == "__main__":
    main()
