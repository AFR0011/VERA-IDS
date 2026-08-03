#!/usr/bin/env python3
"""Build comparison tables for reference-paper profiles through framework surfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src.paths import load_config  # noqa: E402
from ids_eval_framework.src.reference_framework_eval import (  # noqa: E402
    build_reference_framework_comparison,
    comparison_sources,
    configured_profiles,
)
from ids_eval_framework.src.synthetic_workflows import run_synthetic_cli  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build accuracy-vs-full-framework reference-profile comparison tables.")
    parser.add_argument("--config", default="config/reference_framework_eval.yml", help="Framework YAML config path.")
    parser.add_argument("--smoke", action="store_true", help="Read/write the smoke output root configured in YAML.")
    parser.add_argument("--dry-run", action="store_true", help="Validate comparison inputs without writing outputs.")
    parser.add_argument("--synthetic", action="store_true", help="Run a bounded deterministic synthetic workflow.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.synthetic:
        run_synthetic_cli("reference_comparison")
        return
    config = load_config(args.config)
    if args.dry_run:
        profiles = sorted(configured_profiles(config))
        sources = {name: str(path) for name, path in comparison_sources(config, smoke=args.smoke).items()}
        print(f"Reference profiles: {profiles}")
        print(f"Comparison sources: {sources}")
        return
    out_dir = build_reference_framework_comparison(config, smoke=args.smoke)
    print(f"Wrote reference-framework comparison tables to: {out_dir}")


if __name__ == "__main__":
    main()
