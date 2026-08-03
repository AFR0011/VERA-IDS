#!/usr/bin/env python3
"""Run open-set and sink-aware rejection evaluations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src import open_set_rejection  # noqa: E402
from ids_eval_framework.src.paths import load_config  # noqa: E402
from ids_eval_framework.src.synthetic_workflows import run_synthetic_cli  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run uncertainty/open-set and sink-aware rejectors.")
    parser.add_argument(
        "--lane",
        choices=("validation-selected", "exploratory-grid"),
        default="validation-selected",
        help="Explicit experimental lane; neither lane is universally canonical.",
    )
    parser.add_argument("--config", help="Override the selected lane's framework YAML config path.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned rejector calls.")
    parser.add_argument("--synthetic", action="store_true", help="Run a bounded deterministic synthetic workflow.")
    parser.add_argument("--skip-sink-aware", action="store_true", help="Skip sink-aware rejector replay.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.synthetic:
        run_synthetic_cli(f"rejection_{args.lane.replace('-', '_')}")
        return
    defaults = {
        "validation-selected": "config/open_set_validation_selected_rejection.yml",
        "exploratory-grid": "config/open_set_exploratory_threshold_grid.yml",
    }
    config = load_config(args.config or defaults[args.lane])
    open_set_rejection.run_open_set_baselines(config, dry_run=args.dry_run)
    if not args.skip_sink_aware:
        open_set_rejection.run_sink_aware_rejector(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
