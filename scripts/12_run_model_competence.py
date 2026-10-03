#!/usr/bin/env python3
"""Run the staged Protocol-A model-competence sanity check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src.model_competence import run_model_competence  # noqa: E402
from ids_eval_framework.src.paths import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run staged conventional-model competence checks under Protocol A."
    )
    parser.add_argument("--config", default="config/model_competence.yml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--model", action="append")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run_model_competence(
        load_config(args.config),
        dry_run=args.dry_run,
        smoke=args.smoke,
        datasets=args.dataset,
        seeds=args.seed,
        model_families=args.model,
    )
    print(f"Model-competence output root: {out}")


if __name__ == "__main__":
    main()
