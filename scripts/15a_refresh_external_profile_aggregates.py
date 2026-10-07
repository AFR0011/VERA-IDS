#!/usr/bin/env python3
"""Rebuild external-profile aggregate CSVs from existing per-seed outputs only.

No model fitting is performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.src.external_profile_robustness import (  # noqa: E402
    discover_seed_ids,
    write_aggregate_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh external-profile aggregate summaries from existing seed outputs."
    )
    parser.add_argument(
        "--out-root",
        default="outputs/13_external_profile_robustness",
    )
    args = parser.parse_args()

    root = (REPO_ROOT / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    seeds = discover_seed_ids(root, [])
    if not seeds:
        raise SystemExit(f"No seed_* directories found under {root}")

    print(f"Refreshing aggregates from existing seeds: {seeds}", flush=True)
    write_aggregate_outputs(root, seeds)
    print(f"Refreshed: {root / 'summary'}", flush=True)


if __name__ == "__main__":
    main()
