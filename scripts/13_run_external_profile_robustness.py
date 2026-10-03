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
    parser.add_argument(
        "--protocol-a-processed-root",
        help="Path to Protocol-A prepared-data root containing A_stratified/.",
    )
    parser.add_argument(
        "--protocol-b-audit-root",
        action="append",
        help="Support-audit root containing dataset/manifests/*.json. Repeat as needed.",
    )
    parser.add_argument(
        "--protocol-b-processed",
        action="append",
        metavar="DATASET=PATH",
        help="Rebase stale Protocol-B manifest processed_dir for one dataset. Repeat as needed.",
    )
    return parser


def _parse_processed_overrides(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit("--protocol-b-processed must use DATASET=PATH")
        dataset, path = value.split("=", 1)
        dataset = dataset.strip()
        path = path.strip()
        if not dataset or not path:
            raise SystemExit("--protocol-b-processed must use non-empty DATASET=PATH")
        out[dataset] = path
    return out


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
        protocol_a_processed_root=args.protocol_a_processed_root,
        protocol_b_audit_roots=args.protocol_b_audit_root,
        protocol_b_processed_overrides=_parse_processed_overrides(args.protocol_b_processed),
    )
    print(f"External-profile robustness output root: {out}")


if __name__ == "__main__":
    main()
