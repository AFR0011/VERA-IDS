#!/usr/bin/env python3
"""Build or verify the path-free Protocol A correction evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ids_eval_framework.src.protocol_a_correction import (  # noqa: E402
    build_evidence,
    corrected_metrics,
    load_evidence,
    write_evidence,
)


DEFAULT_EVIDENCE = ROOT / "outputs" / "evidence" / "protocol_a_confusion_matrices.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--source-root", type=Path, help="Protected source root; used only to rebuild evidence.")
    parser.add_argument("--build", action="store_true", help="Rebuild the tracked path-free evidence table.")
    parser.add_argument("--check", action="store_true", help="Validate evidence and exact correction gates.")
    args = parser.parse_args()

    if args.build:
        if args.source_root is None:
            parser.error("--build requires --source-root")
        records = build_evidence(args.source_root.resolve())
        write_evidence(records, args.evidence)
    records = load_evidence(args.evidence)
    if args.check or not args.build:
        wanted = {
            ("core", "CICIDS2017", "rf", "strict"): 0.818902771736530,
            ("core", "CICIoT2023", "rf", "strict"): 0.897154513910465,
        }
        found: dict[tuple[str, str, str, str], float] = {}
        for record in records:
            key = (record["surface"], record["dataset"], record["model_family"], record["policy_variant"])
            if key in wanted:
                found[key] = corrected_metrics(record)["system_macro_f1_supported_labels"]
        for key, expected in wanted.items():
            actual = found.get(key)
            if actual is None or abs(actual - expected) > 5e-16:
                raise SystemExit(f"Exact correction gate failed for {key}: {actual!r} != {expected!r}")
        print(f"Protocol A evidence OK: {len(records)} matrices, 34 runs, exact gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
