#!/usr/bin/env python3
"""Derive historical C06 categorical records from the immutable summary only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ids_eval_framework.src.c06_feasibility import (  # noqa: E402
    SOURCE_RELATIVE, derive_c06, write_c06_outputs,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=ROOT / "outputs/17_revision_campaign/stability")
    p.add_argument("--check-only", action="store_true", help="Validate source and print derived counts without writing")
    args = p.parse_args()
    source = ROOT / SOURCE_RELATIVE
    result = derive_c06(source.read_bytes(), verify_frozen_blob=True)
    if not args.check_only:
        write_c06_outputs(result, args.out_root)
    print(
        f"[C06] frozen aggregates={result.evidence_manifest['historical_aggregate_rows']}; "
        f"holdout conditions={len(result.observations)}; "
        f"adjacent descriptive transitions={len(result.transitions)}; "
        f"scientific eligibility={result.evidence_manifest['eligibility_status']}; "
        f"written={not args.check_only}"
    )


if __name__ == "__main__":
    main()
