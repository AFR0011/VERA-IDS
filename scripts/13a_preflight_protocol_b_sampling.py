#!/usr/bin/env python3
"""Inspect Protocol-B sampled Stage-1 class support before expensive model fits."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework._native.protocol_b_grid import collect_split_frame  # noqa: E402


def parse_processed(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit("--processed must use DATASET=PATH")
        dataset, path = value.split("=", 1)
        out[dataset.strip()] = str(Path(path.strip()).resolve())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight Protocol-B Stage-1 sampled class counts."
    )
    parser.add_argument("--audit-root", action="append", required=True)
    parser.add_argument("--processed", action="append", required=True, metavar="DATASET=PATH")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--smoke-cap", type=int, default=30000)
    parser.add_argument("--full-cap", type=int, default=1500000)
    parser.add_argument("--max-manifests", type=int, default=1)
    args = parser.parse_args()

    processed = parse_processed(args.processed)
    manifests: list[Path] = []
    for root in args.audit_root:
        manifests.extend(Path(p) for p in glob.glob(str(Path(root) / "*" / "manifests" / "*.json")))
    manifests = sorted(dict.fromkeys(manifests))
    if args.max_manifests > 0:
        manifests = manifests[: args.max_manifests]
    if not manifests:
        raise SystemExit("No manifests found under supplied audit roots.")

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset = str(manifest["dataset"])
        if dataset not in processed:
            print(f"[skip] {manifest_path.name}: no --processed override for {dataset}")
            continue
        dataset_dir = processed[dataset]
        y1_col = str(manifest["y_stage1_col"])
        y2_col = str(manifest["y_stage2_col"])
        holdout = str(manifest["holdout_family"])

        print(f"\n=== {dataset} holdout={holdout} ===")
        print(f"manifest={manifest_path}")
        print(f"processed={dataset_dir}")

        for label, cap in [("smoke", args.smoke_cap), ("full", args.full_cap)]:
            frame = collect_split_frame(
                dataset_dir,
                "train",
                [y1_col, y2_col],
                int(cap),
                seed=int(args.seed),
            )
            y1 = frame[y1_col].astype(int).to_numpy()
            y2 = frame[y2_col].astype(str).fillna("").to_numpy(dtype=object)
            raw_unique, raw_counts = np.unique(y1, return_counts=True)
            raw = {int(k): int(v) for k, v in zip(raw_unique, raw_counts)}

            keep = ~((y1 == 1) & (y2 == holdout))
            y1_loao = y1[keep]
            loao_unique, loao_counts = np.unique(y1_loao, return_counts=True)
            loao = {int(k): int(v) for k, v in zip(loao_unique, loao_counts)}

            print(
                f"{label:>5} cap={cap:,} sampled_rows={len(frame):,} "
                f"raw_stage1={raw} post_loao_stage1={loao}"
            )


if __name__ == "__main__":
    main()
