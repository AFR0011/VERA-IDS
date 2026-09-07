#!/usr/bin/env python3
"""Preflight checks for the JISA Phase-1 controlled direct comparison.

This script does not train or evaluate models. It checks that the frozen
Protocol-A prepared partitions exist, records split schemas/row counts, and
verifies that both datasets expose the canonical target columns required by the
controlled direct comparison. Output is local-only under .release-audit/.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = REPO_ROOT / "processed_V5" / "A_stratified"
OUT = REPO_ROOT / ".release-audit" / "jisa_phase1_preflight"
DATASETS = ["CICIDS2017", "CICIoT2023"]
SPLITS = ["train", "val", "test"]
REQUIRED_TARGETS = {"y_stage1_attack", "y_stage2_family"}


def inspect_part(path: Path) -> tuple[list[str], int]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return list(pf.schema_arrow.names), int(pf.metadata.num_rows)
    if path.name.endswith(".csv.gz"):
        import pandas as pd
        df = pd.read_csv(path, nrows=5)
        return list(df.columns), -1
    raise RuntimeError(f"Unsupported prepared part: {path}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"processed_root": str(PROCESSED_ROOT), "datasets": {}, "ok": True}
    for dataset in DATASETS:
        ds = {"splits": {}, "schema_consistent": True}
        schema_ref = None
        for split in SPLITS:
            split_dir = PROCESSED_ROOT / dataset / split
            parts = sorted(split_dir.glob("*.parquet")) + sorted(split_dir.glob("*.csv.gz"))
            if not parts:
                ds["splits"][split] = {"exists": False}
                report["ok"] = False
                continue
            cols, first_rows = inspect_part(parts[0])
            if schema_ref is None:
                schema_ref = cols
            elif cols != schema_ref:
                ds["schema_consistent"] = False
                report["ok"] = False
            missing = sorted(REQUIRED_TARGETS.difference(cols))
            if missing:
                report["ok"] = False
            total_rows = None
            if parts[0].suffix.lower() == ".parquet":
                import pyarrow.parquet as pq
                total_rows = sum(int(pq.ParquetFile(p).metadata.num_rows) for p in parts)
            ds["splits"][split] = {
                "exists": True,
                "parts": len(parts),
                "rows": total_rows,
                "columns": cols,
                "missing_required_targets": missing,
            }
        report["datasets"][dataset] = ds
    (OUT / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("JISA Phase-1 preflight")
    print(f"processed root: {PROCESSED_ROOT}")
    for dataset, ds in report["datasets"].items():
        print(dataset)
        for split, info in ds["splits"].items():
            print(f"  {split}: exists={info.get('exists')} parts={info.get('parts')} rows={info.get('rows')} missing_targets={info.get('missing_required_targets')}")
        print(f"  schema_consistent={ds['schema_consistent']}")
    print(f"PASS={report['ok']}")
    print(f"audit: {OUT / 'preflight.json'}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
