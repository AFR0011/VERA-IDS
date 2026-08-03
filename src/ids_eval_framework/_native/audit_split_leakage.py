#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CFG = {
    "dataset_root": Path("processed_V5_cicids17_recovery/B_day_file/CICIDS2017"),
    "out_root": Path("q2_reproducibility_audit"),
    "chunksize": 100_000,
}

TARGET_COLUMNS = {"y_stage1_attack", "y_stage2_family", "y_stage2_fine"}


def part_paths(split_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.parquet", "*.csv", "*.csv.gz"):
        paths.extend(sorted(split_dir.glob(pattern)))
    return paths


def iter_frames(path: Path):
    if path.suffix == ".parquet":
        yield pd.read_parquet(path)
        return
    yield from pd.read_csv(path, chunksize=int(CFG["chunksize"]))


def hash_split(split: str) -> tuple[set[int], set[int], int, list[str]]:
    full_hashes: set[int] = set()
    feature_hashes: set[int] = set()
    total_rows = 0
    feature_columns: list[str] = []
    for path in part_paths(CFG["dataset_root"] / split):
        for frame in iter_frames(path):
            if not feature_columns:
                feature_columns = [c for c in frame.columns if c not in TARGET_COLUMNS]
            total_rows += len(frame)
            full_hashes.update(
                int(v) for v in pd.util.hash_pandas_object(frame, index=False).to_numpy()
            )
            feature_hashes.update(
                int(v)
                for v in pd.util.hash_pandas_object(frame[feature_columns], index=False).to_numpy()
            )
    return full_hashes, feature_hashes, total_rows, feature_columns


def main() -> None:
    out_root = CFG["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    split_rows = []
    for split in ("train", "val", "test"):
        full, features, rows, feature_columns = hash_split(split)
        split_hashes[split] = {"full": full, "features": features}
        split_rows.append(
            {
                "split": split,
                "rows": rows,
                "unique_full_rows": len(full),
                "unique_feature_rows": len(features),
                "feature_columns": len(feature_columns),
            }
        )

    overlap_rows = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap_rows.append(
            {
                "split_left": left,
                "split_right": right,
                "exact_full_row_hash_overlap": len(
                    split_hashes[left]["full"] & split_hashes[right]["full"]
                ),
                "exact_feature_row_hash_overlap": len(
                    split_hashes[left]["features"] & split_hashes[right]["features"]
                ),
            }
        )

    pd.DataFrame(split_rows).to_csv(out_root / "split_hash_summary.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(out_root / "cross_split_exact_duplicate_audit.csv", index=False)

    unit_manifest = pd.read_csv(CFG["dataset_root"] / "protocol_b_unit_manifest.csv")
    with (CFG["dataset_root"] / "protocol_b_file_manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        assignments = json.load(handle)
    assigned_split = {
        str(unit_id): split
        for split, unit_ids in assignments.items()
        for unit_id in unit_ids
    }
    unit_manifest["split"] = unit_manifest["unit_id"].map(assigned_split)
    interval_rows = []
    for source_file, group in unit_manifest.groupby("source_file_name", sort=True):
        ordered = group.sort_values("row_start")
        starts = ordered["row_start"].astype(int).to_numpy()
        ends = ordered["row_end"].astype(int).to_numpy()
        interval_rows.append(
            {
                "source_file_name": source_file,
                "units": len(ordered),
                "all_units_assigned_once": bool(ordered["split"].notna().all()),
                "non_overlapping_intervals": bool(
                    len(ordered) <= 1 or (starts[1:] >= ends[:-1]).all()
                ),
                "contiguous_full_coverage": bool(
                    starts[0] == 0
                    and (len(ordered) <= 1 or (starts[1:] == ends[:-1]).all())
                    and ends[-1] == int(ordered["raw_row_count"].astype(int).sum())
                ),
                "raw_rows_covered": int((ends - starts).sum()),
                "splits_used": "|".join(sorted(ordered["split"].dropna().unique())),
            }
        )
    pd.DataFrame(interval_rows).to_csv(out_root / "raw_row_interval_audit.csv", index=False)
    unit_manifest.to_csv(out_root / "unit_split_assignment_audit.csv", index=False)
    with (out_root / "audit_scope.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": "pandas 64-bit deterministic row hashing",
                "full_row_definition": "all processed predictor and target columns",
                "feature_row_definition": "all processed columns except three target columns",
                "near_duplicate_search_performed": False,
                "raw_row_interval_overlap_checked": True,
                "interpretation_boundary": (
                    "Planning units are disjoint raw row intervals. Hash overlap detects repeated "
                    "processed representations, not repeated raw row identities, and does not detect "
                    "approximate near-duplicates."
                ),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
