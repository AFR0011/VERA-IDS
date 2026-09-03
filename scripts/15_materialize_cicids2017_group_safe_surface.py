#!/usr/bin/env python3
"""Materialize the duplicate-group-safe CICIDS2017 journal sensitivity surface.

This script reuses the frozen Protocol-A processed CICIDS2017 population as the
materialized post-preprocessing source. Exact 84-feature representations are the
indivisible assignment unit: every row sharing the same processed-feature hash is
placed in exactly one of train/validation/test.

The deterministic assignment is the same feasibility policy used by script 14:
seeded 60/20/20 hash assignment followed by the minimum whole-group transfers
needed to satisfy >=200 rows per retained family in every split. Raw day/file
separation is intentionally not preserved on this fallback surface.

No frozen VERA-IDS outputs are modified. The default destination is a new local
JISA-finalization surface under outputs/12_jisa_finalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
TARGETS = {"y_stage1_attack", "y_stage2_family", "y_stage2_fine"}
EXPECTED_ROWS = 2_099_879
EXPECTED_GROUPS = 1_735_180


def parts(split_dir: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in ("*.parquet", "*.csv", "*.csv.gz"):
        out.extend(sorted(split_dir.glob(pattern)))
    return list(dict.fromkeys(out))


def iter_frames(path: Path, chunksize: int):
    if path.suffix.lower() == ".parquet":
        yield pd.read_parquet(path)
    else:
        yield from pd.read_csv(path, chunksize=chunksize)


def mix64(values: np.ndarray, seed: int) -> np.ndarray:
    x = values.astype(np.uint64, copy=True) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def build_assignment(source_root: Path, seed: int, min_support: int, chunksize: int):
    rows: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None
    source_files = 0

    for split in SPLITS:
        split_parts = parts(source_root / split)
        source_files += len(split_parts)
        for path in split_parts:
            for frame in iter_frames(path, chunksize):
                if feature_columns is None:
                    feature_columns = [c for c in frame.columns if c not in TARGETS]
                hashes = pd.util.hash_pandas_object(
                    frame[feature_columns], index=False
                ).to_numpy(dtype=np.uint64)
                rows.append(
                    pd.DataFrame(
                        {
                            "h": hashes,
                            "family": frame["y_stage2_family"].astype(str).to_numpy(),
                        }
                    )
                )

    if feature_columns is None:
        raise RuntimeError(f"No processed parts found under {source_root}")

    row_table = pd.concat(rows, ignore_index=True)
    unique_groups = int(row_table["h"].nunique())
    if len(row_table) != EXPECTED_ROWS or unique_groups != EXPECTED_GROUPS:
        raise RuntimeError(
            "Frozen processed surface mismatch: "
            f"rows={len(row_table)} (expected {EXPECTED_ROWS}), "
            f"groups={unique_groups} (expected {EXPECTED_GROUPS})"
        )

    meta = (
        row_table.groupby("h", sort=False)
        .agg(n=("family", "size"), family=("family", "first"), nfam=("family", "nunique"))
        .reset_index()
    )
    bucket = mix64(meta["h"].to_numpy(dtype=np.uint64), seed) % np.uint64(1_000_000)
    meta["split"] = np.where(
        bucket < 600_000,
        "train",
        np.where(bucket < 800_000, "val", "test"),
    )

    def recount() -> pd.DataFrame:
        mapping = pd.Series(meta["split"].to_numpy(), index=meta["h"].to_numpy())
        tmp = row_table.copy()
        tmp["split"] = tmp["h"].map(mapping)
        support = tmp.groupby(["family", "split"]).size().unstack(fill_value=0)
        for split in SPLITS:
            if split not in support:
                support[split] = 0
        return support[list(SPLITS)].astype(int)

    support = recount()
    moves: list[dict[str, object]] = []
    for family in support.index:
        for target in SPLITS:
            while int(support.loc[family, target]) < min_support:
                donors = sorted(
                    [
                        split
                        for split in SPLITS
                        if split != target and int(support.loc[family, split]) > min_support
                    ],
                    key=lambda split: int(support.loc[family, split]),
                    reverse=True,
                )
                moved = False
                for donor in donors:
                    candidates = meta[
                        (meta["nfam"] == 1)
                        & (meta["family"] == family)
                        & (meta["split"] == donor)
                    ].sort_values(["n", "h"])
                    for idx, row in candidates.iterrows():
                        n_rows = int(row["n"])
                        if int(support.loc[family, donor]) - n_rows < min_support:
                            continue
                        meta.at[idx, "split"] = target
                        support.loc[family, donor] -= n_rows
                        support.loc[family, target] += n_rows
                        moves.append(
                            {
                                "family": str(family),
                                "from": donor,
                                "to": target,
                                "rows": n_rows,
                                "group_family_count": int(row["nfam"]),
                            }
                        )
                        moved = True
                        break
                    if moved:
                        break
                if not moved:
                    raise RuntimeError(
                        f"Unable to repair support for family={family!r}, split={target!r}"
                    )

    support = recount()
    if not bool((support.min(axis=1) >= min_support).all()):
        raise RuntimeError("Support repair completed but minimum-support invariant failed")

    assignment = pd.Series(meta["split"].to_numpy(), index=meta["h"].to_numpy())
    return assignment, meta, support, moves, feature_columns, source_files


def materialize(
    source_root: Path,
    destination_dataset: Path,
    assignment: pd.Series,
    feature_columns: list[str],
    chunksize: int,
) -> dict[str, object]:
    part_index = {split: 0 for split in SPLITS}
    row_counts = {split: 0 for split in SPLITS}
    family_counts = {split: {} for split in SPLITS}
    split_hashes = {split: set() for split in SPLITS}
    all_columns: list[str] | None = None

    for split in SPLITS:
        (destination_dataset / split).mkdir(parents=True, exist_ok=True)

    for source_split in SPLITS:
        for path in parts(source_root / source_split):
            for frame in iter_frames(path, chunksize):
                if all_columns is None:
                    all_columns = list(frame.columns)
                hashes = pd.util.hash_pandas_object(
                    frame[feature_columns], index=False
                ).to_numpy(dtype=np.uint64)
                assigned = pd.Series(hashes).map(assignment).to_numpy()
                if pd.isna(assigned).any():
                    raise RuntimeError(f"Unassigned feature hash encountered while reading {path}")

                for target in SPLITS:
                    mask = assigned == target
                    if not bool(mask.any()):
                        continue
                    out_frame = frame.loc[mask].copy()
                    out_path = destination_dataset / target / f"part_{part_index[target]:05d}.csv.gz"
                    out_frame.to_csv(out_path, index=False, compression="gzip")
                    part_index[target] += 1
                    row_counts[target] += int(len(out_frame))
                    split_hashes[target].update(int(v) for v in hashes[mask])
                    counts = out_frame["y_stage2_family"].astype(str).value_counts()
                    for family, count in counts.items():
                        family_counts[target][str(family)] = int(
                            family_counts[target].get(str(family), 0) + int(count)
                        )

    overlaps = []
    zero_overlap = True
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        n_shared = len(split_hashes[left] & split_hashes[right])
        overlaps.append(
            {
                "split_left": left,
                "split_right": right,
                "shared_exact_feature_groups": int(n_shared),
            }
        )
        zero_overlap = zero_overlap and n_shared == 0

    if sum(row_counts.values()) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Materialized row-count mismatch: {sum(row_counts.values())} != {EXPECTED_ROWS}"
        )
    if not zero_overlap:
        raise RuntimeError(f"Cross-split exact-feature overlap detected: {overlaps}")

    return {
        "row_counts": row_counts,
        "family_counts": family_counts,
        "part_counts": part_index,
        "overlaps": overlaps,
        "zero_cross_split_exact_feature_overlap": bool(zero_overlap),
        "columns": all_columns or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", default="processed_V5")
    parser.add_argument(
        "--out-root",
        default="outputs/12_jisa_finalization/04_group_safe_surface",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-support", type=int, default=200)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the destination surface if it already exists.",
    )
    args = parser.parse_args()

    source_root = Path(args.processed_root) / "A_stratified" / "CICIDS2017"
    out_root = Path(args.out_root)
    destination = out_root / "B_group_safe" / "CICIDS2017"

    if destination.exists():
        if not args.force:
            raise RuntimeError(
                f"Destination already exists: {destination}. Use --force only to rebuild deliberately."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    print(f"[materialize] source:      {source_root}")
    print(f"[materialize] destination: {destination}")
    print("[materialize] rebuilding deterministic exact-feature-group assignment ...")
    assignment, meta, support, moves, feature_columns, source_files = build_assignment(
        source_root,
        seed=int(args.seed),
        min_support=int(args.min_support),
        chunksize=int(args.chunksize),
    )

    print(
        f"[materialize] assignment: {len(meta):,} groups; "
        f"{len(moves)} repair move(s), {sum(int(m['rows']) for m in moves)} row(s)"
    )
    print("[materialize] writing new split surface ...")
    result = materialize(
        source_root,
        destination,
        assignment,
        feature_columns,
        int(args.chunksize),
    )

    support_out = support.reset_index().rename(columns={"family": "family"})
    support_out["min_support"] = support_out[list(SPLITS)].min(axis=1)
    support_out.to_csv(out_root / "materialized_family_support.csv", index=False)
    pd.DataFrame(moves).to_csv(out_root / "materialized_support_repair_moves.csv", index=False)
    pd.DataFrame(result["overlaps"]).to_csv(
        out_root / "materialized_cross_split_duplicate_audit.csv", index=False
    )
    (out_root / "feature_columns.txt").write_text(
        "\n".join(feature_columns) + "\n", encoding="utf-8"
    )

    assignment_counts = meta.groupby("split")["n"].sum().to_dict()
    summary = {
        "dataset": "CICIDS2017",
        "surface_name": "B_group_safe",
        "source_surface": "processed_V5/A_stratified/CICIDS2017 union",
        "source_part_files": int(source_files),
        "seed": int(args.seed),
        "minimum_family_support": int(args.min_support),
        "feature_columns": int(len(feature_columns)),
        "rows": int(sum(result["row_counts"].values())),
        "unique_exact_feature_groups": int(len(meta)),
        "family_conflict_groups": int((meta["nfam"] > 1).sum()),
        "repair_moves": int(len(moves)),
        "repair_rows": int(sum(int(m["rows"]) for m in moves),),
        "split_rows": {split: int(result["row_counts"][split]) for split in SPLITS},
        "split_fractions": {
            split: float(result["row_counts"][split] / EXPECTED_ROWS) for split in SPLITS
        },
        "assignment_row_totals": {split: int(assignment_counts.get(split, 0)) for split in SPLITS},
        "zero_cross_split_exact_feature_overlap": bool(
            result["zero_cross_split_exact_feature_overlap"]
        ),
        "all_families_supported": bool((support.min(axis=1) >= args.min_support).all()),
        "provenance_boundary": (
            "Raw day/file separation is not preserved. The indivisible partition unit is "
            "the exact post-preprocessing feature representation."
        ),
        "interpretation": (
            "Journal sensitivity surface for testing whether CICIDS2017 held-out-family "
            "conclusions persist after eliminating exact processed-feature overlap."
        ),
    }

    # Simple deterministic fingerprint of the split policy and aggregate outcome.
    fingerprint_payload = json.dumps(summary, sort_keys=True).encode("utf-8")
    summary["aggregate_surface_fingerprint_sha256"] = hashlib.sha256(
        fingerprint_payload
    ).hexdigest()

    (out_root / "group_safe_surface_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (destination / "SPLIT_PROTOCOL.json").write_text(
        json.dumps(
            {
                "protocol": "B_group_safe",
                "seed": int(args.seed),
                "target_split_fractions": {"train": 0.60, "val": 0.20, "test": 0.20},
                "group_definition": "exact hash across all 84 processed predictor columns",
                "assignment_unit": "exact processed-feature group",
                "minimum_family_support_per_split": int(args.min_support),
                "support_repair_policy": (
                    "minimum deterministic whole-group transfers, restricted to single-family groups"
                ),
                "raw_day_file_separation_preserved": False,
                "unknown_detection_interpretation": (
                    "supported held-out-family evaluation; not temporal or source-day generalization"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (destination / "USED_COLUMNS.json").write_text(
        json.dumps({"columns": result["columns"]}, indent=2), encoding="utf-8"
    )

    print("[materialize] complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
