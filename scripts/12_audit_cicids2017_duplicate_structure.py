#!/usr/bin/env python3
"""Audit exact processed-feature duplicate structure for CICIDS2017.

This is the first JISA-finalization diagnostic. It is intentionally read-only with
respect to all frozen VERA-IDS result surfaces. The script uses the union of the
existing Protocol-A processed splits as a convenient materialized copy of the
post-preprocessing CICIDS2017 representation and characterizes exact repeated
feature vectors globally and across the current train/validation/test assignment.

It does *not* construct a new Protocol-B split and does *not* claim that Protocol-A
cross-split duplicates are themselves the journal Protocol-B leakage result.

Outputs are aggregate/group-level audit artifacts only and should remain under a
local ignored output root; no benchmark rows are written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
TARGET_COLUMNS = {"y_stage1_attack", "y_stage2_family", "y_stage2_fine"}
FAMILY_COLUMN = "y_stage2_family"
FINE_COLUMN = "y_stage2_fine"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Characterize exact duplicate processed feature vectors in the frozen "
            "CICIDS2017 Protocol-A surface before designing a duplicate-group-safe "
            "Protocol-B split."
        )
    )
    parser.add_argument(
        "--processed-root",
        default="processed_V5",
        help=(
            "Root containing A_stratified/CICIDS2017. The user-local default is "
            "processed_V5 at repository root."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="CICIDS2017",
        help="Dataset directory name under A_stratified (default: CICIDS2017).",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/12_jisa_finalization/01_duplicate_structure_audit",
        help="Local aggregate audit output directory.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for CSV/CSV.GZ inputs. Parquet parts are read one part at a time.",
    )
    return parser


def part_paths(split_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.parquet", "*.csv", "*.csv.gz"):
        paths.extend(sorted(split_dir.glob(pattern)))
    # De-duplicate in case a glob implementation/path pattern ever overlaps.
    return list(dict.fromkeys(paths))


def iter_frames(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() == ".parquet":
        yield pd.read_parquet(path)
        return
    yield from pd.read_csv(path, chunksize=chunksize)


def pct(num: int | float, den: int | float) -> float:
    return float(num) / float(den) if den else float("nan")


def unique_join(values: pd.Series) -> str:
    cleaned = sorted({str(v) for v in values.dropna().astype(str)})
    return "|".join(cleaned)


def main() -> None:
    args = build_parser().parse_args()
    processed_root = Path(args.processed_root)
    dataset_root = processed_root / "A_stratified" / args.dataset
    out_root = Path(args.out_root)

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Expected processed dataset directory not found: {dataset_root}\n"
            "Pass --processed-root if processed_V5 is stored elsewhere."
        )

    out_root.mkdir(parents=True, exist_ok=True)

    row_chunks: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None
    source_files = 0
    split_row_counts: dict[str, int] = {split: 0 for split in SPLITS}

    print(f"[audit] dataset root: {dataset_root}")
    print(f"[audit] output root:  {out_root}")

    for split in SPLITS:
        split_dir = dataset_root / split
        paths = part_paths(split_dir)
        if not paths:
            raise FileNotFoundError(f"No processed parts found under {split_dir}")

        print(f"[audit] {split}: {len(paths)} part file(s)")
        for path in paths:
            source_files += 1
            for frame in iter_frames(path, int(args.chunksize)):
                if FAMILY_COLUMN not in frame.columns:
                    raise ValueError(
                        f"Required target column {FAMILY_COLUMN!r} missing from {path}"
                    )

                current_features = [c for c in frame.columns if c not in TARGET_COLUMNS]
                if feature_columns is None:
                    feature_columns = current_features
                elif current_features != feature_columns:
                    missing = sorted(set(feature_columns) - set(current_features))
                    added = sorted(set(current_features) - set(feature_columns))
                    raise ValueError(
                        "Processed feature schema changed across parts. "
                        f"File={path}; missing={missing[:10]}; added={added[:10]}"
                    )

                feature_hash = pd.util.hash_pandas_object(
                    frame[feature_columns], index=False
                ).to_numpy(dtype="uint64", copy=False)

                compact = pd.DataFrame(
                    {
                        "feature_hash": feature_hash,
                        "split": split,
                        "family": frame[FAMILY_COLUMN].astype("string").to_numpy(),
                    }
                )
                if FINE_COLUMN in frame.columns:
                    compact["fine"] = frame[FINE_COLUMN].astype("string").to_numpy()

                row_chunks.append(compact)
                split_row_counts[split] += int(len(compact))

        print(f"[audit] {split}: {split_row_counts[split]:,} rows scanned")

    if feature_columns is None or not row_chunks:
        raise RuntimeError("No rows were scanned.")

    rows = pd.concat(row_chunks, ignore_index=True)
    del row_chunks
    rows["split"] = pd.Categorical(rows["split"], categories=list(SPLITS))
    rows["family"] = rows["family"].astype("category")
    if "fine" in rows.columns:
        rows["fine"] = rows["fine"].astype("category")

    print(f"[audit] total rows: {len(rows):,}")
    print(f"[audit] feature columns: {len(feature_columns):,}")
    print("[audit] aggregating exact feature-hash groups ...")

    counts_by_split = (
        rows.groupby(["feature_hash", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(SPLITS), fill_value=0)
        .astype("int64")
    )
    counts_by_split.columns = [f"{c}_rows" for c in counts_by_split.columns]

    group_stats = counts_by_split.copy()
    group_stats["total_rows"] = group_stats.sum(axis=1)
    group_stats["n_splits"] = (
        group_stats[[f"{s}_rows" for s in SPLITS]].gt(0).sum(axis=1).astype("int8")
    )
    group_stats["family_nunique"] = (
        rows.groupby("feature_hash", observed=True)["family"].nunique(dropna=False)
    )
    if "fine" in rows.columns:
        group_stats["fine_nunique"] = (
            rows.groupby("feature_hash", observed=True)["fine"].nunique(dropna=False)
        )
    else:
        group_stats["fine_nunique"] = np.nan

    repeated_mask = group_stats["total_rows"] > 1
    cross_split_mask = group_stats["n_splits"] > 1
    family_conflict_mask = group_stats["family_nunique"] > 1
    fine_conflict_mask = group_stats["fine_nunique"].fillna(1) > 1

    repeated_hashes = group_stats.index[repeated_mask]
    cross_split_hashes = group_stats.index[cross_split_mask]

    # Mark rows for family-level summaries. These are local in-memory annotations only.
    rows["in_repeated_group"] = rows["feature_hash"].isin(repeated_hashes)
    rows["in_cross_split_group"] = rows["feature_hash"].isin(cross_split_hashes)

    total_rows = int(len(rows))
    unique_hashes = int(len(group_stats))
    repeated_groups = int(repeated_mask.sum())
    rows_in_repeated_groups = int(group_stats.loc[repeated_mask, "total_rows"].sum())
    cross_split_groups = int(cross_split_mask.sum())
    rows_in_cross_split_groups = int(
        group_stats.loc[cross_split_mask, "total_rows"].sum()
    )
    family_conflict_groups = int(family_conflict_mask.sum())
    family_conflict_rows = int(
        group_stats.loc[family_conflict_mask, "total_rows"].sum()
    )
    fine_conflict_groups = int(fine_conflict_mask.sum())
    fine_conflict_rows = int(group_stats.loc[fine_conflict_mask, "total_rows"].sum())

    summary = {
        "dataset": args.dataset,
        "source_surface": "A_stratified union of train/val/test",
        "source_root": str(dataset_root),
        "source_part_files": int(source_files),
        "feature_columns": int(len(feature_columns)),
        "total_rows": total_rows,
        "unique_feature_hashes": unique_hashes,
        "redundant_rows_beyond_first_occurrence": int(total_rows - unique_hashes),
        "redundant_rows_fraction": pct(total_rows - unique_hashes, total_rows),
        "repeated_feature_groups": repeated_groups,
        "rows_in_repeated_feature_groups": rows_in_repeated_groups,
        "rows_in_repeated_feature_groups_fraction": pct(rows_in_repeated_groups, total_rows),
        "largest_exact_feature_group_rows": int(group_stats["total_rows"].max()),
        "cross_split_feature_groups_in_current_protocol_a": cross_split_groups,
        "rows_in_cross_split_feature_groups_in_current_protocol_a": rows_in_cross_split_groups,
        "rows_in_cross_split_feature_groups_fraction": pct(rows_in_cross_split_groups, total_rows),
        "family_conflict_feature_groups": family_conflict_groups,
        "rows_in_family_conflict_feature_groups": family_conflict_rows,
        "fine_label_conflict_feature_groups": fine_conflict_groups,
        "rows_in_fine_label_conflict_feature_groups": fine_conflict_rows,
        "split_rows": split_row_counts,
    }

    with (out_root / "duplicate_structure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    size_distribution = (
        group_stats["total_rows"]
        .value_counts()
        .sort_index()
        .rename_axis("group_size_rows")
        .reset_index(name="feature_groups")
    )
    size_distribution["rows_represented"] = (
        size_distribution["group_size_rows"] * size_distribution["feature_groups"]
    )
    size_distribution.to_csv(
        out_root / "duplicate_group_size_distribution.csv", index=False
    )

    pair_rows: list[dict[str, int | float | str]] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_col = f"{left}_rows"
        right_col = f"{right}_rows"
        shared = group_stats[left_col].gt(0) & group_stats[right_col].gt(0)
        left_unique = int(group_stats[left_col].gt(0).sum())
        right_unique = int(group_stats[right_col].gt(0).sum())
        shared_groups = int(shared.sum())
        left_rows_shared = int(group_stats.loc[shared, left_col].sum())
        right_rows_shared = int(group_stats.loc[shared, right_col].sum())
        pair_rows.append(
            {
                "split_left": left,
                "split_right": right,
                "shared_exact_feature_groups": shared_groups,
                "shared_fraction_of_left_unique_groups": pct(shared_groups, left_unique),
                "shared_fraction_of_right_unique_groups": pct(shared_groups, right_unique),
                "left_rows_belonging_to_shared_groups": left_rows_shared,
                "right_rows_belonging_to_shared_groups": right_rows_shared,
                "left_row_fraction_in_shared_groups": pct(
                    left_rows_shared, split_row_counts[left]
                ),
                "right_row_fraction_in_shared_groups": pct(
                    right_rows_shared, split_row_counts[right]
                ),
            }
        )
    pd.DataFrame(pair_rows).to_csv(
        out_root / "current_protocol_a_cross_split_duplicate_summary.csv", index=False
    )

    family_rows: list[dict[str, int | float | str]] = []
    for family, frame in rows.groupby("family", observed=True):
        total = int(len(frame))
        repeated_rows = int(frame["in_repeated_group"].sum())
        cross_rows = int(frame["in_cross_split_group"].sum())
        family_rows.append(
            {
                "family": str(family),
                "total_rows": total,
                "unique_feature_hashes": int(frame["feature_hash"].nunique()),
                "rows_in_repeated_groups": repeated_rows,
                "repeated_row_fraction": pct(repeated_rows, total),
                "rows_in_cross_split_groups": cross_rows,
                "cross_split_group_row_fraction": pct(cross_rows, total),
            }
        )
    pd.DataFrame(family_rows).sort_values("family").to_csv(
        out_root / "family_duplicate_summary.csv", index=False
    )

    repeated_group_table = group_stats.loc[repeated_mask].copy().reset_index()
    repeated_group_table["feature_hash_hex"] = repeated_group_table["feature_hash"].map(
        lambda value: f"{int(value):016x}"
    )
    repeated_group_table = repeated_group_table.drop(columns=["feature_hash"])
    repeated_group_table.sort_values(
        ["total_rows", "n_splits"], ascending=[False, False]
    ).to_csv(out_root / "repeated_feature_groups.csv", index=False)

    if family_conflict_groups or fine_conflict_groups:
        conflict_hashes = group_stats.index[family_conflict_mask | fine_conflict_mask]
        conflict_rows = rows.loc[rows["feature_hash"].isin(conflict_hashes)].copy()
        aggregation: dict[str, tuple[str, object]] = {
            "rows": ("feature_hash", "size"),
            "splits": ("split", unique_join),
            "families": ("family", unique_join),
        }
        if "fine" in conflict_rows.columns:
            aggregation["fine_labels"] = ("fine", unique_join)
        conflicts = (
            conflict_rows.groupby("feature_hash", observed=True)
            .agg(**aggregation)
            .reset_index()
        )
        conflicts["feature_hash_hex"] = conflicts["feature_hash"].map(
            lambda value: f"{int(value):016x}"
        )
        conflicts.drop(columns=["feature_hash"]).to_csv(
            out_root / "label_conflicting_feature_groups.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=["rows", "splits", "families", "fine_labels", "feature_hash_hex"]
        ).to_csv(out_root / "label_conflicting_feature_groups.csv", index=False)

    (out_root / "feature_columns.txt").write_text(
        "\n".join(feature_columns) + "\n", encoding="utf-8"
    )

    with (out_root / "audit_scope.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "purpose": (
                    "Pre-split feasibility characterization for a future duplicate-group-safe "
                    "CICIDS2017 Protocol-B reconstruction."
                ),
                "input_interpretation": (
                    "The frozen Protocol-A train/validation/test union is used only as a "
                    "materialized copy of the study's post-preprocessing representation."
                ),
                "hash_method": "pandas deterministic 64-bit hash_pandas_object over feature columns",
                "feature_row_definition": (
                    "All processed columns except y_stage1_attack, y_stage2_family, and "
                    "y_stage2_fine."
                ),
                "collision_boundary": (
                    "The audit treats equal 64-bit feature hashes as exact-equality candidates. "
                    "The collision probability is negligible for this feasibility screen but "
                    "a final split-proof stage should independently verify shared/grouped rows "
                    "before a zero-overlap journal claim is made."
                ),
                "near_duplicate_search_performed": False,
                "new_split_constructed": False,
                "models_trained": False,
                "frozen_outputs_modified": False,
                "protocol_a_cross_split_interpretation": (
                    "Cross-split sharing reported here characterizes the existing row-stratified "
                    "Protocol-A assignment. It is not the recovered Protocol-B overlap result."
                ),
            },
            handle,
            indent=2,
        )

    print("[audit] complete")
    print(f"[audit] unique feature groups: {unique_hashes:,}")
    print(
        "[audit] rows in repeated groups: "
        f"{rows_in_repeated_groups:,} ({pct(rows_in_repeated_groups, total_rows):.2%})"
    )
    print(
        "[audit] current Protocol-A rows in cross-split groups: "
        f"{rows_in_cross_split_groups:,} ({pct(rows_in_cross_split_groups, total_rows):.2%})"
    )
    print(
        f"[audit] family-conflicting groups: {family_conflict_groups:,}; "
        f"fine-label-conflicting groups: {fine_conflict_groups:,}"
    )
    print(f"[audit] wrote aggregate artifacts to {out_root}")


if __name__ == "__main__":
    main()
