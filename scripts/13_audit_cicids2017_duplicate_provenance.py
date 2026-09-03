#!/usr/bin/env python3
"""Audit provenance connectivity induced by exact processed-feature duplicates.

JISA finalization Phase 1B diagnostic.

The existing CICIDS2017 Protocol-B recovery exposes each raw source file as four
contiguous planning units. A duplicate-group-safe split can preserve that design
only if units connected by identical post-preprocessing feature representations
can be assigned together while retaining adequate family support.

This script does not choose a train/validation/test split. It replays the frozen
CICIDS2017 preprocessing transformation on the raw base CSV files, computes exact
feature hashes, constructs a graph linking contiguous units that share at least
one exact feature representation, and summarizes the resulting connected
components and their family support.

No row-level benchmark data or hashes are written. Outputs are aggregate
component/unit diagnostics under the JISA-finalization output root.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ids_eval_framework._native.prepare_datasets import (
    canonical_col,
    canonicalize_columns,
    detect_leakage_cols,
    get_mapper,
    port_bucket,
    protocol_b_partition_specs,
    read_csv_slice,
    resolve_label_col_for_file,
)

TARGET_COLUMNS = {"y_stage1_attack", "y_stage2_family", "y_stage2_fine"}
PORT_COLUMNS = (
    "src_port",
    "dst_port",
    "source_port",
    "destination_port",
    "sport",
    "dsport",
    "srcport",
    "dstport",
)


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Replay CICIDS2017 preprocessing on raw source files and measure whether "
            "exact duplicate groups connect the four-unit provenance partition."
        )
    )
    p.add_argument(
        "--dataset-root",
        default="Datasets/CICIDS 2017",
        help="Raw CICIDS2017 directory (default: Datasets/CICIDS 2017).",
    )
    p.add_argument(
        "--out-root",
        default="outputs/12_jisa_finalization/02_duplicate_provenance_audit",
        help="Aggregate diagnostic output directory.",
    )
    p.add_argument(
        "--subfiles-per-file",
        type=int,
        default=4,
        help="Contiguous units per source file, matching the historical recovery design.",
    )
    return p


def list_base_csvs(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.glob("*.csv")):
        name = p.name.lower()
        if "features" in name or "_plus" in name:
            continue
        files.append(p)
    return files


def preprocess_unit(path: Path, row_start: int, row_end: int) -> pd.DataFrame:
    ds_cfg = {
        "label_col": "Label",
        "has_header": True,
        "drop_cols": [],
        "drop_families": ["Other"],
        "protocol_b_partition_mode": "contiguous_subfiles",
        "protocol_b_subfiles_per_file": 4,
    }
    label_actual = resolve_label_col_for_file(str(path), ds_cfg["label_col"], ds_cfg)
    if label_actual is None:
        raise RuntimeError(f"Could not resolve label column in {path}")
    label_col = canonical_col(label_actual)

    chunk = read_csv_slice(
        str(path),
        ds_cfg,
        start_row=int(row_start),
        nrows=int(row_end - row_start),
        usecols=None,
    )
    chunk = canonicalize_columns(chunk)
    if label_col not in chunk.columns:
        raise RuntimeError(f"Label column {label_col!r} missing from {path}")

    leak = detect_leakage_cols(list(chunk.columns))
    for raw in PORT_COLUMNS:
        if raw in chunk.columns:
            chunk[f"{raw}_bucket"] = port_bucket(chunk[raw])
    leak = [c for c in leak if not c.endswith("_bucket")]
    if leak:
        chunk = chunk.drop(columns=[c for c in leak if c in chunk.columns])

    chunk = chunk.replace([np.inf, -np.inf], np.nan)

    mapper = get_mapper("CICIDS2017")
    fine = chunk[label_col].astype(str).str.strip()
    family = fine.map(mapper).astype(str)
    keep = family != "Other"
    chunk = chunk.loc[keep].copy()
    fine = fine.loc[keep]
    family = family.loc[keep]

    chunk["y_stage1_attack"] = (family != "Benign").astype(np.int8)
    chunk["y_stage2_family"] = family
    chunk["y_stage2_fine"] = fine
    chunk = chunk.drop(columns=[label_col])
    return chunk


def popcount(x: int) -> int:
    return int(x.bit_count())


def main() -> None:
    args = build_parser().parse_args()
    raw_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise SystemExit(f"Raw dataset root does not exist: {raw_root}")

    files = list_base_csvs(raw_root)
    if not files:
        raise SystemExit(
            f"No CICIDS2017 base CSV files found under {raw_root}. "
            "Expected *.csv while excluding *_plus.csv and feature-list files."
        )

    print(f"[provenance] raw root: {raw_root}")
    print(f"[provenance] base CSV files: {len(files)}")

    ds_cfg_specs = {
        "protocol_b_partition_mode": "contiguous_subfiles",
        "protocol_b_subfiles_per_file": int(args.subfiles_per_file),
    }

    unit_specs: list[dict] = []
    for file_idx, path in enumerate(files):
        for spec in protocol_b_partition_specs(
            str(path), ds_cfg_specs, partition_mode="contiguous_subfiles"
        ):
            spec = dict(spec)
            spec["file_index"] = int(file_idx)
            spec["unit_index"] = int(len(unit_specs))
            unit_specs.append(spec)

    print(f"[provenance] contiguous units: {len(unit_specs)}")

    # hash -> bitmask of units in which this exact processed feature vector occurs.
    hash_masks: dict[int, int] = {}
    hash_families: dict[int, str] = {}
    family_conflict_hashes: set[int] = set()

    unit_rows: list[int] = [0] * len(unit_specs)
    unit_family_counts: list[Counter] = [Counter() for _ in unit_specs]
    unit_unique_hashes: list[int] = [0] * len(unit_specs)
    feature_columns_ref: list[str] | None = None
    total_rows = 0

    for spec in unit_specs:
        ui = int(spec["unit_index"])
        path = Path(spec["source_file_path"])
        print(
            f"[provenance] unit {ui + 1:02d}/{len(unit_specs)}: "
            f"{path.name} part {spec['segment_index']}/{spec['segment_count']}"
        )
        frame = preprocess_unit(path, int(spec["row_start"]), int(spec["row_end"]))
        feature_columns = [c for c in frame.columns if c not in TARGET_COLUMNS]
        if feature_columns_ref is None:
            feature_columns_ref = feature_columns
        elif feature_columns != feature_columns_ref:
            raise RuntimeError(
                f"Processed feature schema mismatch in unit {spec['unit_id']}. "
                "The provenance audit requires one frozen feature definition."
            )

        hashes = pd.util.hash_pandas_object(frame[feature_columns], index=False).to_numpy(
            dtype=np.uint64, copy=False
        )
        families = frame["y_stage2_family"].astype(str).to_numpy()
        unique_hashes = np.unique(hashes)

        unit_rows[ui] = int(len(frame))
        total_rows += int(len(frame))
        unit_unique_hashes[ui] = int(len(unique_hashes))
        unit_family_counts[ui].update(families.tolist())

        mask_bit = 1 << ui
        for h in unique_hashes.tolist():
            hv = int(h)
            hash_masks[hv] = int(hash_masks.get(hv, 0) | mask_bit)

        # Family conflicts are checked on the full rows, not only unique hashes.
        for hv, fam in zip(hashes.tolist(), families.tolist()):
            hvi = int(hv)
            prev = hash_families.get(hvi)
            if prev is None:
                hash_families[hvi] = str(fam)
            elif prev != str(fam):
                family_conflict_hashes.add(hvi)

    print("[provenance] building duplicate-induced unit graph ...")
    dsu = DSU(len(unit_specs))
    cross_unit_groups = 0
    cross_source_groups = 0
    unit_pair_shared_groups: Counter = Counter()
    units_by_file = [int(s["file_index"]) for s in unit_specs]

    for mask in hash_masks.values():
        if popcount(mask) <= 1:
            continue
        cross_unit_groups += 1
        units = [i for i in range(len(unit_specs)) if mask & (1 << i)]
        for u in units[1:]:
            dsu.union(units[0], u)
        srcs = {units_by_file[u] for u in units}
        if len(srcs) > 1:
            cross_source_groups += 1
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                unit_pair_shared_groups[(units[i], units[j])] += 1

    components: dict[int, list[int]] = defaultdict(list)
    for ui in range(len(unit_specs)):
        components[dsu.find(ui)].append(ui)

    # Stable component numbering by earliest unit index.
    ordered_components = sorted(components.values(), key=lambda xs: min(xs))
    unit_to_component: dict[int, int] = {}
    component_rows = []
    all_families = ["Benign", "Botnet", "BruteForce", "DDoS", "DoS", "Scan/Recon", "Web/App"]

    for cid, units in enumerate(ordered_components):
        for ui in units:
            unit_to_component[ui] = cid
        fam_counts = Counter()
        for ui in units:
            fam_counts.update(unit_family_counts[ui])
        src_files = sorted({unit_specs[ui]["source_file_name"] for ui in units})
        component_rows.append(
            {
                "component_id": cid,
                "unit_count": len(units),
                "source_file_count": len(src_files),
                "processed_rows": int(sum(unit_rows[ui] for ui in units)),
                "source_files": "|".join(src_files),
                "units": "|".join(unit_specs[ui]["unit_id"] for ui in units),
                **{f"family_{fam}": int(fam_counts.get(fam, 0)) for fam in all_families},
            }
        )

    unit_rows_out = []
    for spec in unit_specs:
        ui = int(spec["unit_index"])
        unit_rows_out.append(
            {
                "unit_index": ui,
                "component_id": int(unit_to_component[ui]),
                "unit_id": spec["unit_id"],
                "source_file_name": spec["source_file_name"],
                "segment_index": int(spec["segment_index"]),
                "segment_count": int(spec["segment_count"]),
                "row_start": int(spec["row_start"]),
                "row_end": int(spec["row_end"]),
                "processed_rows": int(unit_rows[ui]),
                "unique_feature_hashes": int(unit_unique_hashes[ui]),
                **{
                    f"family_{fam}": int(unit_family_counts[ui].get(fam, 0))
                    for fam in all_families
                },
            }
        )

    edge_rows = []
    for (left, right), n_shared in unit_pair_shared_groups.items():
        edge_rows.append(
            {
                "left_unit_index": left,
                "right_unit_index": right,
                "left_unit": unit_specs[left]["unit_id"],
                "right_unit": unit_specs[right]["unit_id"],
                "same_source_file": bool(
                    unit_specs[left]["source_file_name"] == unit_specs[right]["source_file_name"]
                ),
                "shared_exact_feature_groups": int(n_shared),
            }
        )

    pd.DataFrame(unit_rows_out).to_csv(out_root / "provenance_units.csv", index=False)
    pd.DataFrame(component_rows).sort_values(
        ["processed_rows", "unit_count"], ascending=False
    ).to_csv(out_root / "duplicate_connected_components.csv", index=False)
    pd.DataFrame(edge_rows).sort_values(
        "shared_exact_feature_groups", ascending=False
    ).to_csv(out_root / "duplicate_unit_edges.csv", index=False)

    largest = max(component_rows, key=lambda r: int(r["processed_rows"]))
    multi_unit_components = [r for r in component_rows if int(r["unit_count"]) > 1]
    multi_source_components = [r for r in component_rows if int(r["source_file_count"]) > 1]

    summary = {
        "dataset": "CICIDS2017",
        "raw_dataset_root": str(raw_root),
        "base_csv_files": len(files),
        "contiguous_units": len(unit_specs),
        "subfiles_per_source_file": int(args.subfiles_per_file),
        "post_preprocessing_rows": int(total_rows),
        "feature_columns": len(feature_columns_ref or []),
        "unique_exact_feature_hashes": int(len(hash_masks)),
        "exact_feature_groups_spanning_multiple_units": int(cross_unit_groups),
        "exact_feature_groups_spanning_multiple_source_files": int(cross_source_groups),
        "family_conflict_feature_groups": int(len(family_conflict_hashes)),
        "duplicate_connected_components": int(len(component_rows)),
        "multi_unit_components": int(len(multi_unit_components)),
        "multi_source_file_components": int(len(multi_source_components)),
        "largest_component_units": int(largest["unit_count"]),
        "largest_component_source_files": int(largest["source_file_count"]),
        "largest_component_processed_rows": int(largest["processed_rows"]),
        "largest_component_source_file_names": largest["source_files"],
        "interpretation": (
            "Units in one connected component cannot be assigned to different splits "
            "if exact processed-feature duplicates are to be prohibited across splits."
        ),
    }
    with (out_root / "duplicate_provenance_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (out_root / "audit_scope.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "preprocessing_source": "ids_eval_framework._native.prepare_datasets",
                "preprocessing_replayed": [
                    "column canonicalization",
                    "leakage-column detection/removal",
                    "coarse port bucketing before raw port removal",
                    "infinite-to-missing conversion",
                    "CICIDS2017 family mapping",
                    "Other-family removal",
                    "raw label removal",
                ],
                "partition_unit": (
                    f"{int(args.subfiles_per_file)} contiguous row-order segments per base source CSV"
                ),
                "hash_definition": "all post-preprocessing predictor columns; target columns excluded",
                "hash_method": "pandas deterministic 64-bit row hash",
                "near_duplicate_search_performed": False,
                "split_selected": False,
                "row_level_hashes_written": False,
            },
            f,
            indent=2,
        )

    print("[provenance] complete")
    print(f"[provenance] post-preprocessing rows: {total_rows:,}")
    print(f"[provenance] unique feature hashes: {len(hash_masks):,}")
    print(f"[provenance] cross-unit duplicate groups: {cross_unit_groups:,}")
    print(f"[provenance] cross-source duplicate groups: {cross_source_groups:,}")
    print(f"[provenance] connected components: {len(component_rows)}")
    print(
        "[provenance] largest component: "
        f"{largest['unit_count']} unit(s), {largest['source_file_count']} source file(s), "
        f"{int(largest['processed_rows']):,} processed rows"
    )
    print(f"[provenance] wrote aggregate artifacts to {out_root}")


if __name__ == "__main__":
    main()
