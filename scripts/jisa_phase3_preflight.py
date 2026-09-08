#!/usr/bin/env python3
"""Preflight the local five-seed CICIoT2023 Protocol-B evidence for JISA Phase 3.

This script is read-only with respect to historical outputs. It verifies whether the
historical candidate-profile winner for every seed/holdout is uniquely determined by
Stage-2 validation macro-F1 alone, so the Stage-1 AUROC tie-break (which may have seen
the held-out validation family) is unnecessary. It also verifies the five-seed /
six-holdout candidate matrix, strict Stage-1 LOAO, and recoverability of the prepared
Protocol-B dataset referenced by local manifests.

Audit output is written only under .release-audit/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [123, 124, 125, 126, 127]
LANE_ROOT = REPO_ROOT / "outputs" / "10_seed_reliability" / "protocol_b_loao"
AUDIT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase3_preflight"
DATASET = "CICIoT2023"
PRIMARY_TOL = 1e-12


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def resolve_local_path(raw: str) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def candidate_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "model_profile",
        "model_family",
        "apply_loao_stage1",
        "stage1_weight_mode",
        "stage2_weight_mode",
        "stage1_params",
        "stage2_params",
    ]
    return [c for c in preferred if c in df.columns]


def candidate_id(row: pd.Series, cols: list[str]) -> str:
    parts = []
    for col in cols:
        value = row[col]
        if isinstance(value, float) and math.isnan(value):
            value = ""
        parts.append(f"{col}={value}")
    return "|".join(parts)


def short_winner_label(row: pd.Series) -> str:
    profile = str(row.get("model_profile", "")).strip()
    family = str(row.get("model_family", "")).strip()
    weight = str(row.get("stage1_weight_mode", "")).strip()
    if profile and profile not in {"nan", family}:
        return profile
    if family and weight and weight != "nan":
        return f"{family}_{weight}"
    return profile or family or "unknown"


def manifest_candidates(seed_root: Path, aggregate: pd.DataFrame) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    for path in seed_root.rglob("scenario_manifest.json"):
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            out.append(path)

    if "manifest_path" in aggregate.columns:
        for raw in aggregate["manifest_path"].dropna().astype(str):
            path = resolve_local_path(raw)
            key = str(path)
            if path.exists() and key not in seen:
                seen.add(key)
                out.append(path)

    return out


def split_part_count(dataset_dir: Path, split: str) -> int:
    count = 0
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        parent = path.parent.name.lower()
        if name.startswith(split.lower()) or parent == split.lower():
            count += 1
    return count


def main() -> int:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    seed_holdout_sets: dict[int, set[str]] = {}
    missing_seed_files: list[int] = []

    for seed in SEEDS:
        runs_root = LANE_ROOT / f"seed_{seed}" / "runs"
        aggregate_path = runs_root / "aggregate_results.csv"
        if not aggregate_path.exists():
            missing_seed_files.append(seed)
            continue

        df = pd.read_csv(aggregate_path)
        if "dataset" not in df.columns:
            raise RuntimeError(f"Missing dataset column: {aggregate_path}")
        df = df[df["dataset"].astype(str) == DATASET].copy()
        if df.empty:
            seed_rows.append({"seed": seed, "aggregate": str(aggregate_path), "rows": 0, "holdouts": 0})
            seed_holdout_sets[seed] = set()
            continue

        required = {"holdout_family", "stage2_macro_f1_val", "apply_loao_stage1"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise RuntimeError(f"Missing required columns in {aggregate_path}: {missing}")

        cols = candidate_columns(df)
        if not cols:
            raise RuntimeError(f"Cannot identify candidate columns in {aggregate_path}")

        df["stage2_macro_f1_val"] = pd.to_numeric(df["stage2_macro_f1_val"], errors="coerce")
        df["_candidate_id"] = df.apply(lambda r: candidate_id(r, cols), axis=1)
        df["_strict_loao"] = df["apply_loao_stage1"].map(as_bool)

        holdouts = set(df["holdout_family"].astype(str).tolist())
        seed_holdout_sets[seed] = holdouts
        seed_rows.append(
            {
                "seed": seed,
                "aggregate": str(aggregate_path),
                "rows": int(len(df)),
                "holdouts": int(len(holdouts)),
                "all_strict_loao": bool(df["_strict_loao"].all()),
            }
        )

        for holdout, group in df.groupby("holdout_family", sort=True):
            # Aggregate files can contain duplicate append rows after interrupted/resumed
            # execution. Candidate identity, not raw row count, defines matrix completeness.
            group = group.sort_index().drop_duplicates(subset=["_candidate_id"], keep="last").copy()
            group = group[group["stage2_macro_f1_val"].notna()].copy()
            group = group.sort_values(["stage2_macro_f1_val", "_candidate_id"], ascending=[False, True])

            if group.empty:
                selection_rows.append(
                    {
                        "seed": seed,
                        "holdout_family": str(holdout),
                        "candidate_count": 0,
                        "unique_stage2_winner": False,
                        "stage2_margin": float("nan"),
                        "winner_label": "",
                        "winner_candidate_id": "",
                        "all_candidates_strict_loao": False,
                    }
                )
                continue

            top = group.iloc[0]
            if len(group) >= 2:
                second = group.iloc[1]
                margin = float(top["stage2_macro_f1_val"] - second["stage2_macro_f1_val"])
                unique = bool(margin > PRIMARY_TOL)
            else:
                margin = float("nan")
                unique = False

            selection_rows.append(
                {
                    "seed": seed,
                    "holdout_family": str(holdout),
                    "candidate_count": int(len(group)),
                    "unique_stage2_winner": unique,
                    "stage2_margin": margin,
                    "winner_stage2_macro_f1_val": float(top["stage2_macro_f1_val"]),
                    "winner_label": short_winner_label(top),
                    "winner_candidate_id": str(top["_candidate_id"]),
                    "all_candidates_strict_loao": bool(group["_strict_loao"].all()),
                }
            )

        for path in manifest_candidates(runs_root, df):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("dataset", "")) != DATASET:
                continue
            processed_raw = str(payload.get("processed_dir", ""))
            processed_path = resolve_local_path(processed_raw) if processed_raw else Path()
            manifest_records.append(
                {
                    "seed": seed,
                    "manifest_path": str(path),
                    "holdout_family": str(payload.get("holdout_family", "")),
                    "processed_dir_raw": processed_raw,
                    "processed_dir": str(processed_path) if processed_raw else "",
                    "processed_dir_exists": bool(processed_raw and processed_path.exists()),
                }
            )

    seed_df = pd.DataFrame(seed_rows)
    sel_df = pd.DataFrame(selection_rows)
    manifest_df = pd.DataFrame(manifest_records)

    if not seed_df.empty:
        seed_df.to_csv(AUDIT_ROOT / "seed_inventory.csv", index=False)
    if not sel_df.empty:
        sel_df.to_csv(AUDIT_ROOT / "candidate_profile_audit.csv", index=False)
    if not manifest_df.empty:
        manifest_df.to_csv(AUDIT_ROOT / "manifest_inventory.csv", index=False)

    union_holdouts: set[str] = set()
    for values in seed_holdout_sets.values():
        union_holdouts.update(values)
    expected_holdouts = sorted(union_holdouts)

    all_seed_files = len(missing_seed_files) == 0 and len(seed_holdout_sets) == len(SEEDS)
    six_holdouts = len(expected_holdouts) == 6 and all(
        seed_holdout_sets.get(seed, set()) == set(expected_holdouts) for seed in SEEDS
    )

    candidate_groups_expected = len(SEEDS) * 6
    complete_candidate_matrix = (
        len(sel_df) == candidate_groups_expected
        and not sel_df.empty
        and bool((sel_df["candidate_count"] == 3).all())
    )
    strict_loao = (
        not sel_df.empty
        and bool(sel_df["all_candidates_strict_loao"].all())
        and (seed_df.empty or bool(seed_df.get("all_strict_loao", pd.Series([False])).fillna(False).all()))
    )
    unique_primary = not sel_df.empty and bool(sel_df["unique_stage2_winner"].all())

    stable_by_holdout: dict[str, bool] = {}
    winner_labels: dict[str, list[str]] = {}
    if not sel_df.empty:
        for holdout, group in sel_df.groupby("holdout_family", sort=True):
            ids = group["winner_candidate_id"].astype(str).tolist()
            labels = group["winner_label"].astype(str).tolist()
            stable_by_holdout[str(holdout)] = len(group) == len(SEEDS) and len(set(ids)) == 1
            winner_labels[str(holdout)] = labels
    stable_winners = len(stable_by_holdout) == 6 and all(stable_by_holdout.values())

    existing_processed_dirs: list[Path] = []
    if not manifest_df.empty:
        for raw in sorted(set(manifest_df.loc[manifest_df["processed_dir_exists"], "processed_dir"].astype(str))):
            p = Path(raw)
            if p.exists():
                existing_processed_dirs.append(p)

    processed_ready = False
    split_counts: dict[str, dict[str, int]] = {}
    for p in existing_processed_dirs:
        counts = {split: split_part_count(p, split) for split in ("train", "val", "test")}
        split_counts[str(p)] = counts
        if all(v > 0 for v in counts.values()):
            processed_ready = True

    pass_all = all(
        [
            all_seed_files,
            six_holdouts,
            complete_candidate_matrix,
            strict_loao,
            unique_primary,
            stable_winners,
            processed_ready,
        ]
    )

    payload = {
        "dataset": DATASET,
        "seeds_expected": SEEDS,
        "missing_seed_aggregates": missing_seed_files,
        "holdouts": expected_holdouts,
        "all_seed_files": all_seed_files,
        "six_holdouts_per_seed": six_holdouts,
        "candidate_groups_expected": candidate_groups_expected,
        "candidate_groups_found": int(len(sel_df)),
        "complete_three_candidate_matrix": complete_candidate_matrix,
        "strict_stage1_loao": strict_loao,
        "unique_stage2_primary_winner_all_groups": unique_primary,
        "stable_winner_all_holdouts": stable_winners,
        "stable_by_holdout": stable_by_holdout,
        "winner_labels_by_holdout": winner_labels,
        "existing_processed_dirs": [str(x) for x in existing_processed_dirs],
        "split_part_counts": split_counts,
        "processed_protocol_b_ready": processed_ready,
        "historical_profile_reuse_blind_safe": bool(unique_primary and stable_winners),
        "PASS": pass_all,
    }
    (AUDIT_ROOT / "preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"seed aggregates: {len(SEEDS) - len(missing_seed_files)}/{len(SEEDS)}")
    print(f"holdouts: {len(expected_holdouts)} :: {', '.join(expected_holdouts)}")
    print(f"candidate groups: {len(sel_df)}/{candidate_groups_expected}")
    if not sel_df.empty:
        print(f"three-candidate groups: {int((sel_df['candidate_count'] == 3).sum())}/{candidate_groups_expected}")
        print(f"unique Stage-2 primary winners: {int(sel_df['unique_stage2_winner'].sum())}/{candidate_groups_expected}")
    print(f"stable winner holdouts: {sum(stable_by_holdout.values())}/{len(stable_by_holdout) if stable_by_holdout else 6}")
    print(f"strict Stage-1 LOAO: {strict_loao}")
    print(f"processed Protocol-B data ready: {processed_ready}")

    if not sel_df.empty:
        view = sel_df[
            [
                "seed",
                "holdout_family",
                "candidate_count",
                "winner_label",
                "winner_stage2_macro_f1_val",
                "stage2_margin",
                "unique_stage2_winner",
            ]
        ].sort_values(["holdout_family", "seed"])
        print(view.to_string(index=False))

    print(f"historical profile reuse blind-safe: {bool(unique_primary and stable_winners)}")
    print(f"audit: {AUDIT_ROOT / 'preflight.json'}")
    print(f"PASS={pass_all}")
    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
