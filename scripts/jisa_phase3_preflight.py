#!/usr/bin/env python3
"""Preflight local five-seed CICIoT2023 Protocol-B evidence for JISA Phase 3.

This audit is read-only with respect to historical scientific outputs. It verifies
whether each historical candidate-profile winner is uniquely determined by Stage-2
validation macro-F1 alone, so a heldout-visible Stage-1 AUROC tie-break is unnecessary.

Local historical artifacts are not guaranteed to live under the current canonical
``outputs/10_seed_reliability`` path. The preflight therefore discovers candidate
seed aggregates from the Phase-0b inventory first, then from a small set of known
legacy/local research roots. Discovery itself never changes scientific artifacts.
Audit output is written only under ``.release-audit/``.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [123, 124, 125, 126, 127]
AUDIT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase3_preflight"
PHASE0B_INVENTORY = REPO_ROOT / ".release-audit" / "jisa_phase0b" / "interesting_files.csv"
DATASET = "CICIoT2023"
PRIMARY_TOL = 1e-12

# Preferred/current path first, then local legacy surfaces explicitly scanned during
# Phase 0b. We avoid blindly crawling the whole user profile or drive.
SEARCH_ROOTS = [
    REPO_ROOT / "outputs" / "10_seed_reliability",
    REPO_ROOT / "jisa_results_bundle",
    REPO_ROOT.parent / "Codes",
    REPO_ROOT.parent / "outputs",
]

# Prepared-data locations known from the repository configurations plus common local
# aliases. Manifest paths remain preferred whenever they resolve successfully.
PROCESSED_DATA_CANDIDATES = [
    REPO_ROOT / "outputs" / "02_prepared_data" / "processed_V5" / "B_day_file" / DATASET,
    REPO_ROOT / "processed_V5" / "B_day_file" / DATASET,
    REPO_ROOT.parent / "Codes" / "outputs" / "02_prepared_data" / "processed_V5" / "B_day_file" / DATASET,
    REPO_ROOT.parent / "Codes" / "processed_V5" / "B_day_file" / DATASET,
    REPO_ROOT.parent / "outputs" / "02_prepared_data" / "processed_V5" / "B_day_file" / DATASET,
]


REQUIRED_AGGREGATE_COLUMNS = {
    "dataset",
    "holdout_family",
    "stage2_macro_f1_val",
    "apply_loao_stage1",
}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def resolve_local_path(raw: str, anchors: Iterable[Path] = ()) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    candidates = [REPO_ROOT / p, REPO_ROOT.parent / "Codes" / p, REPO_ROOT.parent / p]
    for anchor in anchors:
        candidates.append(anchor / p)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            pass
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


def seed_from_path(path: Path) -> int | None:
    text = str(path).replace("\\", "/").lower()
    for seed in SEEDS:
        if re.search(rf"(?:^|[/_.-])seed[_-]?{seed}(?:[/_.-]|$)", text):
            return seed
        if f"seed_{seed}" in text or f"seed-{seed}" in text:
            return seed
    return None


def seed_from_df(df: pd.DataFrame) -> int | None:
    for col in ("seed", "random_seed", "global_seed"):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique().tolist()
        if len(vals) == 1 and vals[0] in SEEDS:
            return int(vals[0])
    return None


def phase0b_aggregate_candidates() -> list[Path]:
    if not PHASE0B_INVENTORY.exists():
        return []
    try:
        inv = pd.read_csv(PHASE0B_INVENTORY)
    except Exception:
        return []
    if "path" not in inv.columns:
        return []
    out: list[Path] = []
    for raw in inv["path"].dropna().astype(str):
        if Path(raw).name.lower() != "aggregate_results.csv":
            continue
        p = resolve_local_path(raw)
        if p.exists():
            out.append(p)
    return out


def filesystem_aggregate_candidates() -> list[Path]:
    out: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        try:
            out.extend(root.rglob("aggregate_results.csv"))
        except Exception:
            continue
    return out


def discover_aggregate_candidates(extra: list[str]) -> list[Path]:
    paths: list[Path] = []
    paths.extend(phase0b_aggregate_candidates())
    paths.extend(filesystem_aggregate_candidates())
    for raw in extra:
        p = Path(raw).expanduser()
        if p.is_dir():
            paths.extend(p.rglob("aggregate_results.csv"))
        elif p.is_file():
            paths.append(p)

    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower()
        if key in seen or not rp.is_file():
            continue
        seen.add(key)
        out.append(rp)
    return sorted(out, key=lambda x: str(x).lower())


def inspect_aggregate_candidate(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "path_seed": seed_from_path(path),
        "readable": False,
        "qualifies": False,
        "rows_ciciot": 0,
        "holdouts": 0,
        "candidate_groups": 0,
        "three_candidate_groups": 0,
        "df_seed": None,
        "error": "",
    }
    try:
        df = pd.read_csv(path)
        record["readable"] = True
    except Exception as exc:
        record["error"] = f"read_error: {exc}"
        return record

    record["df_seed"] = seed_from_df(df)
    missing = sorted(REQUIRED_AGGREGATE_COLUMNS.difference(df.columns))
    if missing:
        record["error"] = "missing_columns:" + ",".join(missing)
        return record

    ds = df[df["dataset"].astype(str) == DATASET].copy()
    record["rows_ciciot"] = int(len(ds))
    if ds.empty:
        record["error"] = "no_ciciot_rows"
        return record

    cols = candidate_columns(ds)
    if not cols:
        record["error"] = "no_candidate_identity_columns"
        return record
    ds["_candidate_id"] = ds.apply(lambda r: candidate_id(r, cols), axis=1)
    holdouts = set(ds["holdout_family"].astype(str))
    record["holdouts"] = int(len(holdouts))
    counts = (
        ds.groupby("holdout_family", sort=True)["_candidate_id"]
        .nunique()
        .astype(int)
    )
    record["candidate_groups"] = int(len(counts))
    record["three_candidate_groups"] = int((counts == 3).sum())
    record["qualifies"] = bool(len(holdouts) == 6 and len(counts) == 6 and (counts == 3).all())
    return record


def choose_seed_aggregates(candidates: list[Path]) -> tuple[dict[int, Path], pd.DataFrame]:
    records = [inspect_aggregate_candidate(p) for p in candidates]
    audit_df = pd.DataFrame(records)
    selected: dict[int, Path] = {}

    for seed in SEEDS:
        matching = []
        for rec in records:
            inferred = rec.get("df_seed") or rec.get("path_seed")
            if inferred != seed:
                continue
            score = (
                int(bool(rec.get("qualifies"))),
                int(rec.get("three_candidate_groups", 0)),
                int(rec.get("holdouts", 0)),
                int(rec.get("rows_ciciot", 0)),
                int("protocol_b" in str(rec.get("path", "")).lower()),
                int("seed_reliability" in str(rec.get("path", "")).lower()),
            )
            matching.append((score, Path(str(rec["path"]))))
        if matching:
            matching.sort(key=lambda x: (x[0], str(x[1]).lower()), reverse=True)
            selected[seed] = matching[0][1]

    return selected, audit_df


def manifest_candidates(aggregate_path: Path, aggregate: pd.DataFrame) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    # Local run copies are the most trustworthy for resolving historical relative
    # paths because they travel with the aggregate surface.
    for ancestor in [aggregate_path.parent, *list(aggregate_path.parents)[:4]]:
        if not ancestor.exists():
            continue
        try:
            for path in ancestor.rglob("scenario_manifest.json"):
                key = str(path.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    out.append(path)
        except Exception:
            pass

    if "manifest_path" in aggregate.columns:
        for raw in aggregate["manifest_path"].dropna().astype(str):
            path = resolve_local_path(raw, anchors=[aggregate_path.parent])
            key = str(path).lower()
            if path.exists() and key not in seen:
                seen.add(key)
                out.append(path)
    return out


def split_part_count(dataset_dir: Path, split: str) -> int:
    count = 0
    try:
        iterator = dataset_dir.rglob("*")
    except Exception:
        return 0
    for path in iterator:
        if not path.is_file():
            continue
        name = path.name.lower()
        parent = path.parent.name.lower()
        if name.startswith(split.lower()) or parent == split.lower():
            count += 1
    return count


def processed_dir_from_manifest(raw: str, manifest_path: Path, aggregate_path: Path) -> Path:
    anchors = [manifest_path.parent, aggregate_path.parent, REPO_ROOT.parent / "Codes"]
    return resolve_local_path(raw, anchors=anchors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extra-search",
        action="append",
        default=[],
        help="Optional additional aggregate_results.csv file or directory to search; repeatable.",
    )
    args = parser.parse_args()

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

    candidate_paths = discover_aggregate_candidates(args.extra_search)
    selected_aggregates, discovery_df = choose_seed_aggregates(candidate_paths)
    if not discovery_df.empty:
        discovery_df.to_csv(AUDIT_ROOT / "aggregate_discovery.csv", index=False)
    else:
        (AUDIT_ROOT / "aggregate_discovery.csv").write_text("", encoding="utf-8")

    seed_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    seed_holdout_sets: dict[int, set[str]] = {}
    missing_seed_files: list[int] = []

    for seed in SEEDS:
        aggregate_path = selected_aggregates.get(seed)
        if aggregate_path is None or not aggregate_path.exists():
            missing_seed_files.append(seed)
            continue

        df = pd.read_csv(aggregate_path)
        df = df[df["dataset"].astype(str) == DATASET].copy()
        if df.empty:
            seed_rows.append({"seed": seed, "aggregate": str(aggregate_path), "rows": 0, "holdouts": 0})
            seed_holdout_sets[seed] = set()
            continue

        missing = sorted(REQUIRED_AGGREGATE_COLUMNS.difference(df.columns))
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
            # Interrupted/resumed historical execution may append duplicate rows.
            # Candidate identity, not raw append count, defines matrix completeness.
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

        for path in manifest_candidates(aggregate_path, df):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("dataset", "")) != DATASET:
                continue
            processed_raw = str(payload.get("processed_dir", ""))
            processed_path = (
                processed_dir_from_manifest(processed_raw, path, aggregate_path)
                if processed_raw
                else Path()
            )
            manifest_records.append(
                {
                    "seed": seed,
                    "aggregate_path": str(aggregate_path),
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

    processed_candidates: list[Path] = []
    if not manifest_df.empty:
        for raw in sorted(set(manifest_df.loc[manifest_df["processed_dir_exists"], "processed_dir"].astype(str))):
            p = Path(raw)
            if p.exists():
                processed_candidates.append(p)
    for p in PROCESSED_DATA_CANDIDATES:
        if p.exists():
            processed_candidates.append(p.resolve())

    dedup_processed: list[Path] = []
    seen_processed: set[str] = set()
    for p in processed_candidates:
        key = str(p.resolve()).lower()
        if key not in seen_processed:
            seen_processed.add(key)
            dedup_processed.append(p.resolve())

    processed_ready = False
    split_counts: dict[str, dict[str, int]] = {}
    for p in dedup_processed:
        counts = {split: split_part_count(p, split) for split in ("train", "val", "test")}
        split_counts[str(p)] = counts
        if all(v > 0 for v in counts.values()):
            processed_ready = True

    profile_reuse_safe = bool(
        complete_candidate_matrix
        and strict_loao
        and unique_primary
        and stable_winners
    )
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
        "aggregate_candidates_examined": int(len(candidate_paths)),
        "selected_seed_aggregates": {str(k): str(v) for k, v in selected_aggregates.items()},
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
        "existing_processed_dirs": [str(x) for x in dedup_processed],
        "split_part_counts": split_counts,
        "processed_protocol_b_ready": processed_ready,
        "historical_profile_reuse_blind_safe": profile_reuse_safe,
        "PASS": pass_all,
    }
    (AUDIT_ROOT / "preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"aggregate candidates examined: {len(candidate_paths)}")
    for seed in SEEDS:
        chosen = selected_aggregates.get(seed)
        print(f"seed {seed} aggregate: {chosen if chosen else 'NOT FOUND'}")
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

    print(f"historical profile reuse blind-safe: {profile_reuse_safe}")
    print(f"discovery audit: {AUDIT_ROOT / 'aggregate_discovery.csv'}")
    print(f"audit: {AUDIT_ROOT / 'preflight.json'}")
    print(f"PASS={pass_all}")
    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
