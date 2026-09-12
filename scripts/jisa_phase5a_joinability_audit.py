#!/usr/bin/env python3
"""Resolve provenance joinability for JISA Phase 5A without running statistics.

The Phase-5 preflight only establishes that provenance-bearing evidence exists
somewhere in the retained workspace. This audit determines whether claim-bearing
row-level score surfaces themselves carry admissible provenance, or can be linked
through an explicit retained mapping. Synthetic positional row_id values are not
accepted as provenance keys.

Outputs are written only under .release-audit/jisa_phase5_joinability/.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase5_statistics_preflight"
OUT = REPO_ROOT / ".release-audit" / "jisa_phase5_joinability"
OPEN_SET_SOURCE = REPO_ROOT / "src" / "ids_eval_framework" / "_native" / "protocol_b_open_set.py"

PROVENANCE_TERMS = [
    "source_file", "sourcefile", "raw_file", "rawfile", "filename", "file_name",
    "source_day", "capture_day", "file_day", "source_unit", "unit_id", "capture_id",
    "session_id", "pcap", "group_id", "identity_group", "exact_identity",
    "duplicate_group", "source_group",
]
JOIN_TERMS = ["row_id", "sample_id", "record_id", "flow_id", "source_row", "original_index", "index", "id"]
SYNTHETIC_POSITIONAL_KEYS = {"row_id", "index", "original_index"}
MAPPING_SUFFIXES = {".csv", ".gz"}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def split_pipe(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    return [x for x in text.split("|") if x]


def lowset(values: list[str]) -> set[str]:
    return {str(x).strip().lower() for x in values}


def header(path: Path) -> list[str]:
    try:
        if path.name.lower().endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
                return [str(x).strip() for x in next(csv.reader(f))]
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                return [str(x).strip() for x in next(csv.reader(f))]
    except Exception:
        pass
    return []


def term_columns(cols: list[str], terms: list[str]) -> list[str]:
    out: list[str] = []
    for col in cols:
        key = col.lower()
        for term in terms:
            if key == term or term in key:
                out.append(col)
                break
    return sorted(set(out))


def verify_phase3_row_id_is_synthetic() -> bool:
    if not OPEN_SET_SOURCE.exists():
        raise RuntimeError(f"Missing score-builder source: {OPEN_SET_SOURCE}")
    text = OPEN_SET_SOURCE.read_text(encoding="utf-8", errors="ignore")
    # Require the exact construction pattern semantically enough to avoid silently
    # relying on an outdated assumption.
    pattern = re.compile(r'["\']row_id["\']\s*:\s*np\.arange\(len\(frame\)', re.MULTILINE)
    if not pattern.search(text):
        raise RuntimeError(
            "Could not verify that build_score_frame constructs row_id from np.arange(len(frame)). "
            "Stop rather than treating row_id as synthetic by assumption."
        )
    return True


def candidate_mapping_tables(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in metadata.get("path", pd.Series(dtype=str)).astype(str):
        path = REPO_ROOT / raw
        if not path.exists() or not path.is_file():
            continue
        if not (path.suffix.lower() == ".csv" or path.name.lower().endswith(".csv.gz")):
            continue
        cols = header(path)
        if not cols:
            continue
        prov = term_columns(cols, PROVENANCE_TERMS)
        joins = term_columns(cols, JOIN_TERMS)
        if not prov or not joins:
            continue
        rows.append(
            {
                "path": rel(path),
                "join_columns": "|".join(joins),
                "provenance_columns": "|".join(prov),
                "size_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def mapping_support_for_score(score_join_cols: list[str], maps: pd.DataFrame) -> tuple[list[str], list[str]]:
    valid_score_keys = [c for c in score_join_cols if c.lower() not in SYNTHETIC_POSITIONAL_KEYS]
    if not valid_score_keys or maps.empty:
        return [], []
    matched_paths: list[str] = []
    matched_keys: list[str] = []
    wanted = lowset(valid_score_keys)
    for _, row in maps.iterrows():
        mkeys = lowset(split_pipe(row.get("join_columns", "")))
        common = sorted(wanted & mkeys)
        if common:
            matched_paths.append(str(row["path"]))
            matched_keys.extend(common)
    return sorted(set(matched_paths)), sorted(set(matched_keys))


def classify_surface(row: pd.Series, maps: pd.DataFrame) -> dict[str, Any]:
    prov = split_pipe(row.get("provenance_columns", ""))
    joins = split_pipe(row.get("join_key_columns", ""))
    join_low = lowset(joins)
    only_synthetic = bool(joins) and all(x in SYNTHETIC_POSITIONAL_KEYS for x in join_low)
    map_paths, map_keys = mapping_support_for_score(joins, maps)

    if prov:
        cls = "direct_provenance"
        cluster_ready = True
        reason = "score surface contains provenance columns directly"
    elif map_paths and map_keys:
        # The existence of compatible headers is still not enough to declare a verified
        # join. Phase 5B may only promote this class after explicit cardinality/match-rate
        # validation for the selected surface.
        cls = "candidate_mapping_unverified"
        cluster_ready = False
        reason = "non-synthetic score key and provenance-bearing mapping table share a candidate key; exact join not yet verified"
    else:
        cls = "conditional_row_only"
        cluster_ready = False
        if only_synthetic:
            reason = "only candidate score key is synthetic positional row_id/index"
        elif joins:
            reason = "score has candidate key(s) but no explicit provenance-bearing mapping table with a shared non-synthetic key"
        else:
            reason = "score surface has neither direct provenance nor an admissible mapping key"

    return {
        "path": str(row.get("path", "")),
        "dataset_guess": str(row.get("dataset_guess", "")),
        "surface_guess": str(row.get("surface_guess", "")),
        "provenance_columns": "|".join(prov),
        "join_key_columns": "|".join(joins),
        "synthetic_positional_key_only": bool(only_synthetic),
        "mapping_candidate_paths": "|".join(map_paths),
        "mapping_candidate_shared_keys": "|".join(map_keys),
        "evidence_class": cls,
        "cluster_ready_without_further_join_validation": bool(cluster_ready),
        "reason": reason,
    }


def main() -> int:
    preflight_path = PREFLIGHT_ROOT / "preflight.json"
    row_inventory_path = PREFLIGHT_ROOT / "row_level_score_inventory.csv"
    metadata_path = PREFLIGHT_ROOT / "provenance_metadata_candidates.csv"
    for path in [preflight_path, row_inventory_path, metadata_path]:
        if not path.exists():
            raise RuntimeError(f"Missing Phase-5 preflight artifact: {path}")

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not bool(preflight.get("READY_FOR_PHASE5_DESIGN")):
        raise RuntimeError("Phase-5 preflight is not ready for design.")

    synthetic_verified = verify_phase3_row_id_is_synthetic()
    scores = pd.read_csv(row_inventory_path)
    metadata = pd.read_csv(metadata_path) if metadata_path.stat().st_size else pd.DataFrame()
    maps = candidate_mapping_tables(metadata)

    classified = pd.DataFrame([classify_surface(row, maps) for _, row in scores.iterrows()])
    phase3 = classified[classified["surface_guess"].astype(str) == "phase3_validation_blind"].copy()
    visible = classified[classified["surface_guess"].astype(str) == "validation_visible_rejectors"].copy()

    phase3_direct = int((phase3["evidence_class"] == "direct_provenance").sum()) if not phase3.empty else 0
    phase3_candidate = int((phase3["evidence_class"] == "candidate_mapping_unverified").sum()) if not phase3.empty else 0
    phase3_conditional = int((phase3["evidence_class"] == "conditional_row_only").sum()) if not phase3.empty else 0
    phase3_cluster_ready = bool(phase3_direct > 0)

    # Ready for Phase-5B specification once the audit has classified every discovered
    # row-level surface. Candidate mappings remain explicitly unverified and therefore
    # cannot be used for cluster inference until a selected join is validated.
    fully_classified = bool(len(classified) == len(scores) and not classified["evidence_class"].isna().any())
    ready = bool(synthetic_verified and fully_classified)

    OUT.mkdir(parents=True, exist_ok=True)
    classified.to_csv(OUT / "surface_joinability.csv", index=False)
    maps.to_csv(OUT / "mapping_table_candidates.csv", index=False)

    decision = {
        "phase5_preflight_ready": True,
        "phase3_row_id_synthetic_verified_from_source": synthetic_verified,
        "phase3_row_level_surfaces": int(len(phase3)),
        "phase3_direct_provenance_surfaces": phase3_direct,
        "phase3_candidate_mapping_unverified_surfaces": phase3_candidate,
        "phase3_conditional_row_only_surfaces": phase3_conditional,
        "phase3_cluster_aware_row_resampling_ready_now": phase3_cluster_ready,
        "validation_visible_row_level_surfaces": int(len(visible)),
        "all_row_level_surfaces_classified": fully_classified,
        "mapping_table_candidates": int(len(maps)),
        "new_statistical_tests_performed": False,
        "new_model_training_performed": False,
        "READY_FOR_PHASE5B_SPEC": ready,
        "interpretation": (
            "A True Phase-5 preflight means provenance evidence exists somewhere, not that Phase-3 scores are cluster-joinable. "
            "Synthetic positional row_id is inadmissible as provenance. Candidate mappings are not promoted to cluster-ready until exact join validation."
        ),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    print(f"row-level surfaces classified: {len(classified)}/{len(scores)}")
    print(f"mapping table candidates: {len(maps)}")
    print(f"Phase-3 row_id synthetic verified from source: {synthetic_verified}")
    print(f"Phase-3 row-level surfaces: {len(phase3)}")
    print(f"Phase-3 direct-provenance surfaces: {phase3_direct}")
    print(f"Phase-3 candidate-mapping-unverified surfaces: {phase3_candidate}")
    print(f"Phase-3 conditional-row-only surfaces: {phase3_conditional}")
    print(f"Phase-3 cluster-aware row resampling ready now: {phase3_cluster_ready}")
    print(f"validation-visible row-level surfaces: {len(visible)}")
    if not phase3.empty:
        counts = phase3.groupby("evidence_class").size().to_dict()
        print(f"Phase-3 evidence classes: {counts}")
    print(f"audit: {OUT / 'decision.json'}")
    print(f"READY_FOR_PHASE5B_SPEC={ready}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
