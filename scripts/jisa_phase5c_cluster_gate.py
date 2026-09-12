#!/usr/bin/env python3
"""Audit raw rejection-score provenance and cluster cardinality for JISA Phase 5C.

No model fitting, threshold selection, resampling, confidence interval, or hypothesis
test occurs here. The script verifies provenance on the exact raw `test_scores` files
used for rejector replay. Derived summaries cannot establish cluster readiness.

Outputs are written only under `.release-audit/jisa_phase5c_cluster_gate/`.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE3_ROOT = REPO_ROOT / "outputs" / "13_jisa_q1_revision" / "phase3_validation_blind" / "score_generation"
VISIBLE_ROOT = REPO_ROOT / "outputs" / "12_jisa_finalization" / "14_rejector_tradeoff_runs"
PHASE5A_SURFACES = REPO_ROOT / ".release-audit" / "jisa_phase5_joinability" / "surface_joinability.csv"
OUT = REPO_ROOT / ".release-audit" / "jisa_phase5c_cluster_gate"

REQUIRED_SCORE_COLUMNS = {"y_true_sys", "p_attack", "fam_pred_family", "fam_pmax"}
ADMISSIBLE_PROVENANCE = {
    "source_file",
    "source_day",
    "file_day",
    "capture_day",
    "source_unit",
    "unit_id",
    "capture_id",
    "session_id",
    "pcap",
    "group_id",
    "identity_group",
    "exact_identity",
    "duplicate_group",
    "source_group",
}
INADMISSIBLE_POSITIONAL = {"row_id", "index", "original_index"}
MIN_CLUSTERS = 20
MAX_MISSING_FRACTION = 0.01
MAX_LARGEST_CLUSTER_SHARE = 0.50
CHUNKSIZE = 250_000
EXPECTED_PHASE3_CASES = 30


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def normalized_map(columns: list[str]) -> dict[str, str]:
    return {str(c).strip().lower(): str(c).strip() for c in columns}


def score_header(path: Path) -> list[str]:
    try:
        return [str(c) for c in pd.read_csv(path, nrows=0).columns]
    except Exception as exc:
        raise RuntimeError(f"Could not read score header {path}: {type(exc).__name__}: {exc}") from exc


def raw_score_record(path: Path, surface: str) -> dict[str, Any]:
    cols = score_header(path)
    cmap = normalized_map(cols)
    required_present = sorted(REQUIRED_SCORE_COLUMNS & set(cmap))
    missing_required = sorted(REQUIRED_SCORE_COLUMNS - set(cmap))
    provenance = sorted(ADMISSIBLE_PROVENANCE & set(cmap))
    positional = sorted(INADMISSIBLE_POSITIONAL & set(cmap))
    return {
        "path": rel(path),
        "surface": surface,
        "n_columns": len(cols),
        "required_score_columns_present": "|".join(required_present),
        "missing_required_score_columns": "|".join(missing_required),
        "is_raw_score_surface": not missing_required,
        "admissible_provenance_columns": "|".join(provenance),
        "inadmissible_positional_columns": "|".join(positional),
    }


def cluster_stats(path: Path, actual_col: str, normalized_col: str, surface: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    n_rows = 0
    n_missing = 0
    for chunk in pd.read_csv(path, usecols=[actual_col], chunksize=CHUNKSIZE):
        s = chunk[actual_col]
        n_rows += int(len(s))
        missing = s.isna() | s.astype(str).str.strip().isin(["", "nan", "None", "NA", "N/A"])
        n_missing += int(missing.sum())
        vals = s.loc[~missing].astype(str)
        counts.update(vals.tolist())

    sizes = np.asarray(list(counts.values()), dtype=np.int64)
    n_clusters = int(len(sizes))
    nonmissing_rows = int(sizes.sum()) if n_clusters else 0
    missing_fraction = float(n_missing / n_rows) if n_rows else 1.0
    largest_share = float(sizes.max() / nonmissing_rows) if nonmissing_rows and n_clusters else 1.0
    usable = bool(
        n_clusters >= MIN_CLUSTERS
        and missing_fraction <= MAX_MISSING_FRACTION
        and largest_share <= MAX_LARGEST_CLUSTER_SHARE
    )
    return {
        "path": rel(path),
        "surface": surface,
        "provenance_column": normalized_col,
        "actual_column": actual_col,
        "n_rows": n_rows,
        "n_nonmissing_rows": nonmissing_rows,
        "n_missing_rows": n_missing,
        "missing_fraction": missing_fraction,
        "n_clusters": n_clusters,
        "cluster_size_min": int(sizes.min()) if n_clusters else 0,
        "cluster_size_median": float(np.median(sizes)) if n_clusters else float("nan"),
        "cluster_size_mean": float(np.mean(sizes)) if n_clusters else float("nan"),
        "cluster_size_max": int(sizes.max()) if n_clusters else 0,
        "largest_cluster_share": largest_share,
        "passes_cardinality_gate": usable,
    }


def inspect_surface(paths: list[Path], surface: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    headers: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for path in paths:
        rec = raw_score_record(path, surface)
        headers.append(rec)
        if not bool(rec["is_raw_score_surface"]):
            continue
        cols = score_header(path)
        cmap = normalized_map(cols)
        for normalized_col in sorted(ADMISSIBLE_PROVENANCE & set(cmap)):
            stats.append(cluster_stats(path, cmap[normalized_col], normalized_col, surface))
    return pd.DataFrame(headers), pd.DataFrame(stats)


def consistent_usable_columns(headers: pd.DataFrame, stats: pd.DataFrame, expected_files: int | None = None) -> list[str]:
    if headers.empty:
        return []
    raw = headers[headers["is_raw_score_surface"] == True].copy()  # noqa: E712
    if expected_files is not None and len(raw) != expected_files:
        return []
    if raw.empty or stats.empty:
        return []

    file_paths = set(raw["path"].astype(str))
    usable: list[str] = []
    for col, g in stats.groupby("provenance_column", sort=True):
        if set(g["path"].astype(str)) != file_paths:
            continue
        if bool(g["passes_cardinality_gate"].astype(bool).all()):
            usable.append(str(col))
    return usable


def phase5a_discrepancy_audit(raw_paths: set[str]) -> dict[str, Any]:
    if not PHASE5A_SURFACES.exists():
        return {
            "phase5a_surface_inventory_present": False,
            "direct_provenance_claim_surfaces": 0,
            "direct_provenance_claim_surfaces_that_are_exact_raw_scores": 0,
            "direct_provenance_claim_surfaces_not_exact_raw_scores": 0,
        }
    df = pd.read_csv(PHASE5A_SURFACES)
    claim = df[
        df.get("surface_guess", pd.Series(dtype=str)).astype(str).isin(
            ["phase3_validation_blind", "validation_visible_rejectors"]
        )
        & (df.get("evidence_class", pd.Series(dtype=str)).astype(str) == "direct_provenance")
    ].copy()
    exact = claim["path"].astype(str).isin(raw_paths) if "path" in claim.columns else pd.Series(False, index=claim.index)
    return {
        "phase5a_surface_inventory_present": True,
        "direct_provenance_claim_surfaces": int(len(claim)),
        "direct_provenance_claim_surfaces_that_are_exact_raw_scores": int(exact.sum()),
        "direct_provenance_claim_surfaces_not_exact_raw_scores": int((~exact).sum()),
    }


def main() -> int:
    phase3_paths = sorted(PHASE3_ROOT.glob("seed_*/*/test_scores.csv.gz")) if PHASE3_ROOT.exists() else []
    visible_paths = sorted(VISIBLE_ROOT.glob("*/test_scores.csv.gz")) if VISIBLE_ROOT.exists() else []

    p3_headers, p3_stats = inspect_surface(phase3_paths, "phase3_validation_blind")
    vis_headers, vis_stats = inspect_surface(visible_paths, "validation_visible_rejectors")

    phase3_usable = consistent_usable_columns(p3_headers, p3_stats, expected_files=EXPECTED_PHASE3_CASES)
    visible_usable = consistent_usable_columns(vis_headers, vis_stats, expected_files=None)

    raw_paths = set(p3_headers.get("path", pd.Series(dtype=str)).astype(str)) | set(
        vis_headers.get("path", pd.Series(dtype=str)).astype(str)
    )
    discrepancy = phase5a_discrepancy_audit(raw_paths)

    phase3_raw_with_direct = int(
        (p3_headers.get("admissible_provenance_columns", pd.Series(dtype=str)).fillna("").astype(str) != "").sum()
    ) if not p3_headers.empty else 0
    visible_raw_with_direct = int(
        (vis_headers.get("admissible_provenance_columns", pd.Series(dtype=str)).fillna("").astype(str) != "").sum()
    ) if not vis_headers.empty else 0

    phase3_authorized = bool(len(phase3_paths) == EXPECTED_PHASE3_CASES and phase3_usable)
    visible_authorized = bool(len(visible_paths) > 0 and visible_usable)
    phase5d_required = bool(phase3_authorized or visible_authorized)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([p3_headers, vis_headers], ignore_index=True).to_csv(OUT / "raw_score_header_audit.csv", index=False)
    pd.concat([p3_stats, vis_stats], ignore_index=True).to_csv(OUT / "cluster_cardinality.csv", index=False)

    decision = {
        "expected_phase3_test_score_files": EXPECTED_PHASE3_CASES,
        "phase3_test_score_files_found": int(len(phase3_paths)),
        "phase3_raw_score_files_with_direct_admissible_provenance": phase3_raw_with_direct,
        "phase3_consistently_usable_provenance_columns": phase3_usable,
        "phase3_cluster_aware_inference_authorized": phase3_authorized,
        "validation_visible_test_score_files_found": int(len(visible_paths)),
        "validation_visible_raw_score_files_with_direct_admissible_provenance": visible_raw_with_direct,
        "validation_visible_consistently_usable_provenance_columns": visible_usable,
        "validation_visible_cluster_aware_inference_authorized": visible_authorized,
        "phase5a_discrepancy_audit": discrepancy,
        "formal_tests_performed": False,
        "resampling_performed": False,
        "new_model_training_performed": False,
        "PHASE5D_REQUIRED": phase5d_required,
        "interpretation": (
            "Only explicit provenance columns on exact raw rejection-score rows can authorize cluster-aware inference. "
            "Derived summaries and positional row_id do not qualify."
        ),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Phase-3 raw test-score files found: {len(phase3_paths)}/{EXPECTED_PHASE3_CASES}")
    print(f"Phase-3 raw score files with direct admissible provenance: {phase3_raw_with_direct}/{len(phase3_paths)}")
    print(f"Phase-3 consistently usable provenance columns: {phase3_usable}")
    print(f"Phase-3 cluster-aware inference authorized: {phase3_authorized}")
    print(f"validation-visible raw test-score files found: {len(visible_paths)}")
    print(f"validation-visible raw score files with direct admissible provenance: {visible_raw_with_direct}/{len(visible_paths)}")
    print(f"validation-visible consistently usable provenance columns: {visible_usable}")
    print(f"validation-visible cluster-aware inference authorized: {visible_authorized}")
    print(f"Phase-5A direct-provenance claim surfaces: {discrepancy['direct_provenance_claim_surfaces']}")
    print(
        "Phase-5A direct-provenance claim surfaces that are exact raw scores: "
        f"{discrepancy['direct_provenance_claim_surfaces_that_are_exact_raw_scores']}"
    )
    print(
        "Phase-5A direct-provenance claim surfaces that are NOT exact raw scores: "
        f"{discrepancy['direct_provenance_claim_surfaces_not_exact_raw_scores']}"
    )
    print(f"audit: {OUT / 'decision.json'}")
    print(f"PHASE5D_REQUIRED={phase5d_required}")
    print("PASS=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
