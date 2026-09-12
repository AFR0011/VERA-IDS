#!/usr/bin/env python3
"""Read-only preflight for the JISA Phase-4 CICIDS2017 partition-validity analysis.

The script inventories local evidence for the recovered CICIDS2017 Protocol-B surface
and any group-safe / exact-identity-isolated sensitivity surface. It performs no model
training and writes audit outputs only under `.release-audit/`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase4_partition_preflight"

RECOVERED_KNOWN = [
    REPO_ROOT / "outputs" / "summaries" / "protocol_b_best_per_holdout.csv",
    REPO_ROOT / "outputs" / "05_protocol_b_loao" / "protocolB_grid_runs step 3 - CICIDS2017 sweep" / "summary" / "best_per_holdout.csv",
    REPO_ROOT / "outputs" / "05_protocol_b_loao" / "protocolB_grid_runs step 3 - CICIDS2017 sweep" / "aggregate_results.csv",
]

KEYWORDS = {
    "group_safe": ["group_safe", "group-safe", "groupsafe", "identity_isolated", "identity-isolated"],
    "overlap": ["exact overlap", "cross-partition overlap", "representation overlap", "identity overlap"],
    "duplicate": ["duplicate", "dedup", "exact identity", "exact-identity"],
    "partition": ["partition sensitivity", "partition_validity", "partition-validity", "split sensitivity"],
}

TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yml", ".yaml", ".log"}
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__"}
MAX_TEXT_BYTES = 25 * 1024 * 1024


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def path_score(path: Path) -> tuple[int, list[str]]:
    text = str(path).lower()
    score = 0
    hits: list[str] = []
    if "cicids" in text:
        score += 3
        hits.append("cicids")
    if "protocol" in text and "b" in text:
        score += 1
    for label, terms in KEYWORDS.items():
        if any(term in text for term in terms):
            score += 4
            hits.append(label)
    if "recovery" in text or "recovered" in text:
        score += 2
        hits.append("recovered")
    return score, sorted(set(hits))


def scan_text(path: Path) -> tuple[int, list[str]]:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return 0, []
        text = path.read_text(encoding="utf-8", errors="ignore")[:1_000_000].lower()
    except Exception:
        return 0, []
    score = 0
    hits: list[str] = []
    if "cicids2017" in text or "cicids 2017" in text:
        score += 2
        hits.append("cicids_content")
    for label, terms in KEYWORDS.items():
        if any(term in text for term in terms):
            score += 3
            hits.append(label)
    return score, sorted(set(hits))


def candidate_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        pscore, phits = path_score(path)
        if pscore == 0:
            # Avoid opening arbitrary local files unless the path itself is plausibly relevant.
            continue
        cscore, chits = scan_text(path)
        score = pscore + cscore
        if score < 4:
            continue
        rows.append(
            {
                "path": rel(path),
                "size_bytes": path.stat().st_size,
                "score": score,
                "path_hits": "|".join(phits),
                "content_hits": "|".join(chits),
            }
        )
    rows.sort(key=lambda r: (-int(r["score"]), str(r["path"])))
    return rows


def inspect_result_csv(path: Path) -> dict[str, Any] | None:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    cols = list(header.columns)
    needed = {"holdout_family", "unknown_detection_rate"}
    if not needed.issubset(cols):
        return None
    usecols = [c for c in ["dataset", "split_variant", "holdout_family", "unknown_detection_rate", "macro_f1", "accuracy", "model_family", "run_name"] if c in cols]
    try:
        df = pd.read_csv(path, usecols=usecols)
    except Exception:
        return None
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == "CICIDS2017"].copy()
    if df.empty:
        return None
    return {
        "path": rel(path),
        "columns": cols,
        "n_rows_cicids": int(len(df)),
        "holdouts": sorted(set(df["holdout_family"].astype(str))),
        "split_variants": sorted(set(df["split_variant"].dropna().astype(str))) if "split_variant" in df.columns else [],
        "preview": df.head(20).to_dict(orient="records"),
    }


def inspect_recovered() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in RECOVERED_KNOWN:
        if path.exists():
            rec = inspect_result_csv(path)
            if rec is not None:
                out.append(rec)
    return out


def inspect_candidate_result_csvs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for row in candidates:
        path = REPO_ROOT / str(row["path"])
        if path.suffix.lower() != ".csv" or path in seen:
            continue
        seen.add(path)
        rec = inspect_result_csv(path)
        if rec is not None:
            out.append(rec)
    return out


def main() -> int:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    candidates = candidate_files()
    recovered = inspect_recovered()
    result_csvs = inspect_candidate_result_csvs(candidates)

    pd.DataFrame(candidates).to_csv(AUDIT_ROOT / "candidate_artifacts.csv", index=False)
    (AUDIT_ROOT / "recovered_result_surfaces.json").write_text(json.dumps(recovered, indent=2), encoding="utf-8")
    (AUDIT_ROOT / "candidate_result_surfaces.json").write_text(json.dumps(result_csvs, indent=2), encoding="utf-8")

    likely_group_safe = [
        r for r in candidates
        if "group_safe" in str(r.get("path_hits", ""))
        or "group_safe" in str(r.get("content_hits", ""))
        or "overlap" in str(r.get("content_hits", ""))
    ]
    group_result_surfaces = []
    group_paths = {str(r["path"]) for r in likely_group_safe}
    for rec in result_csvs:
        if rec["path"] in group_paths:
            group_result_surfaces.append(rec)

    recovered_ready = any(len(r.get("holdouts", [])) >= 6 for r in recovered)
    group_result_ready = any(len(r.get("holdouts", [])) >= 6 for r in group_result_surfaces)
    group_metadata_ready = len(likely_group_safe) > 0

    payload = {
        "recovered_surface_ready": recovered_ready,
        "group_safe_candidate_artifacts": len(likely_group_safe),
        "group_safe_result_surface_ready": group_result_ready,
        "candidate_artifacts_total": len(candidates),
        "candidate_result_surfaces_total": len(result_csvs),
        "new_training_performed": False,
        "READY_FOR_PHASE4_BUILD": bool(recovered_ready and group_result_ready and group_metadata_ready),
    }
    (AUDIT_ROOT / "preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"recovered CICIDS result surface ready: {recovered_ready}")
    for rec in recovered:
        print(f"  recovered: {rec['path']} holdouts={len(rec['holdouts'])} variants={rec['split_variants']}")
    print(f"candidate partition/overlap artifacts: {len(candidates)}")
    print(f"likely group-safe/identity-overlap artifacts: {len(likely_group_safe)}")
    for row in likely_group_safe[:25]:
        print(f"  candidate: {row['path']} score={row['score']} hits={row['path_hits']}|{row['content_hits']}")
    if len(likely_group_safe) > 25:
        print(f"  ... {len(likely_group_safe) - 25} more")
    print(f"candidate CICIDS result CSV surfaces: {len(result_csvs)}")
    for rec in result_csvs:
        print(f"  result: {rec['path']} holdouts={len(rec['holdouts'])} variants={rec['split_variants']}")
    print(f"group-safe result surface ready: {group_result_ready}")
    print(f"audit: {AUDIT_ROOT / 'preflight.json'}")
    print(f"READY_FOR_PHASE4_BUILD={payload['READY_FOR_PHASE4_BUILD']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
