#!/usr/bin/env python3
"""Local-only Phase-0 evidence inventory for the JISA revision.

This script is deliberately non-destructive. It does not train models, rewrite
results, or copy row-level artifacts into tracked paths. Outputs are written only
under `.release-audit/jisa_phase0/`, which is ignored by Git.

The goal is to answer four questions before any new experiment is run:
1. Which historical/public result surfaces are available and internally hashed?
2. Which local heavy outputs still exist and can be reused without retraining?
3. Which validation/test score pairs can support blind rejector replay?
4. Which local provenance artifacts may support cluster-aware resampling?
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / ".release-audit" / "jisa_phase0"

TRACKED_SUMMARIES = [
    "outputs/summaries/protocol_a_core_summary.csv",
    "outputs/summaries/protocol_a_flat_vs_two_stage.csv",
    "outputs/summaries/protocol_b_best_per_holdout.csv",
    "outputs/summaries/protocol_b_holdout_confidence_intervals_1000.csv",
    "outputs/summaries/protocol_b_paired_method_tests_1000.csv",
    "outputs/summaries/open_set_baseline_comparison.csv",
    "outputs/summaries/seed_reliability_summary.csv",
    "outputs/summaries/sink_aware_comparison.csv",
    "outputs/summaries/validation_selected_sink_summary.csv",
    "outputs/summaries/reference_profile_metric_drop.csv",
    "outputs/summaries/external_protocol_a_summary.csv",
    "outputs/summaries/support_threshold_selection.json",
    "outputs/summaries/support_threshold_sensitivity_by_holdout.csv",
    "outputs/summaries/support_threshold_sensitivity_summary.csv",
    "outputs/evidence/protocol_a_confusion_matrices.jsonl",
]

LOCAL_SURFACE_CANDIDATES = {
    "processed_v5": [
        "outputs/02_prepared_data/processed_V5",
        "processed_V5",
    ],
    "processed_cicids_recovery": [
        "outputs/02_prepared_data/processed_V5_cicids17_recovery",
        "processed_V5_cicids17_recovery",
    ],
    "protocol_b_support": ["outputs/04_protocol_b_support_audit"],
    "protocol_b_loao": ["outputs/05_protocol_b_loao"],
    "open_set": ["outputs/06_open_set_rejection"],
    "statistics": ["outputs/08_statistics"],
    "seed_reliability": ["outputs/10_seed_reliability"],
    "reference_profiles": ["outputs/11_reference_framework_eval"],
}

RAW_DATA_CANDIDATES = {
    "CICIDS2017": [
        "Datasets/CICIDS 2017",
        "Datasets/CICIDS2017",
        "data/raw/CICIDS2017",
    ],
    "CICIoT2023": [
        "Datasets/CIC IoT Dataset 2023",
        "Datasets/CICIoT2023",
        "data/raw/CICIoT2023",
    ],
    "NSL-KDD": ["Datasets/NSL-KDD", "data/raw/NSL-KDD"],
    "UNSW-NB15": ["Datasets/UNSW-NB15", "data/raw/UNSW-NB15"],
}

SCORE_SEARCH_ROOTS = [
    "outputs/05_protocol_b_loao",
    "outputs/06_open_set_rejection",
    "outputs/10_seed_reliability",
    "outputs/11_reference_framework_eval",
]

PROVENANCE_COLUMN_TOKENS = {
    "source_file",
    "source_file_name",
    "source_day",
    "day",
    "unit_id",
    "unit_name",
    "source_unit",
    "capture",
    "group_id",
    "identity_group",
    "segment_index",
    "row_start",
    "row_end",
}

PROVENANCE_FILENAME_HINTS = (
    "split_report",
    "split_manifest",
    "unit",
    "source",
    "allocation",
    "partition",
    "provenance",
    "family_support",
    "eligible_holdout",
)


@dataclass
class TrackedArtifact:
    path: str
    exists: bool
    bytes: int | None
    sha256: str | None
    rows: int | None
    columns: int | None
    header: str | None


@dataclass
class LocalSurface:
    surface: str
    path: str
    exists: bool
    top_level_entries: int | None


@dataclass
class ScoreCandidate:
    run_dir: str
    test_scores: str
    val_scores_exists: bool
    scenario_manifest_exists: bool
    selected_methods_exists: bool
    test_bytes: int
    test_columns: int | None
    provenance_columns: str
    blind_replay_candidate: bool


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> list[str]:
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
                return next(csv.reader(handle), [])
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            return next(csv.reader(handle), [])
    except Exception:
        return []


def count_text_rows(path: Path) -> int | None:
    if path.suffix.lower() not in {".csv", ".jsonl"}:
        return None
    try:
        with path.open("rb") as handle:
            n = sum(1 for _ in handle)
        return max(0, n - (1 if path.suffix.lower() == ".csv" else 0))
    except Exception:
        return None


def git_value(*args: str) -> str:
    try:
        cp = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return cp.stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def top_level_count(path: Path) -> int | None:
    try:
        return sum(1 for _ in path.iterdir())
    except Exception:
        return None


def inspect_tracked_artifacts() -> list[TrackedArtifact]:
    rows: list[TrackedArtifact] = []
    for item in TRACKED_SUMMARIES:
        path = REPO_ROOT / item
        if not path.exists():
            rows.append(TrackedArtifact(item, False, None, None, None, None, None))
            continue
        header = read_header(path) if path.suffix.lower() == ".csv" else []
        rows.append(
            TrackedArtifact(
                path=item,
                exists=True,
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
                rows=count_text_rows(path),
                columns=len(header) if header else None,
                header="|".join(header) if header else None,
            )
        )
    return rows


def inspect_local_surfaces() -> list[LocalSurface]:
    rows: list[LocalSurface] = []
    for surface, candidates in LOCAL_SURFACE_CANDIDATES.items():
        for candidate in candidates:
            path = REPO_ROOT / candidate
            rows.append(
                LocalSurface(
                    surface=surface,
                    path=candidate,
                    exists=path.exists(),
                    top_level_entries=top_level_count(path) if path.exists() else None,
                )
            )
    for dataset, candidates in RAW_DATA_CANDIDATES.items():
        for candidate in candidates:
            path = REPO_ROOT / candidate
            rows.append(
                LocalSurface(
                    surface=f"raw::{dataset}",
                    path=candidate,
                    exists=path.exists(),
                    top_level_entries=top_level_count(path) if path.exists() else None,
                )
            )
    return rows


def iter_score_files() -> Iterator[Path]:
    seen: set[Path] = set()
    for root_rel in SCORE_SEARCH_ROOTS:
        root = REPO_ROOT / root_rel
        if not root.exists():
            continue
        for path in root.rglob("test_scores.csv.gz"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def inspect_score_candidates() -> list[ScoreCandidate]:
    rows: list[ScoreCandidate] = []
    for test_path in iter_score_files():
        run_dir = test_path.parent
        val_path = run_dir / "val_scores.csv.gz"
        manifest_path = run_dir / "scenario_manifest.json"
        selected_path = run_dir / "selected_methods.csv"
        header = read_header(test_path)
        provenance = sorted({c for c in header if c.lower() in PROVENANCE_COLUMN_TOKENS})
        rows.append(
            ScoreCandidate(
                run_dir=rel(run_dir),
                test_scores=rel(test_path),
                val_scores_exists=val_path.exists(),
                scenario_manifest_exists=manifest_path.exists(),
                selected_methods_exists=selected_path.exists(),
                test_bytes=test_path.stat().st_size,
                test_columns=len(header) if header else None,
                provenance_columns="|".join(provenance),
                blind_replay_candidate=bool(val_path.exists() and manifest_path.exists()),
            )
        )
    return sorted(rows, key=lambda row: row.run_dir)


def iter_provenance_candidates() -> Iterator[Path]:
    roots: list[Path] = []
    for candidates in LOCAL_SURFACE_CANDIDATES.values():
        for candidate in candidates:
            path = REPO_ROOT / candidate
            if path.exists():
                roots.append(path)
    seen: set[Path] = set()
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if any(hint in name for hint in PROVENANCE_FILENAME_HINTS):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield path


def provenance_records(limit: int = 5000) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, path in enumerate(iter_provenance_candidates()):
        if idx >= limit:
            rows.append(
                {
                    "path": "TRUNCATED",
                    "bytes": None,
                    "columns": None,
                    "provenance_columns": None,
                }
            )
            break
        header = read_header(path) if (path.name.endswith(".csv") or path.name.endswith(".csv.gz")) else []
        provenance = sorted({c for c in header if c.lower() in PROVENANCE_COLUMN_TOKENS})
        rows.append(
            {
                "path": rel(path),
                "bytes": path.stat().st_size,
                "columns": len(header) if header else None,
                "provenance_columns": "|".join(provenance),
            }
        )
    return sorted(rows, key=lambda row: str(row["path"]))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in records:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    tracked = inspect_tracked_artifacts()
    surfaces = inspect_local_surfaces()
    scores = inspect_score_candidates()
    provenance = provenance_records()

    write_csv(out / "tracked_artifacts.csv", (asdict(x) for x in tracked))
    write_csv(out / "local_surfaces.csv", (asdict(x) for x in surfaces))
    write_csv(out / "score_reuse_candidates.csv", (asdict(x) for x in scores))
    write_csv(out / "provenance_candidates.csv", provenance)

    state = {
        "repo_root": str(REPO_ROOT),
        "branch": git_value("branch", "--show-current"),
        "head": git_value("rev-parse", "HEAD"),
        "status_porcelain": git_value("status", "--porcelain"),
        "python": sys.version,
        "tracked_artifacts": len(tracked),
        "tracked_artifacts_present": sum(x.exists for x in tracked),
        "local_surface_candidates_present": sum(x.exists for x in surfaces),
        "score_candidates": len(scores),
        "blind_replay_candidates": sum(x.blind_replay_candidate for x in scores),
        "score_candidates_with_provenance_columns": sum(bool(x.provenance_columns) for x in scores),
        "provenance_candidate_files": len(provenance),
    }
    (out / "audit_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    print("JISA Phase-0 local evidence audit")
    print(f"repo: {REPO_ROOT}")
    print(f"branch: {state['branch']}")
    print(f"head: {state['head']}")
    print(f"tracked compact artifacts present: {state['tracked_artifacts_present']}/{state['tracked_artifacts']}")
    print(f"local surface candidates present: {state['local_surface_candidates_present']}")
    print(f"test-score candidates found: {state['score_candidates']}")
    print(f"blind replay candidates: {state['blind_replay_candidates']}")
    print(f"score files already carrying provenance columns: {state['score_candidates_with_provenance_columns']}")
    print(f"provenance candidate files: {state['provenance_candidate_files']}")
    print(f"audit output: {out}")
    print("No scientific outputs were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
