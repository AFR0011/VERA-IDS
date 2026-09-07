#!/usr/bin/env python3
"""Deep, non-destructive local inventory for the JISA revision.

This second Phase-0 audit searches local-only research surfaces that are not
tracked by the public repository, including the untracked JISA result bundle,
legacy sibling workspace, prepared parquet schemas, and local ZIP archives.
It never trains models or mutates scientific artifacts. Output is written only
under .release-audit/jisa_phase0b/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / ".release-audit" / "jisa_phase0b"

INTERESTING_TOKENS = (
    "test_scores", "val_scores", "selected_methods", "scenario_manifest",
    "aggregate_results", "best_per_holdout", "seed", "split_report",
    "family_support", "eligible_holdout", "manifest", "prediction",
    "confusion", "stage1_threshold", "method_comparison", "run_metadata",
    "reference_profile", "reliability", "open_set", "protocol_b",
)
SCORE_TOKENS = ("test_scores", "val_scores", "score", "prediction")
MODEL_SUFFIXES = (".joblib", ".pkl", ".pickle", ".onnx", ".pt", ".pth")
PROVENANCE_TOKENS = {
    "source_file", "source_file_name", "source_file_path", "source_day",
    "day", "unit_id", "unit_name", "source_unit", "capture", "group_id",
    "identity_group", "segment_index", "segment_count", "row_start", "row_end",
}
SKIP_DIR_NAMES = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache"}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def candidate_roots(extra_roots: list[str]) -> list[Path]:
    roots = [
        REPO_ROOT / "jisa_results_bundle",
        REPO_ROOT / "outputs",
        REPO_ROOT / "processed_V5",
        REPO_ROOT / "processed_V5_cicids17_recovery",
        REPO_ROOT.parent / "Codes",
        REPO_ROOT.parent / "outputs",
    ]
    roots.extend(Path(x).expanduser() for x in extra_roots)
    out, seen = [], set()
    for root in roots:
        try: key = root.resolve()
        except Exception: key = root
        if key in seen: continue
        seen.add(key)
        if root.exists(): out.append(root)
    return out


def walk_files(root: Path):
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIR_NAMES]
        pcur = Path(cur)
        # Raw benchmark folders can contain huge file counts; filenames there are
        # irrelevant to evidence reuse, so do not recursively inventory them.
        low_parts = {p.lower() for p in pcur.parts}
        if "datasets" in low_parts and root == REPO_ROOT:
            dirs[:] = []
            continue
        for name in files:
            yield pcur / name


def categorize(path: Path) -> str | None:
    low = path.name.lower()
    if low.endswith(MODEL_SUFFIXES): return "model"
    if any(t in low for t in SCORE_TOKENS): return "score_or_prediction"
    if any(t in low for t in INTERESTING_TOKENS): return "experiment_metadata"
    return None


def inspect_files(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        for path in walk_files(root):
            cat = categorize(path)
            if not cat: continue
            try: size = path.stat().st_size
            except Exception: size = None
            rows.append({"root": rel(root), "category": cat, "path": rel(path), "bytes": size})
    return sorted(rows, key=lambda r: (r["category"], r["path"]))


def inspect_archives() -> list[dict]:
    rows = []
    for zpath in sorted(REPO_ROOT.glob("*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                for info in zf.infolist():
                    if info.is_dir(): continue
                    member = Path(info.filename)
                    cat = categorize(member)
                    if cat:
                        rows.append({
                            "archive": rel(zpath), "category": cat,
                            "member": info.filename.replace("\\", "/"),
                            "bytes_uncompressed": info.file_size,
                            "bytes_compressed": info.compress_size,
                        })
        except Exception as exc:
            rows.append({"archive": rel(zpath), "category": "archive_error", "member": str(exc)})
    return rows


def parquet_schema(path: Path) -> tuple[list[str], int | None]:
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        return list(pf.schema_arrow.names), int(pf.metadata.num_rows)
    except Exception:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return list(df.columns), len(df)
        except Exception:
            return [], None


def inspect_processed_schemas(roots: list[Path]) -> list[dict]:
    rows = []
    processed_roots = [r for r in roots if "processed" in r.name.lower()]
    for root in processed_roots:
        # Inspect one parquet/csv.gz part per leaf split directory.
        seen_dirs = set()
        for path in walk_files(root):
            low = path.name.lower()
            if not (low.endswith(".parquet") or low.endswith(".csv.gz")): continue
            parent = path.parent.resolve()
            if parent in seen_dirs: continue
            seen_dirs.add(parent)
            cols, nrows = (parquet_schema(path) if low.endswith(".parquet") else ([], None))
            if low.endswith(".csv.gz"):
                try:
                    import gzip
                    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                        cols = next(csv.reader(f), [])
                except Exception: cols = []
            prov = [c for c in cols if c.lower() in PROVENANCE_TOKENS]
            rows.append({
                "processed_root": rel(root), "sample_file": rel(path),
                "leaf_dir": rel(path.parent), "rows_in_sample_part": nrows,
                "n_columns": len(cols), "provenance_columns": "|".join(prov),
                "columns": "|".join(cols),
            })
    return rows


def collect_split_reports(roots: list[Path]) -> dict:
    reports = {}
    for root in roots:
        for path in walk_files(root):
            if path.name.lower() != "split_report.json": continue
            try:
                reports[rel(path)] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                reports[rel(path)] = {"_error": str(exc)}
    return reports


def score_run_pairs(file_rows: list[dict]) -> list[dict]:
    score_paths = [Path(REPO_ROOT / r["path"]) if not Path(r["path"]).is_absolute() else Path(r["path"])
                   for r in file_rows if r["category"] == "score_or_prediction"]
    dirs = {}
    for p in score_paths:
        dirs.setdefault(str(p.parent), set()).add(p.name.lower())
    rows = []
    for d, names in dirs.items():
        if "test_scores.csv.gz" in names or "test_scores.csv" in names:
            rows.append({
                "run_dir": rel(Path(d)),
                "has_test_scores": True,
                "has_val_scores": ("val_scores.csv.gz" in names or "val_scores.csv" in names),
                "has_scenario_manifest": (Path(d) / "scenario_manifest.json").exists(),
                "has_selected_methods": (Path(d) / "selected_methods.csv").exists(),
                "blind_replay_candidate": ("val_scores.csv.gz" in names or "val_scores.csv" in names) and (Path(d) / "scenario_manifest.json").exists(),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--extra-root", action="append", default=[], help="Additional local research/output root to scan; repeatable.")
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    roots = candidate_roots(args.extra_root)
    files = inspect_files(roots)
    archives = inspect_archives()
    schemas = inspect_processed_schemas(roots)
    reports = collect_split_reports(roots)
    pairs = score_run_pairs(files)

    write_csv(out / "interesting_files.csv", files)
    write_csv(out / "archive_members.csv", archives)
    write_csv(out / "processed_schemas.csv", schemas)
    write_csv(out / "score_run_pairs.csv", pairs)
    (out / "split_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    state = {
        "roots_scanned": [rel(r) for r in roots],
        "interesting_files": len(files),
        "score_or_prediction_files": sum(r["category"] == "score_or_prediction" for r in files),
        "model_files": sum(r["category"] == "model" for r in files),
        "archive_interesting_members": len(archives),
        "processed_schema_samples": len(schemas),
        "processed_schema_samples_with_provenance": sum(bool(r["provenance_columns"]) for r in schemas),
        "split_reports": len(reports),
        "test_score_run_pairs": len(pairs),
        "blind_replay_candidates": sum(bool(r["blind_replay_candidate"]) for r in pairs),
    }
    (out / "audit_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("JISA Phase-0b deep local inventory")
    print("roots scanned:")
    for r in state["roots_scanned"]: print(f"  - {r}")
    for key in ["interesting_files", "score_or_prediction_files", "model_files", "archive_interesting_members", "processed_schema_samples", "processed_schema_samples_with_provenance", "split_reports", "test_score_run_pairs", "blind_replay_candidates"]:
        print(f"{key}: {state[key]}")
    print(f"audit output: {out}")
    print("No scientific outputs were modified.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
