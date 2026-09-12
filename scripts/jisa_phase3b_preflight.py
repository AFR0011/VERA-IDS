#!/usr/bin/env python3
"""Preflight Phase-3B rejector replay and paired validation-visible evidence.

This audit is read-only with respect to scientific outputs. It verifies the frozen
30-case Phase-3A score surface and searches known local/legacy research roots for
validation-visible ``val_scores.csv.gz`` + ``test_scores.csv.gz`` pairs that can be
matched by CICIoT2023 seed, held-out family, and selected model profile.

The purpose is to prevent an unmatched historical rejector summary from being
presented as a paired visible-vs-blind comparison. Audit outputs are written only
under ``.release-audit/jisa_phase3b_preflight/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE3_ROOT = REPO_ROOT / "outputs" / "13_jisa_q1_revision" / "phase3_validation_blind" / "score_generation"
AUDIT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase3b_preflight"
DATASET = "CICIoT2023"
SEEDS = [123, 124, 125, 126, 127]
HOLDOUTS = ["Botnet", "BruteForce", "DDoS", "DoS", "Other", "Scan/Recon"]

SEARCH_ROOTS = [
    REPO_ROOT / "outputs" / "06_open_set_rejection",
    REPO_ROOT / "outputs" / "10_seed_reliability",
    REPO_ROOT / "outputs" / "11_reference_framework_eval",
    REPO_ROOT / "outputs" / "12_jisa_finalization",
    REPO_ROOT / "jisa_results_bundle",
    REPO_ROOT.parent / "Codes",
    REPO_ROOT.parent / "outputs",
]
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".release-audit"}


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def recursive_values(obj: Any, key_names: set[str]) -> list[Any]:
    out: list[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in key_names:
                out.append(v)
            out.extend(recursive_values(v, key_names))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(recursive_values(v, key_names))
    return out


def first_scalar(values: Iterable[Any]) -> str:
    for value in values:
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            return str(value).strip()
    return ""


def infer_seed(path: Path, manifest: dict[str, Any]) -> int | None:
    vals = recursive_values(manifest, {"seed", "random_seed", "global_seed"})
    for raw in vals:
        try:
            seed = int(raw)
            if seed in SEEDS:
                return seed
        except Exception:
            pass
    text = str(path).replace("\\", "/")
    for pattern in [r"seed[_-]?(123|124|125|126|127)(?:\D|$)", r"(?:^|[/_.-])(123|124|125|126|127)(?:[/_.-]|$)"]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def infer_holdout(path: Path, manifest: dict[str, Any]) -> str:
    direct = first_scalar(recursive_values(manifest, {"holdout_family", "holdout"}))
    if direct in HOLDOUTS:
        return direct
    text = str(path).replace("\\", "/").lower()
    aliases = {
        "botnet": "Botnet",
        "bruteforce": "BruteForce",
        "ddos": "DDoS",
        "dos": "DoS",
        "other": "Other",
        "scan_recon": "Scan/Recon",
        "scan-recon": "Scan/Recon",
        "scan/recon": "Scan/Recon",
        "scanrecon": "Scan/Recon",
    }
    # longest/specific aliases first so DDoS is not mistaken for DoS
    for token in ["bruteforce", "scan_recon", "scan-recon", "scanrecon", "botnet", "ddos", "other", "dos"]:
        if f"holdout_{token}" in text or f"holdout-{token}" in text:
            return aliases[token]
    return ""


def infer_dataset(manifest: dict[str, Any], path: Path) -> str:
    direct = first_scalar(recursive_values(manifest, {"dataset"}))
    if direct:
        return direct
    text = str(path).lower()
    return DATASET if "ciciot2023" in text or "ciciot" in text else ""


def infer_profile(run_dir: Path, manifest: dict[str, Any]) -> str:
    direct = first_scalar(recursive_values(manifest, {"selected_candidate_label", "model_profile"}))
    if direct and direct.lower() != "nan":
        return direct
    family = first_scalar(recursive_values(manifest, {"model_family"})).lower()
    weight = first_scalar(recursive_values(manifest, {"stage1_weight_mode"})).lower()
    if family and weight:
        return f"{family}_{weight}"
    selected_methods = run_dir / "selected_methods.csv"
    if selected_methods.exists():
        try:
            df = pd.read_csv(selected_methods, nrows=10)
            if "model_profile" in df.columns:
                vals = df["model_profile"].dropna().astype(str).unique().tolist()
                if len(vals) == 1:
                    return vals[0]
            if {"model_family", "stage1_weight_mode"}.issubset(df.columns):
                vals = (df["model_family"].astype(str) + "_" + df["stage1_weight_mode"].astype(str)).unique().tolist()
                if len(vals) == 1:
                    return vals[0]
        except Exception:
            pass
    text = str(run_dir).lower()
    for label in ["rf_class_weight_balanced", "rf_inv_family_clipped", "xgb_inv_family_clipped"]:
        if label in text:
            return label
    return ""


def expected_phase3_profiles() -> tuple[dict[tuple[int, str], str], bool]:
    expected: dict[tuple[int, str], str] = {}
    complete = True
    for seed in SEEDS:
        for holdout in HOLDOUTS:
            case_dir = PHASE3_ROOT / f"seed_{seed}" / holdout.replace("/", "_")
            selected = case_dir / "selected_profile.json"
            val_scores = case_dir / "val_known_scores.csv.gz"
            test_scores = case_dir / "test_scores.csv.gz"
            if not selected.exists() or not val_scores.exists() or not test_scores.exists():
                complete = False
                continue
            payload = read_json(selected)
            label = str(payload.get("selected_candidate_label", "")).strip()
            if not label:
                complete = False
                continue
            expected[(seed, holdout)] = label
    return expected, complete and len(expected) == len(SEEDS) * len(HOLDOUTS)


def iter_visible_val_files(root: Path):
    if not root.exists():
        return
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
        if "val_scores.csv.gz" in files:
            yield Path(cur) / "val_scores.csv.gz"
        elif "val_scores.csv" in files:
            yield Path(cur) / "val_scores.csv"


def val_has_unknown(path: Path) -> bool | None:
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return None
    if "is_true_unknown" not in header:
        return None
    try:
        for chunk in pd.read_csv(path, usecols=["is_true_unknown"], chunksize=250_000):
            vals = pd.to_numeric(chunk["is_true_unknown"], errors="coerce").fillna(0)
            if bool((vals == 1).any()):
                return True
        return False
    except Exception:
        return None


def inspect_visible_candidates(extra_roots: list[str], expected: dict[tuple[int, str], str]) -> pd.DataFrame:
    roots = list(SEARCH_ROOTS) + [Path(x).expanduser() for x in extra_roots]
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for val_path in iter_visible_val_files(root):
            try:
                run_dir = val_path.parent.resolve()
            except Exception:
                run_dir = val_path.parent
            key = str(run_dir).lower()
            if key in seen:
                continue
            seen.add(key)
            test_path = run_dir / ("test_scores.csv.gz" if (run_dir / "test_scores.csv.gz").exists() else "test_scores.csv")
            manifest_path = run_dir / "scenario_manifest.json"
            manifest = read_json(manifest_path) if manifest_path.exists() else {}
            dataset = infer_dataset(manifest, run_dir)
            holdout = infer_holdout(run_dir, manifest)
            seed = infer_seed(run_dir, manifest)
            profile = infer_profile(run_dir, manifest)
            expected_profile = expected.get((seed, holdout), "") if seed is not None and holdout else ""
            profile_exact = bool(expected_profile and (profile == expected_profile or expected_profile.lower() in str(run_dir).lower()))
            row = {
                "root": str(root.resolve()),
                "run_dir": str(run_dir),
                "val_scores": str(val_path),
                "test_scores": str(test_path) if test_path.exists() else "",
                "has_test_scores": test_path.exists(),
                "has_manifest": manifest_path.exists(),
                "dataset": dataset,
                "seed": seed,
                "holdout_family": holdout,
                "profile": profile,
                "expected_phase3_profile": expected_profile,
                "profile_exact": profile_exact,
                "seed_holdout_match": bool(dataset == DATASET and seed in SEEDS and holdout in HOLDOUTS),
                "val_has_true_unknown": None,
            }
            if row["seed_holdout_match"] and row["has_test_scores"]:
                row["val_has_true_unknown"] = val_has_unknown(val_path)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-root", action="append", default=[], help="Additional local root to scan; repeatable.")
    args = parser.parse_args()

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    expected, phase3_ready = expected_phase3_profiles()
    visible = inspect_visible_candidates(args.extra_root, expected)
    if visible.empty:
        visible.to_csv(AUDIT_ROOT / "visible_score_candidates.csv", index=False)
    else:
        visible.sort_values(["seed", "holdout_family", "run_dir"], na_position="last").to_csv(
            AUDIT_ROOT / "visible_score_candidates.csv", index=False
        )

    seed_holdout = visible.loc[
        visible.get("seed_holdout_match", pd.Series(dtype=bool)).fillna(False)
        & visible.get("has_test_scores", pd.Series(dtype=bool)).fillna(False)
    ].copy() if not visible.empty else pd.DataFrame()
    paired = seed_holdout.loc[
        seed_holdout.get("profile_exact", pd.Series(dtype=bool)).fillna(False)
        & (seed_holdout.get("val_has_true_unknown", pd.Series(dtype=object)) == True)  # noqa: E712
    ].copy() if not seed_holdout.empty else pd.DataFrame()

    matched_keys = set()
    if not paired.empty:
        matched_keys = {(int(r.seed), str(r.holdout_family)) for r in paired.itertuples() if pd.notna(r.seed)}

    matrix_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for holdout in HOLDOUTS:
            subset = paired[(paired["seed"] == seed) & (paired["holdout_family"] == holdout)] if not paired.empty else pd.DataFrame()
            matrix_rows.append(
                {
                    "seed": seed,
                    "holdout_family": holdout,
                    "expected_profile": expected.get((seed, holdout), ""),
                    "paired_visible_candidates": int(len(subset)),
                    "paired_visible_ready": bool(len(subset) > 0),
                }
            )
    matrix = pd.DataFrame(matrix_rows)
    matrix.to_csv(AUDIT_ROOT / "paired_coverage_matrix.csv", index=False)

    holdout_coverage = (
        matrix.groupby("holdout_family", sort=True)["paired_visible_ready"].agg(["sum", "count"]).reset_index()
        if not matrix.empty else pd.DataFrame()
    )
    if not holdout_coverage.empty:
        holdout_coverage.to_csv(AUDIT_ROOT / "holdout_coverage.csv", index=False)

    pair_ready = bool(phase3_ready and len(matched_keys) == 30)
    payload = {
        "phase3a_score_surface_ready": phase3_ready,
        "visible_candidate_dirs_examined": int(len(visible)),
        "seed_holdout_visible_pairs": int(len(seed_holdout)),
        "profile_and_unknown_matched_pairs": int(len(paired)),
        "unique_seed_holdout_pairs_ready": int(len(matched_keys)),
        "paired_visible_matrix_ready": pair_ready,
    }
    (AUDIT_ROOT / "preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Phase-3A frozen score cases ready: {30 if phase3_ready else len(expected)}/30")
    print(f"visible score candidate dirs examined: {len(visible)}")
    print(f"visible candidates with seed+holdout+test: {len(seed_holdout)}")
    print(f"profile-matched candidates with true-Unknown validation: {len(paired)}")
    print(f"unique paired seed+holdout cells ready: {len(matched_keys)}/30")
    if not holdout_coverage.empty:
        print(holdout_coverage.to_string(index=False))
    print(f"audit: {AUDIT_ROOT / 'preflight.json'}")
    print(f"PAIRED_VISIBLE_READY={pair_ready}")
    return 0 if phase3_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
