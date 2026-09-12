#!/usr/bin/env python3
"""Read-only preflight for rebuilding validation-visible rejector comparisons on a common case set.

This is a reporting/provenance cleanup, not a new statistical test. The script searches
retained local outputs for surfaces that explicitly encode rejector method, rejection
budget (or an equivalent constraint field), feasibility, dataset/holdout identity, and
claim-bearing test metrics. It performs no model fitting, threshold tuning, resampling,
or hypothesis testing.

Outputs are written only under `.release-audit/jisa_phase5e_visible_common_case/`.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / ".release-audit" / "jisa_phase5e_visible_common_case"

SEARCH_ROOTS = [
    REPO_ROOT / "outputs" / "06_open_set_rejection",
    REPO_ROOT / "outputs" / "08_statistics",
    REPO_ROOT / "outputs" / "12_jisa_finalization",
    REPO_ROOT / "outputs" / "summaries",
]

METHOD_ALIASES = {"method", "rejector", "rejector_method"}
BUDGET_TERMS = {"budget", "rejection_budget", "overall_reject_budget", "max_overall_reject_rate"}
FEASIBLE_ALIASES = {"feasible", "ok", "is_feasible", "valid"}
DATASET_ALIASES = {"dataset", "dataset_name"}
HOLDOUT_ALIASES = {"holdout_family", "holdout", "left_out_family", "unknown_family"}
METRIC_TERMS = [
    "unknown_detection", "macro_f1", "false_unknown", "overall_reject", "benign_reject",
    "benign_family_fp", "accuracy",
]
PRIMARY_BUDGET = 0.03
TOL = 1e-12
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".release-audit"}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def read_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return [str(x).strip() for x in next(csv.reader(f))]
    except Exception:
        return []


def first_alias(columns: list[str], aliases: set[str]) -> str:
    lookup = {c.lower(): c for c in columns}
    for alias in sorted(aliases):
        if alias in lookup:
            return lookup[alias]
    return ""


def budget_col(columns: list[str]) -> str:
    lookup = {c.lower(): c for c in columns}
    for alias in sorted(BUDGET_TERMS):
        if alias in lookup:
            return lookup[alias]
    for c in columns:
        low = c.lower()
        if "budget" in low and "bootstrap" not in low:
            return c
        if "max" in low and "reject" in low and "rate" in low:
            return c
    return ""


def metric_cols(columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        low = c.lower()
        if any(term in low for term in METRIC_TERMS):
            out.append(c)
    return sorted(set(out))


def inspect_csv(path: Path) -> dict[str, Any] | None:
    cols = read_header(path)
    if not cols:
        return None

    method = first_alias(cols, METHOD_ALIASES)
    budget = budget_col(cols)
    feasible = first_alias(cols, FEASIBLE_ALIASES)
    dataset = first_alias(cols, DATASET_ALIASES)
    holdout = first_alias(cols, HOLDOUT_ALIASES)
    metrics = metric_cols(cols)

    relevant = bool(method or metrics or "reject" in path.name.lower() or "tradeoff" in str(path).lower())
    if not relevant:
        return None

    primary_rows = 0
    n_rows = None
    read_error = ""
    if budget:
        try:
            usecols = [budget]
            df = pd.read_csv(path, usecols=usecols)
            n_rows = int(len(df))
            vals = pd.to_numeric(df[budget], errors="coerce")
            primary_rows = int(((vals - PRIMARY_BUDGET).abs() <= TOL).sum())
        except Exception as exc:
            read_error = f"{type(exc).__name__}: {exc}"
    else:
        try:
            n_rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        except Exception:
            n_rows = None

    explicit_common_case_ready = bool(method and budget and feasible and dataset and holdout and metrics)
    descriptive_case_surface = bool(method and dataset and holdout and metrics)

    return {
        "path": rel(path),
        "n_rows": n_rows,
        "method_column": method,
        "budget_column": budget,
        "feasible_column": feasible,
        "dataset_column": dataset,
        "holdout_column": holdout,
        "metric_columns": "|".join(metrics),
        "primary_3pct_rows": primary_rows,
        "explicit_common_case_ready": explicit_common_case_ready,
        "descriptive_case_surface": descriptive_case_surface,
        "read_error": read_error,
        "columns": "|".join(cols),
    }


def walk_csvs() -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(path)
    return sorted(out)


def inspect_json_budget_manifests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            low = text.lower()
            if "budget" not in low and "reject" not in low:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            flat = json.dumps(obj, sort_keys=True).lower()
            if any(k in flat for k in ["budget", "max_overall_reject_rate", "overall_reject"]):
                rows.append({"path": rel(path), "size_bytes": path.stat().st_size})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    for path in walk_csvs():
        rec = inspect_csv(path)
        if rec is not None:
            csv_rows.append(rec)

    csv_df = pd.DataFrame(csv_rows)
    if not csv_df.empty:
        csv_df = csv_df.sort_values(
            ["explicit_common_case_ready", "primary_3pct_rows", "descriptive_case_surface", "path"],
            ascending=[False, False, False, True],
        )
    manifests = pd.DataFrame(inspect_json_budget_manifests())

    csv_df.to_csv(OUT / "candidate_surfaces.csv", index=False)
    manifests.to_csv(OUT / "budget_manifest_candidates.csv", index=False)

    explicit = csv_df[csv_df["explicit_common_case_ready"] == True].copy() if not csv_df.empty else pd.DataFrame()  # noqa: E712
    explicit_primary = explicit[explicit["primary_3pct_rows"] > 0].copy() if not explicit.empty else pd.DataFrame()
    descriptive = csv_df[csv_df["descriptive_case_surface"] == True].copy() if not csv_df.empty else pd.DataFrame()  # noqa: E712

    ready = bool(len(explicit_primary) > 0)
    decision = {
        "candidate_csv_surfaces": int(len(csv_df)),
        "explicit_common_case_ready_surfaces": int(len(explicit)),
        "explicit_surfaces_with_primary_3pct_rows": int(len(explicit_primary)),
        "descriptive_case_surfaces": int(len(descriptive)),
        "budget_manifest_candidates": int(len(manifests)),
        "new_model_training": False,
        "new_threshold_selection": False,
        "new_statistical_tests": False,
        "READY_TO_BUILD_VISIBLE_COMMON_CASE": ready,
        "fallback_if_false": (
            "Do not retain unequal-feasible-subset method means as if directly comparable. "
            "Use case-level results only, or remove/relabel the aggregate method comparison."
        ),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    print(f"candidate CSV surfaces: {len(csv_df)}")
    print(f"explicit common-case-ready surfaces: {len(explicit)}")
    print(f"explicit surfaces with primary 3% rows: {len(explicit_primary)}")
    for _, row in explicit_primary.head(20).iterrows():
        print(
            f"  ready: {row['path']} primary_rows={int(row['primary_3pct_rows'])} "
            f"budget={row['budget_column']} feasible={row['feasible_column']} metrics={row['metric_columns']}"
        )
    print(f"descriptive case surfaces: {len(descriptive)}")
    print(f"budget-manifest candidates: {len(manifests)}")
    print(f"audit: {OUT / 'decision.json'}")
    print(f"READY_TO_BUILD_VISIBLE_COMMON_CASE={ready}")
    print("PASS=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
