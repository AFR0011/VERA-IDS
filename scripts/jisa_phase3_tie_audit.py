#!/usr/bin/env python3
"""Classify Phase-3 profile-selection ties without touching scientific outputs.

The first Phase-3 preflight intentionally asked a strict question: can every historical
winner be reused without any heldout-visible tie-break? The answer can be no even when
the ambiguity is much narrower than the full three-profile grid. This follow-up audit
uses the already-discovered five seed aggregates and identifies the *top equivalence
set* under the blind-safe primary criterion, Stage-2 validation macro-F1.

Only candidates tied at the maximum Stage-2 validation macro-F1 can require a new
heldout-free Stage-1 tie-break. Candidates strictly below that maximum are dominated
under the frozen lexicographic profile-selection rule and need not be rerun merely to
resolve the tie.

Read-only historical input; writes only under .release-audit/jisa_phase3_tie_audit/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_JSON = REPO_ROOT / ".release-audit" / "jisa_phase3_preflight" / "preflight.json"
OUT = REPO_ROOT / ".release-audit" / "jisa_phase3_tie_audit"
DATASET = "CICIoT2023"
SEEDS = [123, 124, 125, 126, 127]
TOL = 1e-12


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def candidate_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "model_profile", "model_family", "apply_loao_stage1",
        "stage1_weight_mode", "stage2_weight_mode", "stage1_params", "stage2_params",
    ]
    return [c for c in preferred if c in df.columns]


def candidate_id(row: pd.Series, cols: list[str]) -> str:
    parts: list[str] = []
    for col in cols:
        value = row[col]
        if isinstance(value, float) and math.isnan(value):
            value = ""
        parts.append(f"{col}={value}")
    return "|".join(parts)


def short_label(row: pd.Series) -> str:
    profile = str(row.get("model_profile", "")).strip()
    family = str(row.get("model_family", "")).strip()
    weight = str(row.get("stage1_weight_mode", "")).strip()
    if profile and profile not in {"nan", family}:
        return profile
    if family and weight and weight != "nan":
        return f"{family}_{weight}"
    return profile or family or "unknown"


def stage2_signature(row: pd.Series) -> str:
    return "|".join(
        [
            f"model_family={row.get('model_family', '')}",
            f"stage2_weight_mode={row.get('stage2_weight_mode', '')}",
            f"stage2_params={row.get('stage2_params', '')}",
        ]
    )


def main() -> int:
    if not PREFLIGHT_JSON.exists():
        raise SystemExit(f"Missing {PREFLIGHT_JSON}; run scripts/jisa_phase3_preflight.py first.")

    preflight = json.loads(PREFLIGHT_JSON.read_text(encoding="utf-8"))
    selected = preflight.get("selected_seed_aggregates", {})
    if not isinstance(selected, dict):
        raise SystemExit("Preflight JSON does not contain selected_seed_aggregates.")

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        raw = selected.get(str(seed))
        if not raw:
            continue
        path = Path(raw)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["dataset"].astype(str) == DATASET].copy()
        cols = candidate_columns(df)
        if not cols:
            raise RuntimeError(f"No candidate identity columns in {path}")
        df["stage2_macro_f1_val"] = pd.to_numeric(df["stage2_macro_f1_val"], errors="coerce")
        df["_candidate_id"] = df.apply(lambda r: candidate_id(r, cols), axis=1)

        for holdout, group in df.groupby("holdout_family", sort=True):
            group = group.sort_index().drop_duplicates(subset=["_candidate_id"], keep="last").copy()
            group = group[group["stage2_macro_f1_val"].notna()].copy()
            group = group.sort_values(["stage2_macro_f1_val", "_candidate_id"], ascending=[False, True])
            if group.empty:
                continue

            top_value = float(group.iloc[0]["stage2_macro_f1_val"])
            tied = group[(top_value - group["stage2_macro_f1_val"].astype(float)).abs() <= TOL].copy()
            lower = group[(top_value - group["stage2_macro_f1_val"].astype(float)) > TOL].copy()
            next_gap = float(top_value - float(lower.iloc[0]["stage2_macro_f1_val"])) if not lower.empty else float("nan")
            tied_labels = [short_label(r) for _, r in tied.iterrows()]
            tied_ids = [str(r["_candidate_id"]) for _, r in tied.iterrows()]
            signatures = [stage2_signature(r) for _, r in tied.iterrows()]
            shared_stage2 = len(set(signatures)) == 1

            rows.append(
                {
                    "seed": seed,
                    "holdout_family": str(holdout),
                    "candidate_count": int(len(group)),
                    "top_stage2_macro_f1_val": top_value,
                    "top_equivalence_count": int(len(tied)),
                    "top_equivalence_labels": "|".join(tied_labels),
                    "next_lower_stage2_gap": next_gap,
                    "top_equivalence_shared_stage2_definition": bool(shared_stage2),
                    "strict_loao_all_top": bool(tied["apply_loao_stage1"].map(as_bool).all()),
                    "targeted_profile_reruns": int(len(tied)),
                }
            )

            for _, r in group.iterrows():
                candidate_rows.append(
                    {
                        "seed": seed,
                        "holdout_family": str(holdout),
                        "candidate_label": short_label(r),
                        "candidate_id": str(r["_candidate_id"]),
                        "stage2_macro_f1_val": float(r["stage2_macro_f1_val"]),
                        "in_top_equivalence_set": bool(str(r["_candidate_id"]) in set(tied_ids)),
                        "stage2_signature": stage2_signature(r),
                    }
                )

    result = pd.DataFrame(rows).sort_values(["holdout_family", "seed"])
    detail = pd.DataFrame(candidate_rows).sort_values(["holdout_family", "seed", "stage2_macro_f1_val"], ascending=[True, True, False])
    result.to_csv(OUT / "top_equivalence_sets.csv", index=False)
    detail.to_csv(OUT / "candidate_detail.csv", index=False)

    expected_groups = len(SEEDS) * 6
    groups_found = int(len(result))
    targeted = int(result["targeted_profile_reruns"].sum()) if not result.empty else 0
    unique_groups = int((result["top_equivalence_count"] == 1).sum()) if not result.empty else 0
    tied_groups = int((result["top_equivalence_count"] > 1).sum()) if not result.empty else 0
    shared_stage2_tied = int(
        result.loc[result["top_equivalence_count"] > 1, "top_equivalence_shared_stage2_definition"].sum()
    ) if not result.empty else 0
    all_strict = bool(result["strict_loao_all_top"].all()) if not result.empty else False
    lower_dominated = bool(
        result.loc[result["top_equivalence_count"] > 1, "next_lower_stage2_gap"].dropna().gt(TOL).all()
    ) if tied_groups else True
    execution_ready = bool(groups_found == expected_groups and all_strict and lower_dominated)

    payload = {
        "dataset": DATASET,
        "groups_expected": expected_groups,
        "groups_found": groups_found,
        "unique_primary_groups": unique_groups,
        "tied_primary_groups": tied_groups,
        "targeted_candidate_reruns": targeted,
        "tied_groups_with_shared_stage2_definition": shared_stage2_tied,
        "all_top_candidates_strict_loao": all_strict,
        "all_non_top_candidates_strictly_dominated_on_blind_safe_primary": lower_dominated,
        "execution_ready": execution_ready,
    }
    (OUT / "tie_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"groups: {groups_found}/{expected_groups}")
    print(f"unique Stage-2 primary groups: {unique_groups}/{expected_groups}")
    print(f"tied Stage-2 primary groups: {tied_groups}/{expected_groups}")
    print(f"targeted candidate reruns required: {targeted}")
    print(f"tied groups sharing one Stage-2 definition: {shared_stage2_tied}/{tied_groups}")
    print(f"non-top candidates strictly dominated: {lower_dominated}")
    print(result[["seed", "holdout_family", "top_equivalence_count", "top_equivalence_labels", "next_lower_stage2_gap", "top_equivalence_shared_stage2_definition"]].to_string(index=False))
    print(f"audit: {OUT / 'tie_audit.json'}")
    print(f"EXECUTION_READY={execution_ready}")
    return 0 if execution_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
