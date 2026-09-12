#!/usr/bin/env python3
"""Audit the completed JISA Phase-3A heldout-validation-blind score surface.

This script is read-only with respect to scientific outputs. It verifies all 30
CICIoT2023 seed x holdout cases after score generation, including profile-selection
logic, held-out validation exclusion, compact artifact consistency, score schemas,
and absence of persisted fitted Stage-1/Stage-2 classifiers. With ``--deep`` it also
streams every test score file to verify row counts and genuine Unknown support.

Audit output is written only under ``.release-audit/jisa_phase3a_results``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "jisa_phase3_validation_blind.yml"
AUDIT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase3a_results"
DATASET = "CICIoT2023"
TOL = 1e-12

SCORE_COLUMNS = {
    "row_id",
    "split",
    "y_stage1_attack",
    "y_stage2_family",
    "y_true_sys",
    "is_true_unknown",
    "p_attack",
    "fam_pred_idx",
    "fam_pred_family",
    "fam_pmax",
    "top2_margin",
    "stage2_entropy",
    "true_known_family_prob",
}

REQUIRED_CASE_FILES = [
    "run_complete.json",
    "selected_profile.json",
    "candidate_profile_selection.csv",
    "stage1_threshold_grid.csv",
    "stage1_threshold_best.json",
    "val_known_scores.csv.gz",
    "test_scores.csv.gz",
    "preprocessor.joblib",
]


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for this audit.") from exc
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def finite01(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x) and 0.0 <= x <= 1.0


def score_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def audit_val_known(path: Path, holdout: str, expected_rows: int) -> tuple[bool, int, int, int]:
    rows = 0
    heldout_attack_rows = 0
    unknown_rows = 0
    bad_stage1 = 0
    for chunk in pd.read_csv(
        path,
        usecols=["y_stage1_attack", "y_stage2_family", "is_true_unknown"],
        chunksize=250_000,
    ):
        rows += len(chunk)
        y1 = pd.to_numeric(chunk["y_stage1_attack"], errors="coerce")
        fam = chunk["y_stage2_family"].astype(str)
        unk = pd.to_numeric(chunk["is_true_unknown"], errors="coerce").fillna(-1)
        heldout_attack_rows += int(((y1 == 1) & (fam == str(holdout))).sum())
        unknown_rows += int((unk == 1).sum())
        bad_stage1 += int((~y1.isin([0, 1])).sum())
    ok = rows == int(expected_rows) and heldout_attack_rows == 0 and unknown_rows == 0 and bad_stage1 == 0
    return ok, rows, heldout_attack_rows, unknown_rows


def audit_test_deep(path: Path, expected_rows: int) -> tuple[bool, int, int]:
    rows = 0
    unknown_rows = 0
    for chunk in pd.read_csv(path, usecols=["is_true_unknown"], chunksize=250_000):
        rows += len(chunk)
        unk = pd.to_numeric(chunk["is_true_unknown"], errors="coerce").fillna(-1)
        unknown_rows += int((unk == 1).sum())
    ok = rows == int(expected_rows) and unknown_rows > 0
    return ok, rows, unknown_rows


def selection_rule_ok(selection: pd.DataFrame, selected_id: str) -> tuple[bool, str]:
    required = {
        "candidate_id",
        "stage2_macro_f1_blind_val",
        "stage1_auc_blind_val",
        "selected",
    }
    if not required.issubset(selection.columns) or selection.empty:
        return False, "missing_selection_columns"

    work = selection.copy()
    work["stage2_macro_f1_blind_val"] = pd.to_numeric(work["stage2_macro_f1_blind_val"], errors="coerce")
    work["stage1_auc_blind_val"] = pd.to_numeric(work["stage1_auc_blind_val"], errors="coerce")
    if work[["stage2_macro_f1_blind_val", "stage1_auc_blind_val"]].isna().any().any():
        return False, "non_numeric_selection_metrics"

    selected_mask = work["selected"].astype(str).str.lower().isin(["true", "1", "yes"])
    if int(selected_mask.sum()) != 1:
        return False, "selected_count_not_one"
    actual = str(work.loc[selected_mask, "candidate_id"].iloc[0])
    if actual != str(selected_id):
        return False, "selected_id_mismatch"

    top_f1 = float(work["stage2_macro_f1_blind_val"].max())
    eligible = work[(top_f1 - work["stage2_macro_f1_blind_val"].astype(float)).abs() <= TOL].copy()
    top_auc = float(eligible["stage1_auc_blind_val"].max())
    eligible = eligible[(top_auc - eligible["stage1_auc_blind_val"].astype(float)).abs() <= TOL].copy()
    expected = str(sorted(eligible["candidate_id"].astype(str).tolist())[0])
    if expected != actual:
        return False, "lexicographic_selection_rule_violation"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--deep", action="store_true", help="Stream all 30 full test score files and verify row counts/Unknown support.")
    args = parser.parse_args()

    cfg = load_yaml(args.config.resolve())
    exp = cfg.get("experiment", {}) or {}
    seeds = [int(x) for x in exp.get("seeds", [])]
    holdouts = [str(x) for x in exp.get("holdouts", [])]
    output_root = REPO_ROOT / str(exp.get("output_root", "outputs/13_jisa_q1_revision/phase3_validation_blind"))
    score_root = output_root / "score_generation"

    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    compact_hashes: list[dict[str, str]] = []

    for seed in seeds:
        for holdout in holdouts:
            case_dir = score_root / f"seed_{seed}" / holdout.replace("/", "_")
            rec: dict[str, Any] = {
                "seed": seed,
                "holdout_family": holdout,
                "case_dir": str(case_dir),
                "case_exists": case_dir.exists(),
            }
            missing = [name for name in REQUIRED_CASE_FILES if not (case_dir / name).exists()]
            rec["missing_files"] = "|".join(missing)
            rec["files_complete"] = not missing
            if missing:
                rec["PASS"] = False
                rows.append(rec)
                continue

            run_complete = load_json(case_dir / "run_complete.json")
            selected = load_json(case_dir / "selected_profile.json")
            selection = pd.read_csv(case_dir / "candidate_profile_selection.csv")
            threshold_best = load_json(case_dir / "stage1_threshold_best.json")

            for name in ["run_complete.json", "selected_profile.json", "candidate_profile_selection.csv", "stage1_threshold_best.json"]:
                compact_hashes.append({
                    "seed": str(seed),
                    "holdout_family": holdout,
                    "path": str(case_dir / name),
                    "sha256": sha256_file(case_dir / name),
                })

            expected_top_count = 1 if holdout == "BruteForce" else 2
            top_count = int(selected.get("top_equivalence_count", -1))
            rec["top_equivalence_count"] = top_count
            rec["top_equivalence_count_ok"] = top_count == expected_top_count and len(selection) == expected_top_count

            rec["visibility_flags_ok"] = (
                selected.get("heldout_validation_used_for_selection") is False
                and selected.get("test_used_for_selection") is False
                and str(selected.get("preprocessor_exposure", "")) == "historical_full_train_unsupervised_feature_exposure"
            )
            rec["dataset_ok"] = str(selected.get("dataset", "")) == DATASET
            rec["seed_ok"] = int(selected.get("seed", -1)) == seed
            rec["holdout_ok"] = str(selected.get("holdout_family", "")) == holdout
            rec["known_family_count"] = len(selected.get("known_families", []))
            rec["known_families_ok"] = (
                len(selected.get("known_families", [])) == 5
                and holdout not in {str(x) for x in selected.get("known_families", [])}
            )

            selected_id = str(selected.get("selected_candidate_id", ""))
            sel_ok, sel_reason = selection_rule_ok(selection, selected_id)
            rec["selection_rule_ok"] = sel_ok
            rec["selection_rule_reason"] = sel_reason
            rec["selected_candidate_label"] = str(selected.get("selected_candidate_label", ""))
            rec["stage2_macro_f1_blind_val"] = selected.get("stage2_macro_f1_blind_val")
            rec["stage1_auc_blind_val"] = selected.get("stage1_auc_blind_val")
            rec["stage1_threshold"] = selected.get("stage1_threshold")
            rec["stage1_threshold_ok"] = finite01(selected.get("stage1_threshold"))

            threshold_match = False
            try:
                threshold_match = abs(float(run_complete.get("stage1_threshold")) - float(selected.get("stage1_threshold"))) <= TOL
            except Exception:
                threshold_match = False
            rec["threshold_metadata_match"] = threshold_match
            rec["threshold_best_present"] = bool(threshold_best)

            blind_rows = int(selected.get("blind_validation_rows", -1))
            hidden_rows = int(selected.get("hidden_holdout_validation_rows", -1))
            test_rows_expected = int(selected.get("test_rows", -1))
            rec["blind_validation_rows"] = blind_rows
            rec["hidden_holdout_validation_rows"] = hidden_rows
            rec["test_rows_expected"] = test_rows_expected
            rec["hidden_rows_positive"] = hidden_rows > 0

            val_path = case_dir / "val_known_scores.csv.gz"
            test_path = case_dir / "test_scores.csv.gz"
            val_header = set(score_header(val_path))
            test_header = set(score_header(test_path))
            rec["score_schema_ok"] = SCORE_COLUMNS.issubset(val_header) and SCORE_COLUMNS.issubset(test_header)

            val_ok, val_rows, heldout_in_val, unknown_in_val = audit_val_known(val_path, holdout, blind_rows)
            rec["val_known_rows_actual"] = val_rows
            rec["val_known_heldout_attack_rows"] = heldout_in_val
            rec["val_known_unknown_rows"] = unknown_in_val
            rec["val_known_blind_ok"] = val_ok

            model_files = []
            for suffix in ("*.joblib", "*.pkl", "*.pickle", "*.onnx", "*.pt", "*.pth"):
                model_files.extend(case_dir.glob(suffix))
            unexpected_models = [p.name for p in model_files if p.name != "preprocessor.joblib"]
            rec["unexpected_model_files"] = "|".join(sorted(unexpected_models))
            rec["model_persistence_ok"] = len(unexpected_models) == 0 and run_complete.get("models_saved") is False

            if args.deep:
                test_ok, test_rows_actual, test_unknown_rows = audit_test_deep(test_path, test_rows_expected)
                rec["test_rows_actual"] = test_rows_actual
                rec["test_unknown_rows"] = test_unknown_rows
                rec["test_deep_ok"] = test_ok
            else:
                rec["test_rows_actual"] = None
                rec["test_unknown_rows"] = None
                rec["test_deep_ok"] = True

            checks = [
                rec["files_complete"], rec["top_equivalence_count_ok"], rec["visibility_flags_ok"],
                rec["dataset_ok"], rec["seed_ok"], rec["holdout_ok"], rec["known_families_ok"],
                rec["selection_rule_ok"], rec["stage1_threshold_ok"], rec["threshold_metadata_match"],
                rec["threshold_best_present"], rec["hidden_rows_positive"], rec["score_schema_ok"],
                rec["val_known_blind_ok"], rec["model_persistence_ok"], rec["test_deep_ok"],
            ]
            rec["PASS"] = bool(all(checks))
            rows.append(rec)

    df = pd.DataFrame(rows).sort_values(["holdout_family", "seed"]) if rows else pd.DataFrame()
    df.to_csv(AUDIT_ROOT / "case_audit.csv", index=False)
    pd.DataFrame(compact_hashes).to_csv(AUDIT_ROOT / "compact_artifact_hashes.csv", index=False)

    expected_cases = len(seeds) * len(holdouts)
    passed = int(df["PASS"].sum()) if not df.empty and "PASS" in df.columns else 0
    selection_passed = int(df.get("selection_rule_ok", pd.Series(dtype=bool)).fillna(False).sum()) if not df.empty else 0
    blind_passed = int(df.get("val_known_blind_ok", pd.Series(dtype=bool)).fillna(False).sum()) if not df.empty else 0
    schema_passed = int(df.get("score_schema_ok", pd.Series(dtype=bool)).fillna(False).sum()) if not df.empty else 0
    model_passed = int(df.get("model_persistence_ok", pd.Series(dtype=bool)).fillna(False).sum()) if not df.empty else 0
    deep_passed = int(df.get("test_deep_ok", pd.Series(dtype=bool)).fillna(False).sum()) if not df.empty else 0

    selected_by_holdout: dict[str, list[str]] = {}
    threshold_ranges: dict[str, dict[str, float]] = {}
    if not df.empty:
        for holdout, g in df.groupby("holdout_family", sort=True):
            selected_by_holdout[str(holdout)] = sorted(set(g["selected_candidate_label"].astype(str)))
            vals = pd.to_numeric(g["stage1_threshold"], errors="coerce").dropna()
            if not vals.empty:
                threshold_ranges[str(holdout)] = {"min": float(vals.min()), "max": float(vals.max())}

    pass_all = len(df) == expected_cases and passed == expected_cases and (not args.deep or deep_passed == expected_cases)
    payload = {
        "dataset": DATASET,
        "deep": bool(args.deep),
        "cases_expected": expected_cases,
        "cases_found": int(len(df)),
        "cases_passed": passed,
        "selection_rule_passed": selection_passed,
        "blind_validation_passed": blind_passed,
        "score_schema_passed": schema_passed,
        "model_persistence_passed": model_passed,
        "deep_test_passed": deep_passed if args.deep else None,
        "selected_profiles_by_holdout": selected_by_holdout,
        "stage1_threshold_ranges_by_holdout": threshold_ranges,
        "PASS": bool(pass_all),
    }
    (AUDIT_ROOT / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"cases: {len(df)}/{expected_cases}")
    print(f"selection-rule checks: {selection_passed}/{expected_cases}")
    print(f"blind-validation checks: {blind_passed}/{expected_cases}")
    print(f"score-schema checks: {schema_passed}/{expected_cases}")
    print(f"model-persistence checks: {model_passed}/{expected_cases}")
    if args.deep:
        print(f"deep test-row/Unknown checks: {deep_passed}/{expected_cases}")
    print("selected profiles by holdout:")
    for holdout in sorted(selected_by_holdout):
        print(f"  {holdout}: {', '.join(selected_by_holdout[holdout])}")
    print(f"audit: {AUDIT_ROOT / 'audit.json'}")
    print(f"PASS={pass_all}")
    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
