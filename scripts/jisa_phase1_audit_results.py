#!/usr/bin/env python3
"""Audit completed JISA Phase-1 controlled-direct results.

Read-only with respect to scientific outputs. The script validates the expected
2 datasets x 2 model families x 5 seeds matrix, recomputes the five-seed summary,
checks test isolation metadata, checks that no row-level score/prediction or fitted
classifier artifact was persisted, hashes the compact result files, and writes only
under .release-audit/jisa_phase1_results/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "outputs" / "13_jisa_q1_revision" / "phase1_controlled_direct"
AUDIT_ROOT = REPO_ROOT / ".release-audit" / "jisa_phase1_results"
DATASETS = ["CICIDS2017", "CICIoT2023"]
MODELS = ["rf", "xgb"]
SEEDS = [123, 124, 125, 126, 127]
EXPECTED_TEST_ROWS = {"CICIDS2017": 315174, "CICIoT2023": 911053}
EXPECTED_TRAIN_ROWS = {"CICIDS2017": 900000, "CICIoT2023": 1500000}
EXPECTED_SURFACES = {"argmax", "fpr_matched"}
EXPECTED_SPLITS = {"val", "test"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    metric_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict] = []
    item_rows: list[dict] = []

    for dataset in DATASETS:
        for model in MODELS:
            for seed in SEEDS:
                item = RESULT_ROOT / f"seed_{seed}" / dataset / model
                required = [
                    item / "run_plan.json",
                    item / "selected_fpr_gate.json",
                    item / "metrics.csv",
                    item / "run_complete.json",
                    item / "labels.json",
                    item / "validation_family_gate_recall.csv",
                    item / "test_family_gate_recall.csv",
                ]
                missing = [str(p.relative_to(REPO_ROOT)) for p in required if not p.exists()]
                if missing:
                    errors.append(f"{dataset}/{model}/seed_{seed}: missing {missing}")
                    continue

                plan = read_json(item / "run_plan.json")
                complete = read_json(item / "run_complete.json")
                gate = read_json(item / "selected_fpr_gate.json")
                metrics = pd.read_csv(item / "metrics.csv")

                if bool(plan.get("test_used_for_selection", True)):
                    errors.append(f"{dataset}/{model}/seed_{seed}: run_plan says test used for selection")
                if bool(complete.get("test_used_for_selection", True)):
                    errors.append(f"{dataset}/{model}/seed_{seed}: run_complete says test used for selection")
                if bool(plan.get("row_level_predictions_saved", True)):
                    errors.append(f"{dataset}/{model}/seed_{seed}: run_plan says row predictions saved")
                if bool(complete.get("row_level_predictions_saved", True)):
                    errors.append(f"{dataset}/{model}/seed_{seed}: run_complete says row predictions saved")
                if str(gate.get("selection_scope")) != "validation_only":
                    errors.append(f"{dataset}/{model}/seed_{seed}: gate selection_scope is not validation_only")
                if float(gate.get("validation_benign_fpr", 1.0)) > float(gate.get("target_benign_fpr", 0.015)) + 1e-12:
                    errors.append(f"{dataset}/{model}/seed_{seed}: validation FPR exceeds target")

                expected_pairs = {(s, u) for s in EXPECTED_SPLITS for u in EXPECTED_SURFACES}
                actual_pairs = set(zip(metrics["split"].astype(str), metrics["surface"].astype(str)))
                if actual_pairs != expected_pairs or len(metrics) != 4:
                    errors.append(f"{dataset}/{model}/seed_{seed}: unexpected metrics rows/pairs {actual_pairs}")

                test = metrics[metrics["split"].astype(str) == "test"]
                if not test.empty:
                    if not (test["n_eval_rows"].astype(int) == EXPECTED_TEST_ROWS[dataset]).all():
                        errors.append(f"{dataset}/{model}/seed_{seed}: test row count mismatch")
                    if not (test["n_train_rows"].astype(int) == EXPECTED_TRAIN_ROWS[dataset]).all():
                        errors.append(f"{dataset}/{model}/seed_{seed}: train row count mismatch")

                metrics = metrics.copy()
                metrics["source_item"] = f"{dataset}/{model}/seed_{seed}"
                metric_frames.append(metrics)

                # Only a shared preprocessor.joblib is allowed above model item dirs.
                suspicious = []
                for p in item.rglob("*"):
                    if not p.is_file():
                        continue
                    low = p.name.lower()
                    if any(token in low for token in ["prediction", "predictions", "probability", "probabilities", "scores", "score.csv"]):
                        suspicious.append(str(p.relative_to(REPO_ROOT)))
                    if p.suffix.lower() in {".joblib", ".pkl", ".pickle", ".model", ".ubj", ".bin"}:
                        suspicious.append(str(p.relative_to(REPO_ROOT)))
                if suspicious:
                    errors.append(f"{dataset}/{model}/seed_{seed}: forbidden persisted artifacts {sorted(set(suspicious))}")

                for p in required:
                    manifest_rows.append({
                        "dataset": dataset,
                        "model_family": model,
                        "seed": seed,
                        "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
                        "bytes": p.stat().st_size,
                        "sha256": sha256(p),
                    })

                item_rows.append({
                    "dataset": dataset,
                    "model_family": model,
                    "seed": seed,
                    "validation_threshold": gate.get("threshold"),
                    "validation_benign_fpr": gate.get("validation_benign_fpr"),
                    "validation_attack_recall": gate.get("validation_attack_recall"),
                    "validation_min_family_gate_recall": gate.get("validation_min_family_gate_recall"),
                    "fit_seconds": complete.get("fit_seconds"),
                    "status": "ok",
                })

    if metric_frames:
        all_metrics = pd.concat(metric_frames, ignore_index=True)
        all_metrics.to_csv(AUDIT_ROOT / "all_metrics_recomputed.csv", index=False)
        test = all_metrics[all_metrics["split"].astype(str) == "test"].copy()
        grouped = (
            test.groupby(["dataset", "model_family", "surface"], as_index=False)
            .agg(
                n=("seed", "count"),
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_sd=("macro_f1", "std"),
                macro_f1_min=("macro_f1", "min"),
                macro_f1_max=("macro_f1", "max"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_sd=("accuracy", "std"),
                benign_fpr_mean=("benign_family_fp_rate", "mean"),
                benign_fpr_sd=("benign_family_fp_rate", "std"),
            )
        )
        grouped.to_csv(AUDIT_ROOT / "test_seed_summary_recomputed.csv", index=False)
        bad_n = grouped[grouped["n"].astype(int) != 5]
        if len(grouped) != 8 or not bad_n.empty:
            errors.append(f"Expected 8 dataset/model/surface groups with n=5; found {len(grouped)} groups")
    else:
        grouped = pd.DataFrame()
        errors.append("No Phase-1 metrics were found")

    pd.DataFrame(manifest_rows).to_csv(AUDIT_ROOT / "compact_file_manifest.csv", index=False)
    pd.DataFrame(item_rows).to_csv(AUDIT_ROOT / "item_audit.csv", index=False)

    report = {
        "result_root": str(RESULT_ROOT),
        "expected_items": 20,
        "audited_items": len(item_rows),
        "expected_summary_groups": 8,
        "summary_groups": int(len(grouped)),
        "errors": errors,
        "warnings": warnings,
        "pass": not errors and len(item_rows) == 20 and len(grouped) == 8,
    }
    (AUDIT_ROOT / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("JISA Phase-1 results audit")
    print(f"items: {len(item_rows)}/20")
    print(f"summary groups: {len(grouped)}/8")
    if not grouped.empty:
        print(grouped.to_string(index=False))
    if errors:
        print("ERRORS:")
        for err in errors:
            print(f"  - {err}")
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"PASS={report['pass']}")
    print(f"audit: {AUDIT_ROOT / 'audit.json'}")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
