#!/usr/bin/env python3
"""Run one JISA Phase-1 controlled direct-multiclass experiment item.

This runner is deliberately separate from the historical competitive baseline. It
trains exactly one RF or XGB direct classifier for one dataset/seed on the frozen
Protocol-A preparation, reports ordinary multiclass argmax and an FPR-matched
benign/attack gate selected on validation only, and never writes row-level scores.

Heavy outputs stay under outputs/13_jisa_q1_revision/ and are git-ignored.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework._native import protocol_a_competitive as comp  # noqa: E402
from ids_eval_framework.src import two_stage_engine as helper  # noqa: E402
from ids_eval_framework.src.paths import load_config  # noqa: E402

DEFAULT_CONFIG = "config/jisa_phase1_controlled_direct.yml"
DEFAULT_OUTPUT_ROOT = "outputs/13_jisa_q1_revision/phase1_controlled_direct"
# Match the historical Protocol-A Stage-1 sampling budget. CICIDS validation is
# smaller than its cap and is therefore read in full; CICIoT retains the 650k cap.
TRAIN_CAPS = {"CICIDS2017": 900_000, "CICIoT2023": 1_500_000}
VAL_CAPS = {"CICIDS2017": 500_000, "CICIoT2023": 650_000}
CHUNKSIZE = 200_000
N_JOBS = 24


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def full_split_rows(ds_dir: Path, split: str) -> int:
    parts = helper.list_parts(str(ds_dir), split)
    n = comp.estimate_part_rows(parts)
    if n is None:
        raise RuntimeError(f"Cannot determine exact row count for {ds_dir}/{split}")
    return int(n)


def configure_runtime(processed_root: Path, protocol: str, dataset: str, seed: int) -> None:
    comp.CFG["processed_root"] = str(processed_root)
    comp.CFG["protocol"] = protocol
    comp.CFG["datasets"] = [dataset]
    comp.CFG["random_seed"] = int(seed)
    comp.CFG["n_jobs"] = N_JOBS
    comp.CFG["chunksize_rows"] = CHUNKSIZE
    comp.configure_helper(helper)
    helper.CFG["global_seed"] = int(seed)


def build_model(model_name: str, params: dict[str, Any], n_classes: int, seed: int):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            n_jobs=N_JOBS,
            random_state=int(seed),
        )
    if model_name == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not available")
        return XGBClassifier(
            objective="multi:softprob",
            num_class=int(n_classes),
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            tree_method="hist",
            device="cuda",
            n_jobs=N_JOBS,
            random_state=int(seed),
            eval_metric="mlogloss",
            verbosity=1,
        )
    raise ValueError(model_name)


def fit_model(model_name: str, model, X, y):
    if model_name == "xgb":
        sw = compute_sample_weight(class_weight="balanced", y=y).astype(np.float64)
        model.fit(X, y, sample_weight=sw)
    else:
        model.fit(X, y)
    return model


def argmax_predictions(proba: np.ndarray) -> np.ndarray:
    return np.argmax(proba, axis=1).astype(int)


def choose_fpr_threshold(y_idx: np.ndarray, proba: np.ndarray, labels: list[str], target_fpr: float) -> dict[str, float]:
    benign_idx = labels.index("Benign")
    true_benign = y_idx == benign_idx
    if not np.any(true_benign):
        raise RuntimeError("Validation split contains no Benign rows")
    attack_score = 1.0 - proba[:, benign_idx]
    benign_scores = np.asarray(attack_score[true_benign], dtype=np.float64)
    n = int(len(benign_scores))
    allowed = int(math.floor(float(target_fpr) * n))
    desc = np.sort(benign_scores)[::-1]
    if allowed <= 0:
        threshold = float(np.nextafter(desc[0], np.inf))
    else:
        boundary = float(desc[min(allowed - 1, n - 1)])
        count_ge = int(np.sum(benign_scores >= boundary))
        threshold = boundary if count_ge <= allowed else float(np.nextafter(boundary, np.inf))
    pred_attack = attack_score >= threshold
    realized_fpr = float(np.mean(pred_attack[true_benign]))
    true_attack = ~true_benign
    attack_tpr = float(np.mean(pred_attack[true_attack])) if np.any(true_attack) else float("nan")
    if realized_fpr > float(target_fpr) + 1e-12:
        raise RuntimeError(f"Validation FPR constraint violated: {realized_fpr} > {target_fpr}")
    return {
        "threshold": float(threshold),
        "target_benign_fpr": float(target_fpr),
        "validation_benign_fpr": realized_fpr,
        "validation_attack_recall": attack_tpr,
        "n_validation_benign": n,
        "allowed_validation_false_positives": allowed,
    }


def gated_predictions(proba: np.ndarray, labels: list[str], threshold: float) -> np.ndarray:
    benign_idx = labels.index("Benign")
    attack_indices = np.array([i for i, label in enumerate(labels) if label != "Benign"], dtype=int)
    attack_score = 1.0 - proba[:, benign_idx]
    attack_local = np.argmax(proba[:, attack_indices], axis=1)
    attack_pred = attack_indices[attack_local]
    pred = np.full(len(proba), benign_idx, dtype=int)
    gate = attack_score >= float(threshold)
    pred[gate] = attack_pred[gate]
    return pred


def family_gate_recall(y_idx: np.ndarray, proba: np.ndarray, labels: list[str], threshold: float, min_support: int) -> pd.DataFrame:
    benign_idx = labels.index("Benign")
    gate = (1.0 - proba[:, benign_idx]) >= float(threshold)
    rows = []
    for idx, label in enumerate(labels):
        if label == "Benign":
            continue
        mask = y_idx == idx
        support = int(mask.sum())
        rows.append({
            "family": label,
            "support": support,
            "gate_recall": float(np.mean(gate[mask])) if support else float("nan"),
            "included_in_min_family_objective": bool(support >= int(min_support)),
        })
    return pd.DataFrame(rows)


def metric_row(dataset: str, model_name: str, seed: int, split: str, surface: str, y_idx: np.ndarray, pred: np.ndarray, labels: list[str], fit_seconds: float, n_train: int) -> dict[str, Any]:
    payload = comp.metric_payload(y_idx, pred, labels)
    return {
        "dataset": dataset,
        "model_family": model_name,
        "seed": int(seed),
        "split": split,
        "surface": surface,
        "n_train_rows": int(n_train),
        "n_eval_rows": int(len(y_idx)),
        "fit_seconds": float(fit_seconds),
        **payload,
    }


def write_reports(out_dir: Path, prefix: str, y_idx: np.ndarray, pred: np.ndarray, labels: list[str]) -> None:
    report = classification_report(
        y_idx,
        pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(out_dir / f"{prefix}_classification_report.csv")
    cm = confusion_matrix(y_idx, pred, labels=list(range(len(labels))))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"{prefix}_confusion_matrix.csv")


def run_item(config_path: str, dataset: str, model_name: str, seed: int, *, dry_run: bool, force: bool) -> None:
    cfg = load_config(config_path)
    section = cfg.get("experiment", {})
    allowed_datasets = [str(x) for x in section.get("datasets", [])]
    allowed_seeds = [int(x) for x in section.get("seeds", [])]
    if dataset not in allowed_datasets:
        raise ValueError(f"Dataset {dataset} not allowed by frozen config")
    if int(seed) not in allowed_seeds:
        raise ValueError(f"Seed {seed} not allowed by frozen config")
    profiles = cfg.get("model_profiles", {})
    if model_name not in profiles:
        raise ValueError(f"Model {model_name} not allowed by frozen config")

    processed_root = REPO_ROOT / str(section.get("processed_root", "processed_V5"))
    protocol = str(section.get("protocol", "A_stratified"))
    ds_dir = processed_root / protocol / dataset
    output_root = REPO_ROOT / str(section.get("output_root", DEFAULT_OUTPUT_ROOT))
    item_dir = output_root / f"seed_{seed}" / dataset / model_name
    dataset_seed_dir = output_root / f"seed_{seed}" / dataset
    metrics_path = item_dir / "metrics.csv"

    plan = {
        "dataset": dataset,
        "model_family": model_name,
        "seed": int(seed),
        "processed_dir": str(ds_dir),
        "output_dir": str(item_dir),
        "train_cap": TRAIN_CAPS[dataset],
        "val_cap": VAL_CAPS[dataset],
        "test_rows": full_split_rows(ds_dir, "test"),
        "model_profile": profiles[model_name],
        "target_benign_fpr": float(cfg["fpr_matched_gate"]["target_benign_fpr"]),
        "row_level_predictions_saved": False,
        "test_used_for_selection": False,
    }
    if dry_run:
        print(json.dumps(plan, indent=2))
        return
    if metrics_path.exists() and not force:
        print(f"SKIP completed item: {metrics_path}")
        print(pd.read_csv(metrics_path).to_string(index=False))
        return

    item_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(processed_root, protocol, dataset, seed)
    write_json(item_dir / "run_plan.json", plan)

    prep_path = dataset_seed_dir / "preprocessor.joblib"
    if prep_path.exists():
        prep = comp.safe_joblib_load(str(prep_path))
    else:
        dataset_seed_dir.mkdir(parents=True, exist_ok=True)
        prep = helper.fit_preprocessor(str(ds_dir), str(dataset_seed_dir))
        if not prep_path.exists():
            comp.safe_joblib_dump(prep, str(prep_path))

    train = comp.collect_direct_data(helper, str(ds_dir), "train", prep, None, TRAIN_CAPS[dataset], seed)
    val = comp.collect_direct_data(helper, str(ds_dir), "val", prep, train.labels, VAL_CAPS[dataset], seed + 1)
    test_n = full_split_rows(ds_dir, "test")
    test = comp.collect_direct_data(helper, str(ds_dir), "test", prep, train.labels, test_n, seed + 2)
    if val.labels != train.labels or test.labels != train.labels:
        raise RuntimeError("Label taxonomy differs across Protocol-A train/val/test")
    labels = list(train.labels)
    if "Benign" not in labels:
        raise RuntimeError("Benign label missing from direct taxonomy")

    model = build_model(model_name, dict(profiles[model_name].get("params", {})), len(labels), seed)
    started = time.perf_counter()
    model = fit_model(model_name, model, train.X, train.y_idx)
    fit_seconds = time.perf_counter() - started

    p_val = comp.predict_proba_aligned(model, val.X, len(labels))
    gate = choose_fpr_threshold(
        val.y_idx,
        p_val,
        labels,
        float(cfg["fpr_matched_gate"]["target_benign_fpr"]),
    )
    min_support = int(cfg["fpr_matched_gate"].get("min_family_support", 50))
    fam_val = family_gate_recall(val.y_idx, p_val, labels, gate["threshold"], min_support)
    eligible = fam_val[fam_val["included_in_min_family_objective"]]
    gate["validation_min_family_gate_recall"] = float(eligible["gate_recall"].min()) if not eligible.empty else float("nan")
    write_json(item_dir / "selected_fpr_gate.json", gate)
    fam_val.to_csv(item_dir / "validation_family_gate_recall.csv", index=False)

    rows = []
    pred_val_argmax = argmax_predictions(p_val)
    pred_val_gate = gated_predictions(p_val, labels, gate["threshold"])
    rows.append(metric_row(dataset, model_name, seed, "val", "argmax", val.y_idx, pred_val_argmax, labels, fit_seconds, len(train.y_idx)))
    rows.append(metric_row(dataset, model_name, seed, "val", "fpr_matched", val.y_idx, pred_val_gate, labels, fit_seconds, len(train.y_idx)))

    p_test = comp.predict_proba_aligned(model, test.X, len(labels))
    pred_test_argmax = argmax_predictions(p_test)
    pred_test_gate = gated_predictions(p_test, labels, gate["threshold"])
    rows.append(metric_row(dataset, model_name, seed, "test", "argmax", test.y_idx, pred_test_argmax, labels, fit_seconds, len(train.y_idx)))
    rows.append(metric_row(dataset, model_name, seed, "test", "fpr_matched", test.y_idx, pred_test_gate, labels, fit_seconds, len(train.y_idx)))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    family_gate_recall(test.y_idx, p_test, labels, gate["threshold"], min_support).to_csv(item_dir / "test_family_gate_recall.csv", index=False)
    write_json(item_dir / "labels.json", {"labels": labels})
    if bool(cfg.get("reporting", {}).get("save_confusion_matrices", True)):
        write_reports(item_dir, "val_argmax", val.y_idx, pred_val_argmax, labels)
        write_reports(item_dir, "val_fpr_matched", val.y_idx, pred_val_gate, labels)
        write_reports(item_dir, "test_argmax", test.y_idx, pred_test_argmax, labels)
        write_reports(item_dir, "test_fpr_matched", test.y_idx, pred_test_gate, labels)
    write_json(item_dir / "run_complete.json", {
        "dataset": dataset,
        "model_family": model_name,
        "seed": int(seed),
        "fit_seconds": float(fit_seconds),
        "test_used_for_selection": False,
        "row_level_predictions_saved": False,
    })

    print(f"COMPLETE {dataset} {model_name} seed={seed}")
    print(f"fit_seconds={fit_seconds:.1f}")
    print(f"selected_threshold={gate['threshold']:.12g} validation_FPR={gate['validation_benign_fpr']:.6f} validation_attack_recall={gate['validation_attack_recall']:.6f}")
    print(metrics[metrics["split"] == "test"].to_string(index=False))


def summarize(config_path: str) -> None:
    cfg = load_config(config_path)
    root = REPO_ROOT / str(cfg.get("experiment", {}).get("output_root", DEFAULT_OUTPUT_ROOT))
    frames = []
    for path in root.glob("seed_*/*/*/metrics.csv"):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass
    if not frames:
        print("No completed Phase-1 metrics found.")
        return
    df = pd.concat(frames, ignore_index=True)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_dir / "all_metrics.csv", index=False)
    test = df[df["split"] == "test"].copy()
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
    grouped.to_csv(summary_dir / "test_seed_summary.csv", index=False)
    print(grouped.to_string(index=False))
    print(f"summary: {summary_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--dataset", choices=["CICIDS2017", "CICIoT2023"])
    p.add_argument("--model", choices=["rf", "xgb"])
    p.add_argument("--seed", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.summarize_only:
        summarize(args.config)
        return 0
    if args.dataset is None or args.model is None or args.seed is None:
        raise SystemExit("--dataset, --model, and --seed are required unless --summarize-only is used")
    run_item(args.config, args.dataset, args.model, args.seed, dry_run=args.dry_run, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
