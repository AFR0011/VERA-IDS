#!/usr/bin/env python3
"""
21.ProtocolA_CompetitiveMetricsRunner.py
=======================================

Purpose
-------
Build a separate, aggressive Protocol A metric-competition lane without
changing the thesis-safe baseline artifacts.

The script writes a fresh `runs_competitive_metrics/` root by default and keeps
two evidence surfaces separate:
    1. direct multiclass, paper-style closed-set benchmarks
    2. two-stage threshold overlays that reuse completed Protocol A core models

No Protocol B, open-set, or thesis-finalization artifact is overwritten.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight

try:
    import joblib
except Exception:
    joblib = None

try:
    import psutil
except Exception:
    psutil = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
except Exception:
    BalancedRandomForestClassifier = None
    EasyEnsembleClassifier = None

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

from ids_eval_framework.src.paths import repo_path


CFG: Dict[str, object] = {
    "processed_root": repo_path("processed_V5"),
    "protocol": "A_stratified",
    "datasets": ["CICIDS2017", "CICIoT2023"],
    "runs_root": "runs_competitive_metrics",
    "baseline_runs_root": "runs_two_stage_V5_A_core",
    "helper_script": "ids_eval_framework.src.two_stage_engine",
    "random_seed": 123,
    "n_jobs": 8,
    "chunksize_rows": 200_000,
    "max_train_rows": {"CICIDS2017": 700_000, "CICIoT2023": 1_000_000},
    "max_val_rows": {"CICIDS2017": 350_000, "CICIoT2023": 500_000},
    "max_test_rows": {"CICIDS2017": 5_000_000, "CICIoT2023": 5_000_000},
    "direct_candidates": [
        "rf_balanced",
        "extra_trees_balanced",
        "balanced_rf",
        "easy_ensemble",
        "xgb_weighted",
        "lgbm_weighted",
        "catboost_weighted",
    ],
    "baseline_overlay_models": ["rf", "xgb"],
    "weak_family_candidates": {
        "CICIDS2017": ["Benign", "Web/App", "BruteForce", "DoS"],
        "CICIoT2023": ["Benign", "BruteForce", "Other", "Scan/Recon"],
    },
    "probability_bias_grid": [0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 1.80],
    "probability_bias_passes": 2,
    "max_bias_accuracy_drop": 0.015,
    "feature_selection": {
        "enabled": False,
        "k": 80,
        "max_fit_rows": 250_000,
    },
    "selection_metric": "macro_f1",
    "min_meaningful_delta": {"CICIDS2017": 0.03, "CICIoT2023": 0.015},
    "stage1_threshold_grid": [0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050],
    "two_stage_tau_grid": [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90],
    "two_stage_max_benign_family_fp_rate": 0.060,
    "two_stage_max_accuracy_drop": 0.015,
    "summary_dirname": "summary",
}


SMOKE_CFG: Dict[str, object] = {
    "runs_root": "runs_competitive_metrics_smoke",
    "datasets": ["CICIDS2017"],
    "max_train_rows": {"CICIDS2017": 60_000},
    "max_val_rows": {"CICIDS2017": 40_000},
    "max_test_rows": {"CICIDS2017": 45_000},
    "direct_candidates": ["extra_trees_balanced"],
    "baseline_overlay_models": ["rf"],
}


@dataclass
class DirectData:
    X: csr_matrix
    y_idx: np.ndarray
    y_label: np.ndarray
    y_binary: np.ndarray
    labels: List[str]


@dataclass
class ScoreBlock:
    row_ids: np.ndarray
    y_true: np.ndarray
    p_attack: np.ndarray
    fam_pred_idx: np.ndarray
    fam_pmax: np.ndarray
    families: List[str]


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        return None


def load_helper_module(path: str):
    from ids_eval_framework.src import two_stage_engine

    return two_stage_engine


def configure_helper(helper) -> None:
    helper.CFG["processed_root"] = CFG["processed_root"]
    helper.CFG["protocol"] = CFG["protocol"]
    helper.CFG["datasets"] = list(CFG["datasets"])
    helper.CFG["runs_root"] = CFG["runs_root"]
    helper.CFG["global_seed"] = int(CFG["random_seed"])
    helper.CFG["chunksize_rows"] = int(CFG["chunksize_rows"])
    helper.CFG["loao"]["enabled"] = False
    helper.CFG["loao"]["apply_to_stage1"] = False


def apply_smoke_overrides() -> None:
    if os.environ.get("IDS_COMPETITIVE_SMOKE", "").strip() not in {"1", "true", "True", "yes"}:
        return
    for key, value in SMOKE_CFG.items():
        CFG[key] = value


def parse_csv_env(name: str) -> Optional[List[str]]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return values or None


def apply_uniform_cap_env(name: str, cfg_key: str) -> None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return
    value = int(raw)
    CFG[cfg_key] = {str(ds): value for ds in CFG["datasets"]}


def apply_runtime_overrides() -> None:
    datasets = parse_csv_env("IDS_COMPETITIVE_DATASETS")
    if datasets is not None:
        CFG["datasets"] = datasets
    candidates = parse_csv_env("IDS_COMPETITIVE_CANDIDATES")
    if candidates is not None:
        CFG["direct_candidates"] = candidates
    overlays = parse_csv_env("IDS_COMPETITIVE_OVERLAYS")
    if overlays is not None:
        CFG["baseline_overlay_models"] = overlays
    if os.environ.get("IDS_COMPETITIVE_RUNS_ROOT", "").strip():
        CFG["runs_root"] = os.environ["IDS_COMPETITIVE_RUNS_ROOT"].strip()
    if os.environ.get("IDS_COMPETITIVE_N_JOBS", "").strip():
        CFG["n_jobs"] = int(os.environ["IDS_COMPETITIVE_N_JOBS"].strip())
    apply_uniform_cap_env("IDS_COMPETITIVE_MAX_TRAIN_ROWS", "max_train_rows")
    apply_uniform_cap_env("IDS_COMPETITIVE_MAX_VAL_ROWS", "max_val_rows")
    apply_uniform_cap_env("IDS_COMPETITIVE_MAX_TEST_ROWS", "max_test_rows")


def save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def safe_joblib_dump(obj, path: str) -> None:
    if joblib is None:
        raise RuntimeError("joblib is required to persist competitive models.")
    tmp = path + ".tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def safe_joblib_load(path: str):
    if joblib is None:
        raise RuntimeError("joblib is required to load competitive models.")
    return joblib.load(path)


def system_label_from_arrays(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return np.array(["Benign" if int(a) == 0 else str(b) for a, b in zip(y1, y2)], dtype=object)


def ordered_labels(labels: Iterable[str]) -> List[str]:
    uniq = sorted({str(x) for x in labels if str(x) and str(x).lower() != "nan"})
    return ["Benign"] + [x for x in uniq if x != "Benign"]


def collect_direct_data(helper, ds_dir: str, split: str, prep, labels: Optional[List[str]], max_rows: int, seed: int) -> DirectData:
    parts = helper.list_parts(ds_dir, split)
    if not parts:
        raise RuntimeError(f"No {split} parts found under {ds_dir}")

    rng = np.random.RandomState(int(seed))
    usecols = prep.num_cols + prep.cat_cols + [helper.CFG["y_stage1"], helper.CFG["y_stage2"]]
    usecols = [helper.canonical_col(c) for c in usecols]
    total_rows = estimate_part_rows(parts)
    if total_rows is not None and total_rows > 0:
        sample_prob = 1.0 if int(max_rows) >= int(total_rows) else min(1.0, (float(max_rows) / float(total_rows)) * 1.25)
    else:
        sample_prob = 1.0

    X_list: List[csr_matrix] = []
    y1_list: List[np.ndarray] = []
    y2_list: List[np.ndarray] = []
    seen = 0
    for chunk in helper.iter_rows_from_parts(parts, usecols=usecols, chunksize=int(CFG["chunksize_rows"])):
        chunk.columns = [helper.canonical_col(c) for c in chunk.columns]
        if sample_prob < 1.0:
            keep = rng.random(len(chunk)) < sample_prob
            chunk = chunk.loc[keep]
            if chunk.empty:
                continue
        elif total_rows is None:
            remaining = int(max_rows) - seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                idx = rng.choice(len(chunk), size=remaining, replace=False)
                chunk = chunk.iloc[idx]

        y1_chunk = chunk[helper.CFG["y_stage1"]].astype(int).to_numpy()
        y2_chunk = chunk[helper.CFG["y_stage2"]].astype(str).fillna("").to_numpy()
        X_chunk = prep.transform(chunk.drop(columns=[helper.CFG["y_stage1"], helper.CFG["y_stage2"]], errors="ignore"))
        X_list.append(X_chunk)
        y1_list.append(y1_chunk)
        y2_list.append(y2_chunk)
        seen += len(chunk)

    if not X_list:
        X = csr_matrix((0, prep.n_num + prep.n_cat), dtype=np.float64)
        y1 = np.array([], dtype=int)
        y2 = np.array([], dtype=object)
    else:
        from scipy.sparse import vstack

        X = vstack(X_list, format="csr")
        y1 = np.concatenate(y1_list).astype(int)
        y2 = np.concatenate(y2_list).astype(object)
        if len(y1) > int(max_rows):
            idx = rng.choice(len(y1), size=int(max_rows), replace=False)
            X = X[idx]
            y1 = y1[idx]
            y2 = y2[idx]

    y_label = system_label_from_arrays(y1, y2)
    if labels is None:
        labels = ordered_labels(y_label)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    unseen = sorted({str(y) for y in y_label if str(y) not in label_to_idx})
    if unseen:
        labels = list(labels) + unseen
        label_to_idx = {label: i for i, label in enumerate(labels)}
    y_idx = np.array([label_to_idx[str(y)] for y in y_label], dtype=int)
    y_binary = np.array([0 if str(y) == "Benign" else 1 for y in y_label], dtype=int)
    return DirectData(X=X, y_idx=y_idx, y_label=y_label, y_binary=y_binary, labels=list(labels))


def estimate_part_rows(parts: Sequence[str]) -> Optional[int]:
    total = 0
    for part in parts:
        low = str(part).lower()
        if low.endswith(".parquet") and pq is not None:
            total += int(pq.ParquetFile(part).metadata.num_rows)
        else:
            return None
    return int(total)


def class_weight_vector(y_idx: np.ndarray) -> np.ndarray:
    sw = compute_sample_weight(class_weight="balanced", y=y_idx).astype(np.float64)
    median = float(np.median(sw)) if len(sw) else 1.0
    if median > 0:
        sw = sw / median
    return np.clip(sw, 0.10, 25.0)


def class_prior_weights(y_idx: np.ndarray, n_classes: int) -> Dict[str, float]:
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    total = float(np.sum(counts))
    if total <= 0:
        return {str(i): 1.0 for i in range(n_classes)}
    raw = total / np.maximum(1.0, counts * n_classes)
    raw = np.clip(raw, 0.10, 25.0)
    return {str(i): float(raw[i]) for i in range(n_classes)}


def build_direct_model(candidate: str, n_classes: int):
    seed = int(CFG["random_seed"])
    n_jobs = int(CFG["n_jobs"])
    if candidate == "rf_balanced":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if candidate == "extra_trees_balanced":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if candidate == "balanced_rf":
        if BalancedRandomForestClassifier is None:
            raise RuntimeError("imblearn BalancedRandomForestClassifier is not available.")
        return BalancedRandomForestClassifier(
            n_estimators=220,
            max_depth=None,
            min_samples_leaf=1,
            sampling_strategy="not majority",
            replacement=True,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if candidate == "easy_ensemble":
        if EasyEnsembleClassifier is None:
            raise RuntimeError("imblearn EasyEnsembleClassifier is not available.")
        return EasyEnsembleClassifier(
            n_estimators=12,
            sampling_strategy="not majority",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if candidate == "xgb_weighted":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not available.")
        return XGBClassifier(
            objective="multi:softprob",
            num_class=int(n_classes),
            n_estimators=450,
            max_depth=8,
            learning_rate=0.045,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_lambda=1.2,
            tree_method="hist",
            predictor="cpu_predictor",
            n_jobs=n_jobs,
            random_state=seed,
            eval_metric="mlogloss",
            verbosity=0,
        )
    if candidate == "lgbm_weighted":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not available.")
        return LGBMClassifier(
            objective="multiclass",
            num_class=int(n_classes),
            n_estimators=450,
            learning_rate=0.050,
            num_leaves=127,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=-1,
        )
    if candidate == "catboost_weighted":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not available.")
        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=450,
            depth=8,
            learning_rate=0.060,
            l2_leaf_reg=4.0,
            auto_class_weights="Balanced",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown direct candidate: {candidate}")


def maybe_fit_selector(X: csr_matrix, y: np.ndarray, out_dir: str):
    cfg = dict(CFG.get("feature_selection", {}) or {})
    if not bool(cfg.get("enabled", False)):
        return None
    k = int(cfg.get("k", 80))
    if X.shape[1] <= k:
        return None
    max_rows = min(int(cfg.get("max_fit_rows", 250_000)), X.shape[0])
    X_fit = X[:max_rows]
    y_fit = y[:max_rows]
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_fit, y_fit)
    safe_joblib_dump(selector, os.path.join(out_dir, "feature_selector.joblib"))
    return selector


def ensure_probability_width(proba: np.ndarray, model_classes: Sequence[int], n_classes: int) -> np.ndarray:
    out = np.zeros((proba.shape[0], int(n_classes)), dtype=np.float64)
    for j, cls in enumerate(model_classes):
        if int(cls) < n_classes:
            out[:, int(cls)] = proba[:, j]
    row_sum = out.sum(axis=1)
    zero = row_sum <= 0
    if np.any(zero):
        out[zero, :] = 1.0 / float(n_classes)
        row_sum = out.sum(axis=1)
    return out / row_sum[:, None]


def predict_proba_aligned(model, X: csr_matrix, n_classes: int) -> np.ndarray:
    proba = model.predict_proba(X).astype(np.float64)
    classes = getattr(model, "classes_", np.arange(proba.shape[1]))
    return ensure_probability_width(proba, classes, n_classes)


def apply_biases(proba: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    adjusted = proba * multipliers[None, :]
    row_sum = np.maximum(1e-12, adjusted.sum(axis=1))
    return adjusted / row_sum[:, None]


def metric_payload(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, labels: List[str]) -> Dict[str, object]:
    acc = float(accuracy_score(y_true_idx, y_pred_idx))
    macro = float(f1_score(y_true_idx, y_pred_idx, average="macro", labels=list(range(len(labels))), zero_division=0))
    weighted = float(f1_score(y_true_idx, y_pred_idx, average="weighted", labels=list(range(len(labels))), zero_division=0))
    y_true_bin = np.array([0 if labels[i] == "Benign" else 1 for i in y_true_idx], dtype=int)
    y_pred_bin = np.array([0 if labels[i] == "Benign" else 1 for i in y_pred_idx], dtype=int)
    binary_f1 = float(f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0))
    binary_macro = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))
    benign_mask = y_true_bin == 0
    attack_mask = y_true_bin == 1
    benign_fp = float(np.mean(y_pred_bin[benign_mask] == 1)) if np.any(benign_mask) else 0.0
    attack_fn = float(np.mean(y_pred_bin[attack_mask] == 0)) if np.any(attack_mask) else 0.0
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "binary_attack_f1": binary_f1,
        "binary_macro_f1": binary_macro,
        "benign_family_fp_rate": benign_fp,
        "attack_to_benign_rate": attack_fn,
    }


def tune_probability_biases(
    y_true_idx: np.ndarray,
    proba: np.ndarray,
    labels: List[str],
    dataset: str,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, object]]:
    base_pred = np.argmax(proba, axis=1).astype(int)
    base = metric_payload(y_true_idx, base_pred, labels)
    best_metric = float(base["macro_f1"])
    min_acc = float(base["accuracy"]) - float(CFG["max_bias_accuracy_drop"])
    multipliers = np.ones(len(labels), dtype=np.float64)
    grid = [float(x) for x in CFG["probability_bias_grid"]]
    candidate_labels = [x for x in CFG["weak_family_candidates"].get(dataset, ["Benign"]) if x in labels]
    candidate_indices = [labels.index(x) for x in candidate_labels]
    rows: List[Dict[str, object]] = []

    for pass_idx in range(int(CFG["probability_bias_passes"])):
        improved = False
        for label, label_idx in zip(candidate_labels, candidate_indices):
            local_best_value = float(multipliers[label_idx])
            local_best_metric = best_metric
            local_best_payload = None
            for value in grid:
                trial = multipliers.copy()
                trial[label_idx] = value
                pred = np.argmax(apply_biases(proba, trial), axis=1).astype(int)
                payload = metric_payload(y_true_idx, pred, labels)
                row = {
                    "pass": int(pass_idx + 1),
                    "label": label,
                    "multiplier": float(value),
                    **payload,
                }
                rows.append(row)
                if float(payload["accuracy"]) < min_acc:
                    continue
                if float(payload["macro_f1"]) > local_best_metric + 1e-12:
                    local_best_metric = float(payload["macro_f1"])
                    local_best_value = float(value)
                    local_best_payload = payload
            if local_best_metric > best_metric + 1e-12:
                multipliers[label_idx] = local_best_value
                best_metric = local_best_metric
                improved = True
            if local_best_payload is not None:
                rows.append({"pass": int(pass_idx + 1), "label": label, "selected_multiplier": local_best_value, **local_best_payload})
        if not improved:
            break

    tuned_pred = np.argmax(apply_biases(proba, multipliers), axis=1).astype(int)
    tuned = metric_payload(y_true_idx, tuned_pred, labels)
    meta = {
        "base": base,
        "tuned": tuned,
        "multipliers": {labels[i]: float(multipliers[i]) for i in range(len(labels)) if not math.isclose(float(multipliers[i]), 1.0)},
        "candidate_labels": candidate_labels,
    }
    return multipliers, meta, {"rows": rows}


def write_classification_outputs(out_dir: str, prefix: str, y_true_idx: np.ndarray, y_pred_idx: np.ndarray, labels: List[str]) -> None:
    report = classification_report(
        y_true_idx,
        y_pred_idx,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).transpose().to_csv(os.path.join(out_dir, f"{prefix}_classification_report.csv"))
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(labels))))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(out_dir, f"{prefix}_confusion_matrix.csv"))


def fit_candidate(candidate: str, Xtr: csr_matrix, ytr: np.ndarray, n_classes: int):
    model = build_direct_model(candidate, n_classes)
    sw = class_weight_vector(ytr)
    if candidate in {"rf_balanced", "extra_trees_balanced", "xgb_weighted"}:
        model.fit(Xtr, ytr, sample_weight=sw)
    elif candidate == "lgbm_weighted":
        model.fit(Xtr, ytr, sample_weight=sw)
    else:
        model.fit(Xtr, ytr)
    return model


def run_direct_multiclass(helper, dataset: str, ds_dir: str, ds_out: str, prep) -> pd.DataFrame:
    direct_dir = os.path.join(ds_out, "direct_multiclass")
    models_dir = os.path.join(direct_dir, "models")
    safe_mkdir(models_dir)

    train = collect_direct_data(
        helper,
        ds_dir,
        "train",
        prep,
        labels=None,
        max_rows=int(CFG["max_train_rows"][dataset]),
        seed=int(CFG["random_seed"]),
    )
    val = collect_direct_data(
        helper,
        ds_dir,
        "val",
        prep,
        labels=train.labels,
        max_rows=int(CFG["max_val_rows"][dataset]),
        seed=int(CFG["random_seed"]) + 1,
    )
    labels = val.labels
    n_classes = len(labels)
    save_json(
        os.path.join(direct_dir, "label_mapping.json"),
        {
            "labels": labels,
            "n_classes": n_classes,
            "train_rows": int(train.X.shape[0]),
            "val_rows": int(val.X.shape[0]),
            "class_prior_weights": class_prior_weights(train.y_idx, n_classes),
        },
    )

    selector = maybe_fit_selector(train.X, train.y_idx, direct_dir)
    Xtr = selector.transform(train.X) if selector is not None else train.X
    Xva = selector.transform(val.X) if selector is not None else val.X

    val_rows: List[Dict[str, object]] = []
    proba_cache: Dict[str, np.ndarray] = {}
    usable_candidates: List[str] = []

    for candidate in list(CFG["direct_candidates"]):
        candidate_dir = os.path.join(direct_dir, candidate)
        safe_mkdir(candidate_dir)
        started = time.perf_counter()
        fit_seconds = None
        predict_seconds = None
        try:
            model = fit_candidate(str(candidate), Xtr, train.y_idx, n_classes)
            fit_seconds = time.perf_counter() - started
            safe_joblib_dump(model, os.path.join(models_dir, f"{candidate}.joblib"))

            pred_started = time.perf_counter()
            proba = predict_proba_aligned(model, Xva, n_classes)
            predict_seconds = time.perf_counter() - pred_started
            multipliers, bias_meta, trace = tune_probability_biases(val.y_idx, proba, labels, dataset)
            proba_tuned = apply_biases(proba, multipliers)
            pred = np.argmax(proba_tuned, axis=1).astype(int)
            payload = metric_payload(val.y_idx, pred, labels)
            write_classification_outputs(candidate_dir, "val", val.y_idx, pred, labels)
            pd.DataFrame(trace["rows"]).to_csv(os.path.join(candidate_dir, "bias_tuning_trace.csv"), index=False)
            save_json(os.path.join(candidate_dir, "probability_bias.json"), bias_meta)
            proba_cache[str(candidate)] = proba_tuned
            usable_candidates.append(str(candidate))
            row = {
                "dataset": dataset,
                "surface": "direct_multiclass",
                "candidate": str(candidate),
                "claim_status": "literature-comparable",
                "split": "val",
                "n_train_rows": int(train.X.shape[0]),
                "n_eval_rows": int(val.X.shape[0]),
                "n_features": int(Xtr.shape[1]),
                "fit_seconds": float(fit_seconds),
                "predict_seconds": float(predict_seconds),
                "rss_mb": rss_mb(),
                "status": "ok",
                **payload,
            }
        except Exception as exc:
            row = {
                "dataset": dataset,
                "surface": "direct_multiclass",
                "candidate": str(candidate),
                "claim_status": "exploratory",
                "split": "val",
                "n_train_rows": int(train.X.shape[0]),
                "n_eval_rows": int(val.X.shape[0]),
                "n_features": int(Xtr.shape[1]),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "rss_mb": rss_mb(),
                "status": "skipped_or_failed",
                "error": str(exc),
            }
        val_rows.append(row)
        pd.DataFrame(val_rows).to_csv(os.path.join(direct_dir, "candidate_results_val.csv"), index=False)

    if usable_candidates:
        ranked = pd.DataFrame([r for r in val_rows if r.get("status") == "ok"]).sort_values(
            ["macro_f1", "accuracy", "weighted_f1"],
            ascending=[False, False, False],
        )
        top = ranked.head(min(3, len(ranked)))["candidate"].astype(str).tolist()
        if len(top) >= 2:
            ensemble_proba = np.mean([proba_cache[name] for name in top], axis=0)
            multipliers, bias_meta, trace = tune_probability_biases(val.y_idx, ensemble_proba, labels, dataset)
            pred = np.argmax(apply_biases(ensemble_proba, multipliers), axis=1).astype(int)
            payload = metric_payload(val.y_idx, pred, labels)
            ensemble_dir = os.path.join(direct_dir, "soft_vote_top3")
            safe_mkdir(ensemble_dir)
            write_classification_outputs(ensemble_dir, "val", val.y_idx, pred, labels)
            save_json(os.path.join(ensemble_dir, "probability_bias.json"), bias_meta)
            save_json(os.path.join(ensemble_dir, "ensemble_members.json"), {"members": top})
            pd.DataFrame(trace["rows"]).to_csv(os.path.join(ensemble_dir, "bias_tuning_trace.csv"), index=False)
            val_rows.append(
                {
                    "dataset": dataset,
                    "surface": "direct_multiclass",
                    "candidate": "soft_vote_top3",
                    "claim_status": "literature-comparable",
                    "split": "val",
                    "n_train_rows": int(train.X.shape[0]),
                    "n_eval_rows": int(val.X.shape[0]),
                    "n_features": int(Xtr.shape[1]),
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "rss_mb": rss_mb(),
                    "status": "ok",
                    **payload,
                }
            )
            pd.DataFrame(val_rows).to_csv(os.path.join(direct_dir, "candidate_results_val.csv"), index=False)

    ok = pd.DataFrame([r for r in val_rows if r.get("status") == "ok"])
    if ok.empty:
        return pd.DataFrame(val_rows)
    winner = ok.sort_values(["macro_f1", "accuracy", "weighted_f1"], ascending=[False, False, False]).iloc[0].to_dict()
    winner_name = str(winner["candidate"])
    save_json(os.path.join(direct_dir, "selected_winner.json"), winner)

    test = collect_direct_data(
        helper,
        ds_dir,
        "test",
        prep,
        labels=labels,
        max_rows=int(CFG["max_test_rows"][dataset]),
        seed=int(CFG["random_seed"]) + 2,
    )
    eval_labels = test.labels
    Xte = selector.transform(test.X) if selector is not None else test.X
    test_started = time.perf_counter()
    if winner_name == "soft_vote_top3":
        member_meta = json.load(open(os.path.join(direct_dir, "soft_vote_top3", "ensemble_members.json"), "r", encoding="utf-8"))
        probs = []
        for member in member_meta["members"]:
            model = safe_joblib_load(os.path.join(models_dir, f"{member}.joblib"))
            probs.append(predict_proba_aligned(model, Xte, len(eval_labels)))
        proba_test = np.mean(probs, axis=0)
        bias_meta = json.load(open(os.path.join(direct_dir, "soft_vote_top3", "probability_bias.json"), "r", encoding="utf-8"))
    else:
        model = safe_joblib_load(os.path.join(models_dir, f"{winner_name}.joblib"))
        proba_test = predict_proba_aligned(model, Xte, len(eval_labels))
        bias_meta = json.load(open(os.path.join(direct_dir, winner_name, "probability_bias.json"), "r", encoding="utf-8"))
    multipliers = np.ones(len(eval_labels), dtype=np.float64)
    for label, value in dict(bias_meta.get("multipliers", {})).items():
        if label in eval_labels:
            multipliers[eval_labels.index(label)] = float(value)
    pred_test = np.argmax(apply_biases(proba_test, multipliers), axis=1).astype(int)
    predict_seconds = time.perf_counter() - test_started
    test_payload = metric_payload(test.y_idx, pred_test, eval_labels)
    winner_dir = os.path.join(direct_dir, winner_name)
    safe_mkdir(winner_dir)
    write_classification_outputs(winner_dir, "test", test.y_idx, pred_test, eval_labels)
    test_row = {
        "dataset": dataset,
        "surface": "direct_multiclass",
        "candidate": winner_name,
        "claim_status": "literature-comparable",
        "split": "test",
        "n_train_rows": int(train.X.shape[0]),
        "n_eval_rows": int(test.X.shape[0]),
        "n_features": int(Xtr.shape[1]),
        "fit_seconds": float(winner.get("fit_seconds", 0.0) or 0.0),
        "predict_seconds": float(predict_seconds),
        "rss_mb": rss_mb(),
        "status": "winner_test",
        **test_payload,
    }
    pd.DataFrame([test_row]).to_csv(os.path.join(direct_dir, "winner_results_test.csv"), index=False)
    return pd.concat([pd.DataFrame(val_rows), pd.DataFrame([test_row])], ignore_index=True)


def latest_complete_baseline_run(dataset: str, model_family: str) -> Optional[str]:
    root = str(CFG["baseline_runs_root"])
    prefix = f"{CFG['protocol']}__{dataset}__{model_family}__"
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not name.startswith(prefix):
            continue
        required = ["stage1_best.joblib", "stage2_best.joblib", "families.json", "system_compare_val.json", "system_compare_test.json"]
        if all(os.path.exists(os.path.join(path, r)) for r in required):
            candidates.append(path)
    return sorted(candidates, reverse=True)[0] if candidates else None


def collect_two_stage_scores(helper, ds_dir: str, split: str, prep, stage1, platt, stage2, families: List[str], T: float, max_rows: int) -> ScoreBlock:
    parts = helper.list_parts(ds_dir, split)
    if not parts:
        raise RuntimeError(f"No {split} parts found under {ds_dir}")
    row_ids: List[str] = []
    y_true: List[str] = []
    p_attack_list: List[np.ndarray] = []
    fam_idx_list: List[np.ndarray] = []
    pmax_list: List[np.ndarray] = []
    seen = 0
    usecols = prep.num_cols + prep.cat_cols + [helper.CFG["y_stage1"], helper.CFG["y_stage2"]]
    usecols = [helper.canonical_col(c) for c in usecols]
    for part in parts:
        part_name = os.path.basename(part)
        for chunk_idx, chunk in enumerate(helper.iter_rows_from_parts([part], usecols=usecols, chunksize=int(CFG["chunksize_rows"]))):
            chunk.columns = [helper.canonical_col(c) for c in chunk.columns]
            remaining = max_rows - seen
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
            y1 = chunk[helper.CFG["y_stage1"]].astype(int).to_numpy()
            y2 = chunk[helper.CFG["y_stage2"]].astype(str).fillna("").to_numpy()
            labels = system_label_from_arrays(y1, y2)
            X = prep.transform(chunk.drop(columns=[helper.CFG["y_stage1"], helper.CFG["y_stage2"]], errors="ignore"))
            p_attack = stage1.predict_proba(X)[:, 1].astype(np.float64)
            p_attack = helper.apply_platt(p_attack, platt)
            p2 = stage2.predict_proba(X).astype(np.float64)
            p2 = helper.apply_temperature(p2, T)
            fam_idx = np.argmax(p2, axis=1).astype(int)
            pmax = np.max(p2, axis=1).astype(np.float64)
            start = seen
            row_ids.extend([f"{split}:{part_name}:{start + i}" for i in range(len(chunk))])
            y_true.extend(list(labels))
            p_attack_list.append(p_attack)
            fam_idx_list.append(fam_idx)
            pmax_list.append(pmax)
            seen += len(chunk)
            if seen >= max_rows:
                break
        if seen >= max_rows:
            break
    return ScoreBlock(
        row_ids=np.array(row_ids, dtype=object),
        y_true=np.array(y_true, dtype=object),
        p_attack=np.concatenate(p_attack_list) if p_attack_list else np.array([], dtype=float),
        fam_pred_idx=np.concatenate(fam_idx_list) if fam_idx_list else np.array([], dtype=int),
        fam_pmax=np.concatenate(pmax_list) if pmax_list else np.array([], dtype=float),
        families=list(families),
    )


def two_stage_predict(block: ScoreBlock, thr: float, tau: float, use_unknown: bool) -> np.ndarray:
    labels = np.array(["Benign"] + block.families, dtype=object)
    pred = np.empty(len(block.y_true), dtype=object)
    attack = block.p_attack >= float(thr)
    pred[~attack] = "Benign"
    if np.any(attack):
        fam_labels = labels[1:][block.fam_pred_idx[attack]]
        pred[attack] = fam_labels
        if use_unknown and tau > 0:
            reject_idx = np.where(attack)[0][block.fam_pmax[attack] < float(tau)]
            pred[reject_idx] = "Unknown"
    return pred


def two_stage_metrics(block: ScoreBlock, pred: np.ndarray, use_unknown: bool) -> Dict[str, object]:
    labels = ["Benign"] + block.families + (["Unknown"] if use_unknown else [])
    y_true = block.y_true
    acc = float(accuracy_score(y_true, pred))
    macro = float(f1_score(y_true, pred, average="macro", labels=labels, zero_division=0))
    weighted = float(f1_score(y_true, pred, average="weighted", labels=labels, zero_division=0))
    y_true_bin = np.array([0 if str(y) == "Benign" else 1 for y in y_true], dtype=int)
    y_pred_bin = np.array([0 if str(y) == "Benign" else 1 for y in pred], dtype=int)
    benign_mask = y_true_bin == 0
    attack_mask = y_true_bin == 1
    reject_mask = pred == "Unknown"
    return {
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "binary_attack_f1": float(f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0)),
        "binary_macro_f1": float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)),
        "benign_family_fp_rate": float(np.mean(y_pred_bin[benign_mask] == 1)) if np.any(benign_mask) else 0.0,
        "attack_to_benign_rate": float(np.mean(y_pred_bin[attack_mask] == 0)) if np.any(attack_mask) else 0.0,
        "overall_reject_rate": float(np.mean(reject_mask)),
    }


def threshold_candidates_from_fpr(block: ScoreBlock) -> List[float]:
    benign = block.p_attack[block.y_true == "Benign"]
    if len(benign) == 0:
        return [0.5]
    out = []
    for target in CFG["stage1_threshold_grid"]:
        q = float(np.clip(1.0 - float(target), 0.0, 1.0))
        out.append(float(np.quantile(benign, q)))
    out.extend([0.20, 0.35, 0.50, 0.65, 0.80])
    return sorted({round(float(x), 12) for x in out})


def write_two_stage_report(out_dir: str, prefix: str, block: ScoreBlock, pred: np.ndarray, use_unknown: bool) -> None:
    labels = ["Benign"] + block.families + (["Unknown"] if use_unknown else [])
    report = classification_report(block.y_true, pred, labels=labels, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(out_dir, f"{prefix}_classification_report.csv"))
    cm = confusion_matrix(block.y_true, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(out_dir, f"{prefix}_confusion_matrix.csv"))


def run_two_stage_overlay(helper, dataset: str, ds_dir: str, ds_out: str) -> pd.DataFrame:
    overlay_dir = os.path.join(ds_out, "two_stage_overlay")
    safe_mkdir(overlay_dir)
    rows: List[Dict[str, object]] = []
    test_rows: List[Dict[str, object]] = []

    for model_family in CFG["baseline_overlay_models"]:
        base_run = latest_complete_baseline_run(dataset, str(model_family))
        if base_run is None:
            rows.append(
                {
                    "dataset": dataset,
                    "surface": "two_stage_overlay",
                    "candidate": f"{model_family}_threshold_overlay",
                    "claim_status": "thesis-safe",
                    "split": "val",
                    "status": "missing_baseline_run",
                }
            )
            continue
        candidate_dir = os.path.join(overlay_dir, str(model_family))
        safe_mkdir(candidate_dir)
        prep = safe_joblib_load(os.path.join(base_run, "preprocessor.joblib"))
        stage1 = safe_joblib_load(os.path.join(base_run, "stage1_best.joblib"))
        stage2 = safe_joblib_load(os.path.join(base_run, "stage2_best.joblib"))
        platt_path = os.path.join(base_run, "stage1_platt.json")
        platt = json.load(open(platt_path, "r", encoding="utf-8")).get("platt") if os.path.exists(platt_path) else None
        families = json.load(open(os.path.join(base_run, "families.json"), "r", encoding="utf-8"))
        T = float(json.load(open(os.path.join(base_run, "stage2_temperature.json"), "r", encoding="utf-8")).get("T", 1.0))
        block_val = collect_two_stage_scores(
            helper,
            ds_dir,
            "val",
            prep,
            stage1,
            platt,
            stage2,
            families,
            T,
            int(CFG["max_val_rows"][dataset]),
        )
        baseline_compare = json.load(open(os.path.join(base_run, "system_compare_val.json"), "r", encoding="utf-8"))
        baseline_strict = baseline_compare.get("strict", {})
        min_acc = float(baseline_strict.get("accuracy", 0.0)) - float(CFG["two_stage_max_accuracy_drop"])
        best = None
        trace: List[Dict[str, object]] = []
        for use_unknown in [False, True]:
            tau_values = [0.0] if not use_unknown else [float(x) for x in CFG["two_stage_tau_grid"]]
            for thr in threshold_candidates_from_fpr(block_val):
                for tau in tau_values:
                    pred = two_stage_predict(block_val, float(thr), float(tau), bool(use_unknown))
                    payload = two_stage_metrics(block_val, pred, bool(use_unknown))
                    row = {
                        "dataset": dataset,
                        "surface": "two_stage_overlay",
                        "candidate": f"{model_family}_{'reject' if use_unknown else 'closed'}_threshold_overlay",
                        "claim_status": "thesis-safe" if not use_unknown else "exploratory",
                        "split": "val",
                        "baseline_run": base_run,
                        "threshold": float(thr),
                        "tau": float(tau),
                        "use_unknown": bool(use_unknown),
                        "n_eval_rows": int(len(block_val.y_true)),
                        "status": "ok",
                        **payload,
                    }
                    trace.append(row)
                    if float(payload["benign_family_fp_rate"]) > float(CFG["two_stage_max_benign_family_fp_rate"]):
                        continue
                    if float(payload["accuracy"]) < min_acc:
                        continue
                    if best is None or float(payload["macro_f1"]) > float(best["macro_f1"]) + 1e-12:
                        best = row
                    elif best is not None and math.isclose(float(payload["macro_f1"]), float(best["macro_f1"])):
                        if float(payload["accuracy"]) > float(best["accuracy"]):
                            best = row
        pd.DataFrame(trace).to_csv(os.path.join(candidate_dir, "threshold_trace_val.csv"), index=False)
        if best is None and trace:
            best = sorted(trace, key=lambda r: (float(r["macro_f1"]), float(r["accuracy"])), reverse=True)[0]
        if best is None:
            continue
        rows.append(best)
        save_json(os.path.join(candidate_dir, "selected_overlay.json"), best)
        pred_val = two_stage_predict(block_val, float(best["threshold"]), float(best["tau"]), bool(best["use_unknown"]))
        write_two_stage_report(candidate_dir, "val_selected", block_val, pred_val, bool(best["use_unknown"]))

        block_test = collect_two_stage_scores(
            helper,
            ds_dir,
            "test",
            prep,
            stage1,
            platt,
            stage2,
            families,
            T,
            int(CFG["max_test_rows"][dataset]),
        )
        pred_test = two_stage_predict(block_test, float(best["threshold"]), float(best["tau"]), bool(best["use_unknown"]))
        test_payload = two_stage_metrics(block_test, pred_test, bool(best["use_unknown"]))
        write_two_stage_report(candidate_dir, "test_selected", block_test, pred_test, bool(best["use_unknown"]))
        test_row = {
            **{k: v for k, v in best.items() if k not in test_payload and k not in {"split", "n_eval_rows"}},
            "split": "test",
            "n_eval_rows": int(len(block_test.y_true)),
            "status": "winner_test",
            **test_payload,
        }
        test_rows.append(test_row)

    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(test_rows)], ignore_index=True) if (rows or test_rows) else pd.DataFrame()
    out.to_csv(os.path.join(overlay_dir, "overlay_results.csv"), index=False)
    return out


def run_dataset(helper, dataset: str) -> pd.DataFrame:
    ds_dir = os.path.join(str(CFG["processed_root"]), str(CFG["protocol"]), dataset)
    if not os.path.isdir(ds_dir):
        raise RuntimeError(f"Processed dataset folder not found: {ds_dir}")
    ds_out = os.path.join(str(CFG["runs_root"]), dataset)
    safe_mkdir(ds_out)
    save_json(
        os.path.join(ds_out, "run_config.json"),
        {
            "dataset": dataset,
            "started_at": now_stamp(),
            "processed_dir": ds_dir,
            "cfg": CFG,
        },
    )
    prep_path = os.path.join(ds_out, "preprocessor.joblib")
    if os.path.exists(prep_path):
        prep = safe_joblib_load(prep_path)
    else:
        prep = helper.fit_preprocessor(ds_dir, ds_out)

    direct = run_direct_multiclass(helper, dataset, ds_dir, ds_out, prep)
    overlay = run_two_stage_overlay(helper, dataset, ds_dir, ds_out)
    pieces = [x for x in [direct, overlay] if x is not None and not x.empty]
    if not pieces:
        return pd.DataFrame()
    dataset_summary = pd.concat(pieces, ignore_index=True)
    dataset_summary.to_csv(os.path.join(ds_out, "competitive_results_all.csv"), index=False)
    return dataset_summary


def write_root_summary(frames: List[pd.DataFrame]) -> None:
    summary_dir = os.path.join(str(CFG["runs_root"]), str(CFG["summary_dirname"]))
    safe_mkdir(summary_dir)
    current_datasets = {
        str(df["dataset"].iloc[0])
        for df in frames
        if df is not None and not df.empty and "dataset" in df.columns
    }
    preserved: List[pd.DataFrame] = []
    if os.path.isdir(str(CFG["runs_root"])):
        for name in sorted(os.listdir(str(CFG["runs_root"]))):
            if name in current_datasets or name == str(CFG["summary_dirname"]):
                continue
            path = os.path.join(str(CFG["runs_root"]), name, "competitive_results_all.csv")
            if os.path.exists(path):
                try:
                    preserved.append(pd.read_csv(path))
                except Exception:
                    pass
    pieces = [df for df in frames if df is not None and not df.empty] + preserved
    if pieces:
        all_rows = pd.concat(pieces, ignore_index=True)
    else:
        all_rows = pd.DataFrame()
    all_rows.to_csv(os.path.join(summary_dir, "competitive_results_all.csv"), index=False)
    if not all_rows.empty:
        val = all_rows[(all_rows["split"] == "val") & (all_rows["status"] == "ok")].copy()
        if not val.empty:
            val = val.sort_values(["dataset", "macro_f1", "accuracy"], ascending=[True, False, False])
            val.to_csv(os.path.join(summary_dir, "validation_leaderboard.csv"), index=False)
        test = all_rows[all_rows["split"] == "test"].copy()
        if not test.empty:
            test = test.sort_values(["dataset", "macro_f1", "accuracy"], ascending=[True, False, False])
            test.to_csv(os.path.join(summary_dir, "winner_test_results.csv"), index=False)
    manifest = []
    for root, _, files in os.walk(str(CFG["runs_root"])):
        for name in files:
            path = os.path.join(root, name)
            manifest.append({"path": path, "bytes": int(os.path.getsize(path))})
    pd.DataFrame(manifest).sort_values("path").to_csv(os.path.join(summary_dir, "output_manifest.csv"), index=False)


def main() -> None:
    apply_smoke_overrides()
    apply_runtime_overrides()
    safe_mkdir(str(CFG["runs_root"]))
    helper = load_helper_module(str(CFG["helper_script"]))
    configure_helper(helper)
    if hasattr(helper, "ensure_deps"):
        helper.ensure_deps()
    frames: List[pd.DataFrame] = []
    started = time.perf_counter()
    for dataset in list(CFG["datasets"]):
        print(f"\n=== Competitive metrics: {dataset} ===")
        ds_frame = run_dataset(helper, str(dataset))
        frames.append(ds_frame)
    write_root_summary(frames)
    save_json(
        os.path.join(str(CFG["runs_root"]), str(CFG["summary_dirname"]), "run_complete.json"),
        {
            "completed_at": now_stamp(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "datasets": list(CFG["datasets"]),
            "runs_root": str(CFG["runs_root"]),
        },
    )
    print(f"Wrote competitive metrics root: {CFG['runs_root']}")


if __name__ == "__main__":
    main()
