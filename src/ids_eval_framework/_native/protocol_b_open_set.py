#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import re
import shutil
import threading
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ids_eval_framework._native.full_scope_utils import load_module, resolve_repo_path, safe_mkdir, split_variant_name_for_dataset


CFG: Dict[str, object] = {
    "best_csv": os.path.join("thesis_core_pack", "protocol_b_best_per_holdout_standardized.csv"),
    "audit_roots": ["protocolB_support_audit_out", "protocolB_support_audit_out_cicids17_recovery"],
    "base_helper_script": "ids_eval_framework.src.two_stage_engine",
    "grid_script": "5.ProtocolB_GridRunner_V1.2.py",
    "runs_root": "protocolB_grid_runs step 5 - open-set baselines",
    "summary_dirname": "summary",
    "selected_cases": [],
    "max_train_rows": {"CICIDS2017": 900_000, "CICIoT2023": 1_500_000},
    "max_val_rows": {"CICIDS2017": 500_000, "CICIoT2023": 700_000},
    "max_test_rows": {"CICIDS2017": None, "CICIoT2023": None},
    "tau_grid": [0.10, 0.30, 0.50, 0.70, 0.80, 0.90],
    "margin_quantiles": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "entropy_quantiles": [0.60, 0.70, 0.80, 0.90, 0.95],
    "conformal_alpha_grid": [0.01, 0.05, 0.10, 0.15, 0.20],
    "bootstrap_resamples": 1_000,
    "random_seed": 123,
    "case_parallel_workers": None,
    "threads_per_worker": None,
    "parallel_backend": "process",
    "xgb_device": "auto",
}


_HELPER_MODULE_LOCK = threading.Lock()
_HELPER_MODULE_CACHE: Dict[str, object] = {}


def safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass


def load_cached_helper_module(grid_mod, path: str):
    """Load the legacy helper once per process; its fixed module name is not thread-safe."""
    key = os.path.normcase(os.path.abspath(path))
    with _HELPER_MODULE_LOCK:
        helper = _HELPER_MODULE_CACHE.get(key)
        if helper is None:
            helper = grid_mod.load_helper_module(path)
            _HELPER_MODULE_CACHE[key] = helper
        return helper


def apply_env_overrides() -> None:
    runs_root = os.environ.get("IDS_STEP5_RUNS_ROOT")
    if runs_root:
        CFG["runs_root"] = runs_root

    selected_cases_json = os.environ.get("IDS_STEP5_SELECTED_CASES_JSON")
    if selected_cases_json:
        parsed = json.loads(selected_cases_json)
        if not isinstance(parsed, list):
            raise ValueError("IDS_STEP5_SELECTED_CASES_JSON must decode to a JSON list.")
        CFG["selected_cases"] = [str(x) for x in parsed]

    for env_name, cfg_key in [
        ("IDS_STEP5_CASE_WORKERS", "case_parallel_workers"),
        ("IDS_STEP5_THREADS_PER_WORKER", "threads_per_worker"),
        ("IDS_STEP5_BOOTSTRAP_RESAMPLES", "bootstrap_resamples"),
    ]:
        value = os.environ.get(env_name)
        if value is not None and str(value).strip():
            CFG[cfg_key] = int(value)
    backend = os.environ.get("IDS_STEP5_PARALLEL_BACKEND")
    if backend is not None and str(backend).strip():
        CFG["parallel_backend"] = str(backend).strip().lower()


def parse_json_dict(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    return dict(json.loads(text)) if text else {}


def normalize_case(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def sanitize_token(value: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[\\/:*?\"<>|\s]+", "_", str(value).strip())).strip("._") or "NA"


def case_run_dir(out_root: str, dataset: str, holdout: str, model_profile: object = "") -> str:
    profile = str(model_profile or "").strip()
    profile_token = f"__{sanitize_token(profile)}" if profile else ""
    return os.path.join(out_root, f"{dataset}__holdout_{sanitize_token(holdout)}{profile_token}__winner_replay")


def case_curve_cache_path(run_dir: str) -> str:
    return os.path.join(run_dir, "unknown_known_curves_case.csv")


def case_ci_cache_path(run_dir: str) -> str:
    return os.path.join(run_dir, "confidence_intervals_case.csv")


def case_paired_cache_path(run_dir: str) -> str:
    return os.path.join(run_dir, "paired_method_tests_case.csv")


def detect_xgb_gpu_available() -> bool:
    requested = str(CFG.get("xgb_device", "auto")).strip().lower()
    if requested == "cpu":
        return False
    if requested == "cuda":
        return True
    return shutil.which("nvidia-smi") is not None


def resolve_execution_settings(num_cases: int) -> Dict[str, object]:
    cpu_total = max(1, int(os.cpu_count() or 1))
    threads_cfg = CFG.get("threads_per_worker")
    workers_cfg = CFG.get("case_parallel_workers")

    if threads_cfg is None and workers_cfg is None:
        case_parallel_workers = max(1, min(num_cases or 1, min(16, cpu_total)))
        threads_per_worker = max(1, cpu_total // case_parallel_workers)
    else:
        if threads_cfg is None:
            case_parallel_workers = max(1, min(num_cases or 1, int(workers_cfg)))
            threads_per_worker = max(1, cpu_total // case_parallel_workers)
        elif workers_cfg is None:
            threads_per_worker = max(1, min(cpu_total, int(threads_cfg)))
            case_parallel_workers = max(1, min(num_cases or 1, max(1, cpu_total // threads_per_worker)))
        else:
            threads_per_worker = max(1, min(cpu_total, int(threads_cfg)))
            case_parallel_workers = max(1, min(num_cases or 1, int(workers_cfg)))

    return {
        "cpu_total": cpu_total,
        "threads_per_worker": int(threads_per_worker),
        "case_parallel_workers": int(case_parallel_workers),
        "parallel_backend": str(CFG.get("parallel_backend", "process")).strip().lower(),
        "use_xgb_gpu": bool(detect_xgb_gpu_available()),
    }


def apply_execution_overrides(grid_mod, combo: Dict[str, object], exec_cfg: Dict[str, object]) -> Dict[str, object]:
    combo_local = {
        "model_family": str(combo["model_family"]),
        "model_profile": str(combo.get("model_profile", combo["model_family"])),
        "apply_loao_stage1": bool(combo["apply_loao_stage1"]),
        "stage1_weight_mode": str(combo["stage1_weight_mode"]),
        "stage2_weight_mode": str(combo.get("stage2_weight_mode", "balanced")),
        "stage1_params": dict(combo["stage1_params"]),
        "stage2_params": dict(combo["stage2_params"]),
    }
    threads_per_worker = int(exec_cfg["threads_per_worker"])

    grid_mod.CFG["xgb_binary_defaults"] = dict(grid_mod.CFG["xgb_binary_defaults"])
    grid_mod.CFG["xgb_multi_defaults"] = dict(grid_mod.CFG["xgb_multi_defaults"])
    grid_mod.CFG["xgb_binary_defaults"]["n_jobs"] = threads_per_worker
    grid_mod.CFG["xgb_multi_defaults"]["n_jobs"] = threads_per_worker
    grid_mod.CFG["xgb_binary_defaults"].pop("predictor", None)
    grid_mod.CFG["xgb_multi_defaults"].pop("predictor", None)
    if bool(exec_cfg["use_xgb_gpu"]):
        grid_mod.CFG["xgb_binary_defaults"]["device"] = "cuda"
        grid_mod.CFG["xgb_multi_defaults"]["device"] = "cuda"
    else:
        grid_mod.CFG["xgb_binary_defaults"].pop("device", None)
        grid_mod.CFG["xgb_multi_defaults"].pop("device", None)

    combo_local["stage1_params"]["n_jobs"] = threads_per_worker
    combo_local["stage2_params"]["n_jobs"] = threads_per_worker
    return combo_local


def case_is_complete(run_dir: str) -> bool:
    needed = [
        "scenario_manifest.json",
        "method_grid.csv",
        "selected_methods.csv",
        "method_comparison_test.csv",
        "val_scores.csv.gz",
        "test_scores.csv.gz",
    ]
    return all(os.path.exists(os.path.join(run_dir, name)) for name in needed)


def case_has_selection_outputs(run_dir: str) -> bool:
    needed = [
        "scenario_manifest.json",
        "method_grid.csv",
        "selected_methods.csv",
        "val_scores.csv.gz",
        "test_scores.csv.gz",
    ]
    return all(os.path.exists(os.path.join(run_dir, name)) for name in needed)


def case_has_score_outputs(run_dir: str) -> bool:
    needed = [
        "scenario_manifest.json",
        "stage1_threshold_best.json",
        "val_scores.csv.gz",
        "test_scores.csv.gz",
    ]
    return all(os.path.exists(os.path.join(run_dir, name)) for name in needed)


def predict_binary_proba(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)


def predict_multi_proba(model, X) -> np.ndarray:
    p = np.asarray(model.predict_proba(X), dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def top2_margin(p2: np.ndarray) -> np.ndarray:
    if p2.shape[1] < 2:
        return np.ones(len(p2), dtype=np.float64)
    part = np.partition(p2, kth=-2, axis=1)
    return np.asarray(part[:, -1] - part[:, -2], dtype=np.float64)


def normalized_entropy(p2: np.ndarray) -> np.ndarray:
    if p2.shape[1] <= 1:
        return np.zeros(len(p2), dtype=np.float64)
    ent = -(p2 * np.log(np.clip(p2, 1e-12, 1.0))).sum(axis=1)
    return np.asarray(ent / np.log(p2.shape[1]), dtype=np.float64)


def load_manifest_index() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for audit_root in list(CFG["audit_roots"]):
        path = resolve_repo_path(os.path.join(str(audit_root), "eligible_holdouts_all.csv"))
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError("No eligible_holdouts_all.csv files were found.")
    out = pd.concat(frames, ignore_index=True)
    out["case_key"] = out["dataset"].astype(str) + "::" + out["holdout_family"].astype(str).map(normalize_case)
    return out


def load_winner_rows(index_df: pd.DataFrame) -> pd.DataFrame:
    winners = pd.read_csv(resolve_repo_path(str(CFG["best_csv"]))).copy()
    if "model_profile" not in winners.columns:
        winners["model_profile"] = winners.get("model_family", "")
    if "stage2_weight_mode" not in winners.columns:
        winners["stage2_weight_mode"] = "balanced"
    winners["case_key"] = winners["dataset"].astype(str) + "::" + winners["holdout_family"].astype(str).map(normalize_case)
    winners = winners.merge(index_df[["case_key", "manifest_path"]], on="case_key", how="left", validate="many_to_one")
    if winners["manifest_path"].isna().any():
        missing = winners.loc[winners["manifest_path"].isna(), ["dataset", "holdout_family"]]
        raise RuntimeError(f"Missing manifests for winner rows:\n{missing.to_string(index=False)}")
    selected = {str(x) for x in list(CFG["selected_cases"])}
    if selected:
        winners = winners.loc[winners.apply(lambda r: f"{r['dataset']}::{r['holdout_family']}" in selected, axis=1)].copy()
    return winners.sort_values(["dataset", "holdout_family", "model_profile"]).reset_index(drop=True)


def build_score_frame(grid_mod, prep, stage1_model, stage2_model, split: str, frame: pd.DataFrame, y1_col: str, y2_col: str, holdout_family: str, families: Sequence[str], benign_label: str, unknown_label: str) -> pd.DataFrame:
    X = prep.transform(frame.drop(columns=[y1_col, y2_col], errors="ignore"))
    p_attack = predict_binary_proba(stage1_model, X)
    p2 = predict_multi_proba(stage2_model, X)
    fam_pred_idx = np.argmax(p2, axis=1).astype(int)
    fam_pmax = np.max(p2, axis=1).astype(np.float64)
    y1 = frame[y1_col].astype(int).to_numpy()
    y2 = frame[y2_col].astype(str).fillna("").to_numpy(dtype=object)
    y_true_sys = grid_mod.system_labels_from_truth(y1, y2, holdout_family, list(families), benign_label, unknown_label)
    fam_to_idx = {str(fam): idx for idx, fam in enumerate(families)}
    true_known_prob = np.full(len(frame), np.nan, dtype=np.float64)
    known_mask = (y1 == 1) & np.isin(y2, list(families))
    if np.any(known_mask):
        known_idx = np.array([fam_to_idx[str(f)] for f in y2[known_mask]], dtype=int)
        true_known_prob[known_mask] = p2[known_mask, known_idx]
    return pd.DataFrame(
        {
            "row_id": np.arange(len(frame), dtype=int),
            "split": split,
            "y_stage1_attack": y1,
            "y_stage2_family": y2,
            "y_true_sys": y_true_sys,
            "is_true_unknown": (y_true_sys == unknown_label).astype(int),
            "p_attack": p_attack,
            "fam_pred_idx": fam_pred_idx,
            "fam_pred_family": np.array([families[idx] for idx in fam_pred_idx], dtype=object),
            "fam_pmax": fam_pmax,
            "top2_margin": top2_margin(p2),
            "stage2_entropy": normalized_entropy(p2),
            "true_known_family_prob": true_known_prob,
        }
    )


def conformal_family_thresholds(val_scores: pd.DataFrame, families: Sequence[str], alpha: float) -> Dict[str, float]:
    known = val_scores.loc[(val_scores["y_stage1_attack"] == 1) & val_scores["y_stage2_family"].isin(list(families))].copy()
    out: Dict[str, float] = {}
    for fam in families:
        vals = pd.to_numeric(known.loc[known["y_stage2_family"] == fam, "true_known_family_prob"], errors="coerce").dropna()
        out[str(fam)] = 0.0 if vals.empty else float(np.quantile(vals.to_numpy(dtype=np.float64), float(alpha)))
    return out


def predict_for_method(scores: pd.DataFrame, method: str, thr_high: float, families: Sequence[str], unknown_label: str, params: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_attack = pd.to_numeric(scores["p_attack"], errors="coerce").to_numpy(dtype=np.float64)
    fam_pred = scores["fam_pred_family"].astype(str).to_numpy(dtype=object)
    fam_pmax = pd.to_numeric(scores["fam_pmax"], errors="coerce").to_numpy(dtype=np.float64)
    margin = pd.to_numeric(scores["top2_margin"], errors="coerce").to_numpy(dtype=np.float64)
    entropy = pd.to_numeric(scores["stage2_entropy"], errors="coerce").to_numpy(dtype=np.float64)
    pred = np.empty(len(scores), dtype=object)
    pred[p_attack < thr_high] = "Benign"
    reject = np.zeros(len(scores), dtype=bool)
    uncertainty = np.zeros(len(scores), dtype=np.float64)
    attack_idx = np.where(p_attack >= thr_high)[0]
    if len(attack_idx) == 0:
        return pred, reject, uncertainty
    if method == "control_tau":
        uncertainty[attack_idx] = 1.0 - fam_pmax[attack_idx]
        reject_attack = fam_pmax[attack_idx] < float(params["tau"])
    elif method == "margin_reject":
        uncertainty[attack_idx] = 1.0 - margin[attack_idx]
        reject_attack = margin[attack_idx] < float(params["margin_thr"])
    elif method == "entropy_reject":
        uncertainty[attack_idx] = entropy[attack_idx]
        reject_attack = entropy[attack_idx] > float(params["entropy_thr"])
    elif method == "conformal_reject":
        family_thresholds = dict(params["family_thresholds"])
        threshold_vec = np.array([float(family_thresholds.get(str(fam), 0.0)) for fam in fam_pred[attack_idx]], dtype=np.float64)
        uncertainty[attack_idx] = threshold_vec - fam_pmax[attack_idx]
        reject_attack = fam_pmax[attack_idx] < threshold_vec
    else:
        raise ValueError(f"Unknown method: {method}")
    reject[attack_idx] = reject_attack
    pred[attack_idx] = fam_pred[attack_idx]
    pred[attack_idx[reject_attack]] = unknown_label
    return pred, reject, uncertainty


def evaluate_predictions(grid_mod, scores: pd.DataFrame, pred: np.ndarray, reject: np.ndarray, uncertainty_score: np.ndarray, families: Sequence[str], unknown_label: str) -> Dict[str, object]:
    y_true = scores["y_true_sys"].astype(str).to_numpy(dtype=object)
    metrics = grid_mod.compute_system_metrics(y_true, pred, reject, list(families), "Benign", unknown_label)
    target = scores["is_true_unknown"].astype(int).to_numpy(dtype=int)
    if np.unique(target).size >= 2:
        auroc = float(roc_auc_score(target, uncertainty_score))
        aupr = float(average_precision_score(target, uncertainty_score))
    else:
        auroc = float("nan")
        aupr = float("nan")
    return {**metrics, "coverage": float(1.0 - metrics["overall_reject_rate"]), "unknown_known_auroc": auroc, "unknown_known_aupr": aupr}


def candidate_rows_for_method(method: str, val_scores: pd.DataFrame, families: Sequence[str]) -> List[Dict[str, object]]:
    if method == "control_tau":
        vals = pd.to_numeric(val_scores["fam_pmax"], errors="coerce").dropna()
        grid = list(CFG["tau_grid"])
        if not vals.empty:
            grid.extend([float(np.quantile(vals.to_numpy(dtype=np.float64), q)) for q in [0.05, 0.10, 0.20]])
        return [{"tau": float(np.clip(t, 0.0, 0.999))} for t in sorted(set(grid))]
    if method == "margin_reject":
        vals = pd.to_numeric(val_scores["top2_margin"], errors="coerce").dropna()
        return [{"margin_thr": float(np.quantile(vals.to_numpy(dtype=np.float64), q))} for q in CFG["margin_quantiles"]] if not vals.empty else [{"margin_thr": 0.0}]
    if method == "entropy_reject":
        vals = pd.to_numeric(val_scores["stage2_entropy"], errors="coerce").dropna()
        return [{"entropy_thr": float(np.quantile(vals.to_numpy(dtype=np.float64), q))} for q in CFG["entropy_quantiles"]] if not vals.empty else [{"entropy_thr": 1.0}]
    if method == "conformal_reject":
        return [{"alpha": float(alpha), "family_thresholds": conformal_family_thresholds(val_scores, families, float(alpha))} for alpha in CFG["conformal_alpha_grid"]]
    raise ValueError(f"Unsupported method: {method}")


def select_best_candidate(method: str, grid_mod, val_scores: pd.DataFrame, thr_high: float, families: Sequence[str], unknown_label: str) -> Tuple[Dict[str, object], pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    for params in candidate_rows_for_method(method, val_scores, families):
        pred, reject, uncertainty = predict_for_method(val_scores, method, thr_high, families, unknown_label, params)
        metrics = evaluate_predictions(grid_mod, val_scores, pred, reject, uncertainty, families, unknown_label)
        row = {"method": method, "selection_param_json": json.dumps(params, sort_keys=True), **params, **metrics}
        ok = (
            float(metrics["benign_family_fp_rate"]) <= 0.02 + 1e-12
            and float(metrics["benign_reject_rate"]) <= 0.10 + 1e-12
            and float(metrics["overall_reject_rate"]) <= 0.10 + 1e-12
            and float(metrics["false_unknown_rate_all_known"]) <= 0.05 + 1e-12
            and float(metrics["false_unknown_rate_known_attacks"]) <= 0.10 + 1e-12
        )
        row["ok"] = bool(ok)
        rows.append(row)
        if not ok:
            continue
        if best is None:
            best = dict(row)
        elif float(row["unknown_detection_rate"]) > float(best["unknown_detection_rate"]):
            best = dict(row)
        elif float(row["unknown_detection_rate"]) == float(best["unknown_detection_rate"]):
            if float(row["macro_f1"]) > float(best["macro_f1"]):
                best = dict(row)
            elif float(row["macro_f1"]) == float(best["macro_f1"]) and float(row["false_unknown_rate_all_known"]) < float(best["false_unknown_rate_all_known"]):
                best = dict(row)
    df = pd.DataFrame(rows)
    if best is None:
        if df.empty:
            raise RuntimeError(f"No candidate rows generated for {method}.")
        best = df.sort_values(["unknown_detection_rate", "macro_f1"], ascending=[False, False]).iloc[0].to_dict()
    return best, df


def _metrics_from_state_counts(
    state_counts: np.ndarray,
    true_idx: np.ndarray,
    pred_idx: np.ndarray,
    reject_state: np.ndarray,
    n_labels: int,
    unknown_idx: int,
) -> Tuple[float, float, float, float]:
    confusion = np.bincount(
        true_idx * n_labels + pred_idx,
        weights=state_counts,
        minlength=n_labels * n_labels,
    ).reshape(n_labels, n_labels)
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    macro_f1 = float(np.mean(f1))

    unknown_total = float(confusion[unknown_idx, :].sum())
    unknown_detected = float(confusion[unknown_idx, unknown_idx])
    known_total = float(confusion.sum() - unknown_total)
    known_pred_unknown = float(confusion[:, unknown_idx].sum() - unknown_detected)
    total = float(state_counts.sum())
    reject_rate = float(np.dot(state_counts, reject_state) / total) if total else 0.0
    return (
        macro_f1,
        unknown_detected / unknown_total if unknown_total else 0.0,
        known_pred_unknown / known_total if known_total else 0.0,
        reject_rate,
    )


def _single_method_states(
    y_true: np.ndarray,
    pred: np.ndarray,
    reject: np.ndarray,
    families: Sequence[str],
    unknown_label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    labels = ["Benign"] + list(families) + [unknown_label]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    true_idx_all = np.array([label_to_idx[str(value)] for value in y_true], dtype=np.int16)
    pred_idx_all = np.array([label_to_idx[str(value)] for value in pred], dtype=np.int16)
    reject_all = np.asarray(reject, dtype=np.int8)
    state_code = (true_idx_all * len(labels) + pred_idx_all) * 2 + reject_all
    unique_code, counts = np.unique(state_code, return_counts=True)
    reject_state = unique_code % 2
    pair_code = unique_code // 2
    pred_idx = pair_code % len(labels)
    true_idx = pair_code // len(labels)
    return counts.astype(np.int64), true_idx, pred_idx, reject_state, len(labels), label_to_idx[unknown_label]


def bootstrap_ci(grid_mod, y_true: np.ndarray, pred: np.ndarray, reject: np.ndarray, families: Sequence[str], unknown_label: str, resamples: int, seed: int) -> Dict[str, float]:
    del grid_mod
    rng = np.random.default_rng(seed)
    counts, true_idx, pred_idx, reject_state, n_labels, unknown_idx = _single_method_states(
        y_true, pred, reject, families, unknown_label
    )
    probabilities = counts / counts.sum()
    n_rows = int(counts.sum())
    macro_vals: List[float] = []
    udr_vals: List[float] = []
    false_unknown_vals: List[float] = []
    reject_rate_vals: List[float] = []
    for _ in range(int(resamples)):
        sampled_counts = rng.multinomial(n_rows, probabilities)
        macro_f1, udr, false_unknown, reject_rate = _metrics_from_state_counts(
            sampled_counts, true_idx, pred_idx, reject_state, n_labels, unknown_idx
        )
        macro_vals.append(macro_f1)
        udr_vals.append(udr)
        false_unknown_vals.append(false_unknown)
        reject_rate_vals.append(reject_rate)
    return {
        "macro_f1_ci_low": float(np.quantile(macro_vals, 0.025)),
        "macro_f1_ci_high": float(np.quantile(macro_vals, 0.975)),
        "unknown_detection_ci_low": float(np.quantile(udr_vals, 0.025)),
        "unknown_detection_ci_high": float(np.quantile(udr_vals, 0.975)),
        "false_unknown_ci_low": float(np.quantile(false_unknown_vals, 0.025)),
        "false_unknown_ci_high": float(np.quantile(false_unknown_vals, 0.975)),
        "reject_rate_ci_low": float(np.quantile(reject_rate_vals, 0.025)),
        "reject_rate_ci_high": float(np.quantile(reject_rate_vals, 0.975)),
    }


def paired_bootstrap_diff(grid_mod, y_true: np.ndarray, pred_a: np.ndarray, reject_a: np.ndarray, pred_b: np.ndarray, reject_b: np.ndarray, families: Sequence[str], unknown_label: str, resamples: int, seed: int) -> Dict[str, float]:
    del grid_mod
    rng = np.random.default_rng(seed)
    labels = ["Benign"] + list(families) + [unknown_label]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n_labels = len(labels)
    unknown_idx = label_to_idx[unknown_label]
    true_idx_all = np.array([label_to_idx[str(value)] for value in y_true], dtype=np.int16)
    pred_a_idx_all = np.array([label_to_idx[str(value)] for value in pred_a], dtype=np.int16)
    pred_b_idx_all = np.array([label_to_idx[str(value)] for value in pred_b], dtype=np.int16)
    reject_a_all = np.asarray(reject_a, dtype=np.int8)
    reject_b_all = np.asarray(reject_b, dtype=np.int8)
    state_code = (
        ((((true_idx_all * n_labels) + pred_a_idx_all) * 2 + reject_a_all) * n_labels + pred_b_idx_all) * 2
        + reject_b_all
    )
    unique_code, counts = np.unique(state_code, return_counts=True)
    code = unique_code.copy()
    reject_b_state = code % 2
    code //= 2
    pred_b_idx = code % n_labels
    code //= n_labels
    reject_a_state = code % 2
    code //= 2
    pred_a_idx = code % n_labels
    true_idx = code // n_labels
    probabilities = counts / counts.sum()
    n_rows = int(counts.sum())
    diff_macro: List[float] = []
    diff_udr: List[float] = []
    for _ in range(int(resamples)):
        sampled_counts = rng.multinomial(n_rows, probabilities)
        macro_a, udr_a, _, _ = _metrics_from_state_counts(
            sampled_counts, true_idx, pred_a_idx, reject_a_state, n_labels, unknown_idx
        )
        macro_b, udr_b, _, _ = _metrics_from_state_counts(
            sampled_counts, true_idx, pred_b_idx, reject_b_state, n_labels, unknown_idx
        )
        diff_macro.append(macro_a - macro_b)
        diff_udr.append(udr_a - udr_b)
    p_macro = 2.0 * min(float(np.mean(np.array(diff_macro) <= 0)), float(np.mean(np.array(diff_macro) >= 0)))
    p_udr = 2.0 * min(float(np.mean(np.array(diff_udr) <= 0)), float(np.mean(np.array(diff_udr) >= 0)))
    return {
        "delta_macro_f1_mean": float(np.mean(diff_macro)),
        "delta_macro_f1_ci_low": float(np.quantile(diff_macro, 0.025)),
        "delta_macro_f1_ci_high": float(np.quantile(diff_macro, 0.975)),
        "delta_macro_f1_pvalue_bootstrap": float(min(1.0, p_macro)),
        "delta_unknown_detection_mean": float(np.mean(diff_udr)),
        "delta_unknown_detection_ci_low": float(np.quantile(diff_udr, 0.025)),
        "delta_unknown_detection_ci_high": float(np.quantile(diff_udr, 0.975)),
        "delta_unknown_detection_pvalue_bootstrap": float(min(1.0, p_udr)),
    }


def build_case_outputs_from_scores(
    case_idx: int,
    run_dir: str,
    dataset: str,
    holdout: str,
    combo: Dict[str, object],
    grid_mod,
    families: Sequence[str],
    unknown_label: str,
    thr_high: float,
    val_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
) -> Dict[str, object]:
    grid_frames: List[pd.DataFrame] = []
    selected_frames: List[pd.DataFrame] = []
    selected_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    selected_metrics: Dict[str, Dict[str, object]] = {}
    curve_rows: List[Dict[str, object]] = []
    ci_rows: List[Dict[str, object]] = []
    paired_rows: List[Dict[str, object]] = []

    for method in ["control_tau", "margin_reject", "entropy_reject", "conformal_reject"]:
        best_row, grid_df = select_best_candidate(method, grid_mod, val_scores, float(thr_high), families, unknown_label)
        grid_df.insert(0, "dataset", dataset)
        grid_df.insert(1, "holdout_family", holdout)
        grid_df.insert(2, "split_variant", split_variant_name_for_dataset(dataset))
        grid_df.insert(3, "model_family", combo["model_family"])
        grid_df.insert(4, "model_profile", combo.get("model_profile", combo["model_family"]))
        grid_df.insert(5, "stage1_weight_mode", combo["stage1_weight_mode"])
        grid_df.insert(6, "stage2_weight_mode", combo.get("stage2_weight_mode", "balanced"))
        grid_df.insert(7, "thr_high", float(thr_high))
        grid_frames.append(grid_df)
        params = parse_json_dict(best_row["selection_param_json"])
        for split_name, scores_df in [("val", val_scores), ("test", test_scores)]:
            pred, reject, uncertainty = predict_for_method(scores_df, method, float(thr_high), families, unknown_label, params)
            metrics = evaluate_predictions(grid_mod, scores_df, pred, reject, uncertainty, families, unknown_label)
            selected_frames.append(pd.DataFrame([{
                "dataset": dataset,
                "holdout_family": holdout,
                "split_variant": split_variant_name_for_dataset(dataset),
                "model_family": combo["model_family"],
                "model_profile": combo.get("model_profile", combo["model_family"]),
                "stage1_weight_mode": combo["stage1_weight_mode"],
                "stage2_weight_mode": combo.get("stage2_weight_mode", "balanced"),
                "split": split_name,
                "method": method,
                "thr_high": float(thr_high),
                "selection_param_json": json.dumps(params, sort_keys=True),
                **metrics,
            }]))
            if split_name == "test":
                selected_predictions[method] = (pred.copy(), reject.copy())
                selected_metrics[method] = metrics
        curve_rows.extend(grid_df.assign(curve_source="validation_search").to_dict("records"))
        for row in grid_df.to_dict("records"):
            params_eval = parse_json_dict(row["selection_param_json"])
            pred, reject, uncertainty = predict_for_method(test_scores, method, float(thr_high), families, unknown_label, params_eval)
            metrics = evaluate_predictions(grid_mod, test_scores, pred, reject, uncertainty, families, unknown_label)
            curve_rows.append({
                "dataset": dataset,
                "holdout_family": holdout,
                "split_variant": split_variant_name_for_dataset(dataset),
                "model_family": combo["model_family"],
                "model_profile": combo.get("model_profile", combo["model_family"]),
                "stage1_weight_mode": combo["stage1_weight_mode"],
                "stage2_weight_mode": combo.get("stage2_weight_mode", "balanced"),
                "method": method,
                "curve_source": "test_curve",
                "thr_high": float(thr_high),
                "selection_param_json": row["selection_param_json"],
                **metrics,
            })

    method_grid_all = pd.concat(grid_frames, ignore_index=True)
    selected_all = pd.concat(selected_frames, ignore_index=True)
    method_grid_all.to_csv(os.path.join(run_dir, "method_grid.csv"), index=False)
    selected_all.to_csv(os.path.join(run_dir, "selected_methods.csv"), index=False)

    test_rows = selected_all.loc[selected_all["split"] == "test"].copy()
    control_row = test_rows.loc[test_rows["method"] == "control_tau"].iloc[0]
    comparison_rows: List[Dict[str, object]] = []
    y_true_test = test_scores["y_true_sys"].astype(str).to_numpy(dtype=object)
    for method, (pred, reject) in selected_predictions.items():
        ci_rows.append({
            "dataset": dataset,
            "holdout_family": holdout,
            "split_variant": split_variant_name_for_dataset(dataset),
            "model_family": combo["model_family"],
            "model_profile": combo.get("model_profile", combo["model_family"]),
            "stage1_weight_mode": combo["stage1_weight_mode"],
            "stage2_weight_mode": combo.get("stage2_weight_mode", "balanced"),
            "method": method,
            **selected_metrics[method],
            **bootstrap_ci(grid_mod, y_true_test, pred, reject, families, unknown_label, int(CFG["bootstrap_resamples"]), int(CFG["random_seed"]) + case_idx),
        })
        comparison_rows.append({
            "dataset": dataset,
            "holdout_family": holdout,
            "split_variant": split_variant_name_for_dataset(dataset),
            "model_family": combo["model_family"],
            "stage1_weight_mode": combo["stage1_weight_mode"],
            "method": method,
            **selected_metrics[method],
            "delta_macro_f1_vs_control": float(selected_metrics[method]["macro_f1"]) - float(control_row["macro_f1"]),
            "delta_unknown_detection_vs_control": float(selected_metrics[method]["unknown_detection_rate"]) - float(control_row["unknown_detection_rate"]),
        })
    pd.DataFrame(comparison_rows).to_csv(os.path.join(run_dir, "method_comparison_test.csv"), index=False)
    control_pred, control_reject = selected_predictions["control_tau"]
    for method, (pred, reject) in selected_predictions.items():
        if method == "control_tau":
            continue
        paired_rows.append({
            "dataset": dataset,
            "holdout_family": holdout,
            "split_variant": split_variant_name_for_dataset(dataset),
            "model_family": combo["model_family"],
            "model_profile": combo.get("model_profile", combo["model_family"]),
            "stage1_weight_mode": combo["stage1_weight_mode"],
            "stage2_weight_mode": combo.get("stage2_weight_mode", "balanced"),
            "method_a": method,
            "method_b": "control_tau",
            **paired_bootstrap_diff(grid_mod, y_true_test, pred, reject, control_pred, control_reject, families, unknown_label, int(CFG["bootstrap_resamples"]), int(CFG["random_seed"]) + 1000 + case_idx),
        })

    pd.DataFrame(curve_rows).to_csv(case_curve_cache_path(run_dir), index=False)
    pd.DataFrame(ci_rows).to_csv(case_ci_cache_path(run_dir), index=False)
    pd.DataFrame(paired_rows).to_csv(case_paired_cache_path(run_dir), index=False)

    return {
        "case_idx": int(case_idx),
        "aggregate_rows": selected_all.to_dict("records"),
        "curve_rows": curve_rows,
        "ci_rows": ci_rows,
        "paired_rows": paired_rows,
    }


def run_case(case_idx: int, rec: Dict[str, object], exec_cfg: Dict[str, object]) -> Dict[str, object]:
    dataset = str(rec["dataset"])
    holdout = str(rec["holdout_family"])
    out_root = resolve_repo_path(str(CFG["runs_root"]))
    run_dir = case_run_dir(out_root, dataset, holdout, rec.get("model_profile", ""))
    safe_mkdir(run_dir)
    run_name = os.path.basename(run_dir)

    grid_mod = load_module(str(CFG["grid_script"]), f"ids_open_set_grid_{sanitize_token(dataset)}_{sanitize_token(holdout)}_{case_idx}")
    helper = load_cached_helper_module(grid_mod, resolve_repo_path(str(CFG["base_helper_script"])))

    manifest = grid_mod.load_json(str(rec["manifest_path"]))
    grid_mod.write_json(os.path.join(run_dir, "scenario_manifest.json"), manifest)
    grid_mod.write_json(os.path.join(run_dir, "winner_reference.json"), rec)

    combo = {
        "model_family": str(rec["model_family"]),
        "model_profile": str(rec.get("model_profile", rec["model_family"])),
        "apply_loao_stage1": bool(rec["apply_loao_stage1"]),
        "stage1_weight_mode": str(rec["stage1_weight_mode"]),
        "stage2_weight_mode": str(rec.get("stage2_weight_mode", "balanced")),
        "stage1_params": parse_json_dict(rec["stage1_params"]),
        "stage2_params": parse_json_dict(rec["stage2_params"]),
    }
    combo = apply_execution_overrides(grid_mod, combo, exec_cfg)
    grid_mod.write_json(os.path.join(run_dir, "combo.json"), combo)
    grid_mod.write_json(
        os.path.join(run_dir, "execution_settings.json"),
        {
            "case_idx": int(case_idx),
            "threads_per_worker": int(exec_cfg["threads_per_worker"]),
            "case_parallel_workers": int(exec_cfg["case_parallel_workers"]),
            "use_xgb_gpu": bool(exec_cfg["use_xgb_gpu"]),
        },
    )

    dataset_dir = str(manifest["processed_dir"])
    y1_col = grid_mod.canonical_col(str(manifest["y_stage1_col"]))
    y2_col = grid_mod.canonical_col(str(manifest["y_stage2_col"]))
    benign_label = str(manifest["benign_label"])
    unknown_label = str(manifest.get("unknown_label", "Unknown"))
    manifest_families = list(manifest["valid_known_families"])
    families = list(manifest_families)
    seed = int(CFG["random_seed"]) + int(case_idx)

    stage1_best_path = os.path.join(run_dir, "stage1_threshold_best.json")
    val_scores_path = os.path.join(run_dir, "val_scores.csv.gz")
    test_scores_path = os.path.join(run_dir, "test_scores.csv.gz")
    if case_has_score_outputs(run_dir):
        safe_print(f"[open-set {case_idx}] resume scores -> {run_name}")
        thr_meta = grid_mod.load_json(stage1_best_path)
        thr_high = float(thr_meta.get("thr_high", thr_meta.get("thr")))
        val_scores = pd.read_csv(val_scores_path)
        test_scores = pd.read_csv(test_scores_path)
        return build_case_outputs_from_scores(
            case_idx,
            run_dir,
            dataset,
            holdout,
            combo,
            grid_mod,
            families,
            unknown_label,
            thr_high,
            val_scores,
            test_scores,
        )

    preprocessor_path = os.path.join(run_dir, "preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        try:
            safe_print(f"[open-set {case_idx}] resume preprocessor -> {run_name}")
            prep = helper.safe_joblib_load(preprocessor_path)
        except Exception as exc:
            safe_print(f"[open-set {case_idx}] refit preprocessor after load error ({type(exc).__name__}) -> {run_name}")
            prep = helper.fit_preprocessor(dataset_dir, run_dir)
    else:
        safe_print(f"[open-set {case_idx}] fit preprocessor -> {run_name}")
        prep = helper.fit_preprocessor(dataset_dir, run_dir)
    usecols = prep.num_cols + prep.cat_cols + [y1_col, y2_col]
    safe_print(f"[open-set {case_idx}] collect splits -> {run_name}")
    train_df = grid_mod.collect_split_frame(dataset_dir, "train", usecols, dict(CFG["max_train_rows"]).get(dataset, None), seed=seed)
    val_df = grid_mod.collect_split_frame(dataset_dir, "val", usecols, dict(CFG["max_val_rows"]).get(dataset, None), seed=seed + 1)
    test_df = grid_mod.collect_split_frame(dataset_dir, "test", usecols, dict(CFG["max_test_rows"]).get(dataset, None), seed=seed + 2)

    safe_print(f"[open-set {case_idx}] train stage 1 -> {run_name}")
    s1_train_df = train_df.copy()
    if combo["apply_loao_stage1"]:
        drop_mask = (s1_train_df[y1_col].astype(int) == 1) & (s1_train_df[y2_col].astype(str) == holdout)
        s1_train_df = s1_train_df.loc[~drop_mask].reset_index(drop=True)
    X1_train = prep.transform(s1_train_df.drop(columns=[y1_col, y2_col], errors="ignore"))
    y1_train = s1_train_df[y1_col].astype(int).to_numpy()
    y2_train = s1_train_df[y2_col].astype(str).fillna("").to_numpy(dtype=object)
    stage1_model = grid_mod.build_stage1_model(combo["model_family"], combo["stage1_params"], combo["stage1_weight_mode"], seed=seed)
    s1_weight = grid_mod.make_stage1_sample_weights(y1_train, y2_train, combo["stage1_weight_mode"], float(grid_mod.CFG["inv_family_clip_max"]))
    stage1_model.fit(X1_train, y1_train, sample_weight=s1_weight) if s1_weight is not None else stage1_model.fit(X1_train, y1_train)

    X_val = prep.transform(val_df.drop(columns=[y1_col, y2_col], errors="ignore"))
    y1_val = val_df[y1_col].astype(int).to_numpy()
    y2_val = val_df[y2_col].astype(str).fillna("").to_numpy(dtype=object)
    p_attack_val = predict_binary_proba(stage1_model, X_val)
    thr_high, thr_meta, thr_df = grid_mod.select_stage1_threshold(helper, y1_val, y2_val, p_attack_val)
    thr_df.to_csv(os.path.join(run_dir, "stage1_threshold_grid.csv"), index=False)
    grid_mod.write_json(stage1_best_path, thr_meta)

    attack_train_df = train_df.loc[(train_df[y1_col].astype(int) == 1) & train_df[y2_col].astype(str).isin(families)].copy().reset_index(drop=True)
    if attack_train_df.empty:
        raise RuntimeError("Stage-2 training set is empty after applying valid_known_families.")
    train_family_set = set(attack_train_df[y2_col].astype(str).tolist())
    trained_families = [fam for fam in families if fam in train_family_set]
    missing_train_families = [fam for fam in families if fam not in train_family_set]
    if len(trained_families) < 2:
        raise RuntimeError(
            "Stage-2 training set has fewer than two known families after row capping: "
            f"{trained_families}. Increase max_train_rows or choose another case."
        )
    if missing_train_families:
        print(
            "[WARN] Stage-2 row cap dropped known families; evaluating with trained subset: "
            + ", ".join(missing_train_families)
        )
        attack_train_df = attack_train_df.loc[attack_train_df[y2_col].astype(str).isin(trained_families)].reset_index(drop=True)
    families = trained_families
    fam_to_idx = {fam: i for i, fam in enumerate(families)}
    y2_train_idx = np.array([fam_to_idx[f] for f in attack_train_df[y2_col].astype(str).tolist()], dtype=int)
    X2_train = prep.transform(attack_train_df.drop(columns=[y1_col, y2_col], errors="ignore"))
    stage2_model = grid_mod.build_stage2_model(combo["model_family"], combo["stage2_params"], len(families), seed=seed + 10)
    s2_weight = grid_mod.stage2_sample_weights(y2_train_idx, len(families), combo.get("stage2_weight_mode", "balanced"))
    safe_print(f"[open-set {case_idx}] train stage 2 -> {run_name}")
    stage2_model.fit(X2_train, y2_train_idx, sample_weight=s2_weight) if s2_weight is not None else stage2_model.fit(X2_train, y2_train_idx)

    safe_print(f"[open-set {case_idx}] score splits -> {run_name}")
    val_scores = build_score_frame(grid_mod, prep, stage1_model, stage2_model, "val", val_df, y1_col, y2_col, holdout, families, benign_label, unknown_label)
    test_scores = build_score_frame(grid_mod, prep, stage1_model, stage2_model, "test", test_df, y1_col, y2_col, holdout, families, benign_label, unknown_label)
    val_scores.to_csv(val_scores_path, index=False, compression="gzip")
    test_scores.to_csv(test_scores_path, index=False, compression="gzip")

    safe_print(f"[open-set {case_idx}] select/bootstrap -> {run_name}")
    return build_case_outputs_from_scores(
        case_idx,
        run_dir,
        dataset,
        holdout,
        combo,
        grid_mod,
        families,
        unknown_label,
        float(thr_high),
        val_scores,
        test_scores,
    )


def load_completed_case(case_idx: int, rec: Dict[str, object]) -> Dict[str, object]:
    out_root = resolve_repo_path(str(CFG["runs_root"]))
    dataset = str(rec["dataset"])
    holdout = str(rec["holdout_family"])
    run_dir = case_run_dir(out_root, dataset, holdout, rec.get("model_profile", ""))
    grid_mod = load_module(str(CFG["grid_script"]), f"ids_open_set_grid_resume_{sanitize_token(dataset)}_{sanitize_token(holdout)}_{case_idx}")
    manifest = json.load(open(os.path.join(run_dir, "scenario_manifest.json"), "r", encoding="utf-8"))
    families = list(manifest["valid_known_families"])
    unknown_label = str(manifest.get("unknown_label", "Unknown"))

    method_grid = pd.read_csv(os.path.join(run_dir, "method_grid.csv"))
    selected_all = pd.read_csv(os.path.join(run_dir, "selected_methods.csv"))
    curve_cache = case_curve_cache_path(run_dir)
    ci_cache = case_ci_cache_path(run_dir)
    paired_cache = case_paired_cache_path(run_dir)
    comparison_cache = os.path.join(run_dir, "method_comparison_test.csv")
    if os.path.exists(curve_cache) and os.path.exists(ci_cache) and os.path.exists(paired_cache):
        return {
            "case_idx": int(case_idx),
            "aggregate_rows": selected_all.to_dict("records"),
            "curve_rows": pd.read_csv(curve_cache).to_dict("records"),
            "ci_rows": pd.read_csv(ci_cache).to_dict("records"),
            "paired_rows": pd.read_csv(paired_cache).to_dict("records"),
        }

    test_scores = pd.read_csv(os.path.join(run_dir, "test_scores.csv.gz"))

    selected_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    selected_metrics: Dict[str, Dict[str, object]] = {}
    curve_rows = method_grid.assign(curve_source="validation_search").to_dict("records")
    ci_rows: List[Dict[str, object]] = []
    paired_rows: List[Dict[str, object]] = []

    for row in method_grid.to_dict("records"):
        params_eval = parse_json_dict(row["selection_param_json"])
        pred, reject, uncertainty = predict_for_method(
            test_scores,
            str(row["method"]),
            float(row["thr_high"]),
            families,
            unknown_label,
            params_eval,
        )
        metrics = evaluate_predictions(
            grid_mod,
            test_scores,
            pred,
            reject,
            uncertainty,
            families,
            unknown_label,
        )
        curve_rows.append({
            "dataset": dataset,
            "holdout_family": holdout,
            "split_variant": split_variant_name_for_dataset(dataset),
            "model_family": str(row["model_family"]),
            "stage1_weight_mode": str(row["stage1_weight_mode"]),
            "method": str(row["method"]),
            "curve_source": "test_curve",
            "thr_high": float(row["thr_high"]),
            "selection_param_json": row["selection_param_json"],
            **metrics,
        })

    test_rows = selected_all.loc[selected_all["split"] == "test"].copy()
    y_true_test = test_scores["y_true_sys"].astype(str).to_numpy(dtype=object)
    for row in test_rows.to_dict("records"):
        method = str(row["method"])
        params = parse_json_dict(row["selection_param_json"])
        pred, reject, uncertainty = predict_for_method(test_scores, method, float(row["thr_high"]), families, unknown_label, params)
        metrics = evaluate_predictions(
            grid_mod,
            test_scores,
            pred,
            reject,
            uncertainty,
            families,
            unknown_label,
        )
        selected_predictions[method] = (pred, reject)
        selected_metrics[method] = metrics
        ci_rows.append({
            "dataset": dataset,
            "holdout_family": holdout,
            "split_variant": split_variant_name_for_dataset(dataset),
            "model_family": str(row["model_family"]),
            "stage1_weight_mode": str(row["stage1_weight_mode"]),
            "method": method,
            **metrics,
            **bootstrap_ci(
                grid_mod,
                y_true_test,
                pred,
                reject,
                families,
                unknown_label,
                int(CFG["bootstrap_resamples"]),
                int(CFG["random_seed"]) + case_idx,
            ),
        })

    control_pred, control_reject = selected_predictions["control_tau"]
    control_row = test_rows.loc[test_rows["method"] == "control_tau"].iloc[0]
    comparison_rows: List[Dict[str, object]] = []
    for method, (pred, reject) in selected_predictions.items():
        comparison_rows.append({
            "dataset": dataset,
            "holdout_family": holdout,
            "split_variant": split_variant_name_for_dataset(dataset),
            "model_family": str(control_row["model_family"]),
            "stage1_weight_mode": str(control_row["stage1_weight_mode"]),
            "method": method,
            **selected_metrics[method],
            "delta_macro_f1_vs_control": float(selected_metrics[method]["macro_f1"]) - float(control_row["macro_f1"]),
            "delta_unknown_detection_vs_control": float(selected_metrics[method]["unknown_detection_rate"]) - float(control_row["unknown_detection_rate"]),
        })
        if method == "control_tau":
            continue
        paired_rows.append({
            "dataset": dataset,
            "holdout_family": holdout,
            "split_variant": split_variant_name_for_dataset(dataset),
            "model_family": str(control_row["model_family"]),
            "stage1_weight_mode": str(control_row["stage1_weight_mode"]),
            "method_a": method,
            "method_b": "control_tau",
            **paired_bootstrap_diff(
                grid_mod,
                y_true_test,
                pred,
                reject,
                control_pred,
                control_reject,
                families,
                unknown_label,
                int(CFG["bootstrap_resamples"]),
                int(CFG["random_seed"]) + 1000 + case_idx,
            ),
        })
    pd.DataFrame(comparison_rows).to_csv(comparison_cache, index=False)
    pd.DataFrame(curve_rows).to_csv(curve_cache, index=False)
    pd.DataFrame(ci_rows).to_csv(ci_cache, index=False)
    pd.DataFrame(paired_rows).to_csv(paired_cache, index=False)

    return {
        "case_idx": int(case_idx),
        "aggregate_rows": selected_all.to_dict("records"),
        "curve_rows": curve_rows,
        "ci_rows": ci_rows,
        "paired_rows": paired_rows,
    }


def process_case(case_idx: int, rec: Dict[str, object], exec_cfg: Dict[str, object]) -> Dict[str, object]:
    out_root = resolve_repo_path(str(CFG["runs_root"]))
    run_dir = case_run_dir(out_root, str(rec["dataset"]), str(rec["holdout_family"]), rec.get("model_profile", ""))
    run_name = os.path.basename(run_dir)
    error_path = os.path.join(run_dir, "error.json")
    try:
        if os.path.exists(error_path):
            os.remove(error_path)
        if case_is_complete(run_dir):
            safe_print(f"[open-set {case_idx}] skip completed -> {run_name}")
            return load_completed_case(case_idx, rec)
        if case_has_selection_outputs(run_dir):
            safe_print(f"[open-set {case_idx}] resume selections -> {run_name}")
            return load_completed_case(case_idx, rec)
        safe_print(f"[open-set {case_idx}] run -> {run_name}")
        result = run_case(case_idx, rec, exec_cfg)
        safe_print(f"[open-set {case_idx}] done -> {run_name}")
        return result
    except Exception as exc:
        safe_mkdir(run_dir)
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "case_idx": int(case_idx),
                    "dataset": str(rec["dataset"]),
                    "holdout_family": str(rec["holdout_family"]),
                    "model_profile": str(rec.get("model_profile", "")),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                f,
                indent=2,
            )
        safe_print(f"[open-set {case_idx}] ERROR -> {run_name}: {type(exc).__name__}: {exc}")
        raise


def main() -> None:
    apply_env_overrides()
    out_root = resolve_repo_path(str(CFG["runs_root"]))
    summary_root = os.path.join(out_root, str(CFG["summary_dirname"]))
    safe_mkdir(out_root)
    safe_mkdir(summary_root)

    winners = load_winner_rows(load_manifest_index())
    winners.to_csv(os.path.join(out_root, "scenario_plan.csv"), index=False)
    exec_cfg = resolve_execution_settings(len(winners))
    with open(os.path.join(out_root, "execution_settings.json"), "w", encoding="utf-8") as f:
        json.dump(exec_cfg, f, indent=2)

    aggregate_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    ci_rows: List[Dict[str, object]] = []
    paired_rows: List[Dict[str, object]] = []

    payloads = [(case_idx, rec, exec_cfg) for case_idx, rec in enumerate(winners.to_dict("records"), start=1)]
    results: List[Dict[str, object]] = []
    if int(exec_cfg["case_parallel_workers"]) <= 1:
        for case_idx, rec, exec_local in payloads:
            results.append(process_case(case_idx, rec, exec_local))
    elif str(exec_cfg.get("parallel_backend", "process")).strip().lower() == "thread":
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(exec_cfg["case_parallel_workers"])) as pool:
            future_map = {
                pool.submit(process_case, case_idx, rec, exec_local): (case_idx, rec)
                for case_idx, rec, exec_local in payloads
            }
            for future in concurrent.futures.as_completed(future_map):
                results.append(future.result())
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(exec_cfg["case_parallel_workers"]),
            mp_context=mp_ctx,
        ) as pool:
            future_map = {
                pool.submit(process_case, case_idx, rec, exec_local): (case_idx, rec)
                for case_idx, rec, exec_local in payloads
            }
            for future in concurrent.futures.as_completed(future_map):
                results.append(future.result())

    results.sort(key=lambda x: int(x["case_idx"]))
    for result in results:
        aggregate_rows.extend(result["aggregate_rows"])
        curve_rows.extend(result["curve_rows"])
        ci_rows.extend(result["ci_rows"])
        paired_rows.extend(result["paired_rows"])

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(os.path.join(out_root, "aggregate_results.csv"), index=False)
    holdout_metrics = aggregate_df.loc[aggregate_df["split"] == "test"].copy()
    holdout_metrics.to_csv(os.path.join(summary_root, "open_set_holdout_metrics.csv"), index=False)
    comparison = holdout_metrics.copy()
    control_keys = [
        c
        for c in [
            "dataset",
            "holdout_family",
            "model_profile",
            "stage1_weight_mode",
            "stage2_weight_mode",
        ]
        if c in comparison.columns
    ]
    control_macro = comparison.loc[
        comparison["method"] == "control_tau",
        control_keys + ["macro_f1", "unknown_detection_rate"],
    ].rename(columns={"macro_f1": "control_macro_f1", "unknown_detection_rate": "control_unknown_detection_rate"})
    comparison = comparison.merge(control_macro, on=control_keys, how="left")
    comparison["delta_macro_f1_vs_control"] = pd.to_numeric(comparison["macro_f1"], errors="coerce") - pd.to_numeric(comparison["control_macro_f1"], errors="coerce")
    comparison["delta_unknown_detection_vs_control"] = pd.to_numeric(comparison["unknown_detection_rate"], errors="coerce") - pd.to_numeric(comparison["control_unknown_detection_rate"], errors="coerce")
    comparison.to_csv(os.path.join(summary_root, "open_set_baseline_comparison.csv"), index=False)
    pd.DataFrame(curve_rows).to_csv(os.path.join(summary_root, "open_set_unknown_known_curves.csv"), index=False)
    pd.DataFrame(ci_rows).to_csv(os.path.join(summary_root, "protocol_b_holdout_confidence_intervals.csv"), index=False)
    pd.DataFrame(paired_rows).to_csv(os.path.join(summary_root, "protocol_b_paired_method_tests.csv"), index=False)


if __name__ == "__main__":
    main()
