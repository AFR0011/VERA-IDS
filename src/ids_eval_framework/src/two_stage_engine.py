"""Merged two-stage IDS training/evaluation engine.

This module is derived from the legacy `3.TrainEvalBase_V4.1B.py` helper and
folds in the Protocol A CSV.GZ part handling from `3.TrainEvalBase_V4.1A.py`.
Protocol-specific behavior is selected through `configure_for_protocol()` and
`run_protocol()` instead of keeping separate A/B engine files active.
"""

import os
import re
import json
import time
import glob
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd

from scipy.special import softmax
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack, vstack

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

import warnings
warnings.filterwarnings("once", category=UserWarning)

try:
    import joblib
except Exception:
    joblib = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from ids_eval_framework.src.paths import deep_update, repo_path, resolve_repo_path  # noqa: E402


# ============================================================
# CONFIG
# ============================================================

CFG = {
    "processed_root": "processed_V5",
    "protocol": "B_day_file",
    "datasets": ["CICIDS2017", "CICIoT2023"],
# Protocol B-1 (LOAO) — leave-one-attack-family-out for stage-2 (open-set / zero-day)
# When enabled:
#   - Stage-2 training excludes the holdout family
#   - System-level ground truth maps (attack family not in trained families) -> Unknown
#   - Additional open-set metrics are exported (unknown detection rate, false unknown rates)
"loao": {
    "enabled": True,
    "holdout_family": {
        "CICIDS2017": "BruteForce",
        "CICIoT2023": "BruteForce",
    },
    # If True, also remove holdout-family attack rows from Stage-1 training (stricter, usually worse).
    "apply_to_stage1": False,

    # Optional: tune tau primarily for unknown detection (instead of macro-F1), under false-unknown constraints.
    "optimize_tau_for_unknown": True,
    "max_false_unknown_rate_all": 0.05,          # among all known (benign + known attacks)
    "max_false_unknown_rate_known_attacks": 0.10, # among known attacks only
},

# Formal calibration reporting
"calibration_reporting": {
    "enabled": True,
    "ece_bins": 15,     # 10–15 bins recommended
    "plot_dpi": 160,
},


    # Processed columns
    "y_stage1": "y_stage1_attack",   # 0/1
    "y_stage2": "y_stage2_family",   # string family for attack samples; benign may be empty/NA

    # Columns that must never be used as features
    "never_feature_cols": {"label", "attempted_category", "y_stage1_attack", "y_stage2_family", "y_stage2_fine"},

    # Preprocessor
    "max_unique_per_cat_col": 50,
    "cat_na_token": "NA",
    "chunksize_rows": 400_000,
    "scale_numeric": True,
    "max_stat_rows": 600_000,

    # Stage-1 XGB grid (binary)
    "stage1_xgb_grid": {
        "enabled": True,
        "max_train_rows": {"CICIDS2017": 900_000, "CICIoT2023": 1_500_000},
        "max_val_rows":   {"CICIDS2017": 500_000, "CICIoT2023": 650_000},
        "seed": 42,
        "results_csv": "stage1_xgb_grid_results.csv",
        "progress_json": "stage1_xgb_grid_progress.json",
        "models_dir": "stage1_xgb_grid_models",
        "target_fpr": 0.01,
# =========================
# Sweep controls (A–C)
# =========================
# Selection bars (used when choosing the "best" config)
"fpr_bar": 0.015,                 # user-set: 1.5% max operating FPR
# Evaluate operating thresholds at these target FPRs (A)
"target_fpr_list": [0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.050],
# Family-aware objective knobs (A)
"min_family_support_list": [50, 100, 200, 300],
"objective_mode_list": ["min_family_recall", "p10_family_recall"],
"p10_quantile_list": [0.05, 0.10, 0.20],
# Selection preference (critical for CICIDS tail safety)
"select_objective_mode": "min_family_recall",
# Stage-1 cost sensitivity (B)
"weight_mode_list": ["none", "inv_family", "inv_family_clipped"],
"inv_family_clip_max": 20.0,
# Early stopping (C)
"use_early_stopping_list": [False, True],
"early_stop_n_estimators": 3000,
"early_stopping_rounds": 50,

        # Option C knobs
        "min_family_support": 300,          # ignore tiny families in min-recall objective
        "objective_mode": "p10_family_recall",  # "min_family_recall" | "p10_family_recall"
        "p10_quantile": 0.10,

        "grid": [
            {"n_estimators": 800, "max_depth": 6,  "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            {"n_estimators": 1200,"max_depth": 8,  "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            {"n_estimators": 1400,"max_depth": 10, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1.0},
            {"n_estimators": 1800,"max_depth": 10, "learning_rate": 0.02, "subsample": 0.85,"colsample_bytree": 0.75,"reg_lambda": 1.2},
        ],
        # GPU options
        "tree_method": "hist",
        "device": "cuda",
        "predictor": None,
        "n_jobs": 0,
        "verbosity": 0,
        "eval_metric": "logloss",
    },

    # Stage-2 XGB grid (multiclass)
    "stage2_xgb_grid": {
        "enabled": True,
        "max_train_rows": {"CICIDS2017": 1_000_000, "CICIoT2023": 1_500_000},
        "max_val_rows":   {"CICIDS2017": 600_000, "CICIoT2023": 800_000},
        "seed": 43,
        "results_csv": "stage2_xgb_grid_results.csv",
        "progress_json": "stage2_xgb_grid_progress.json",
        "models_dir": "stage2_xgb_grid_models",
        "grid": [
            {"n_estimators": 1400, "max_depth": 8,  "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
            {"n_estimators": 1800, "max_depth": 10, "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.75,"reg_lambda": 1.2},
            {"n_estimators": 2200, "max_depth": 12, "learning_rate": 0.02, "subsample": 0.85,"colsample_bytree": 0.70,"reg_lambda": 1.4},
        ],
        # GPU options
        "tree_method": "hist",
        "device": "cuda",
        "predictor": None,
        "n_jobs": 0,
        "verbosity": 0,
        "eval_metric": "mlogloss",
    },

    # Stage-1 strict threshold (thr_high)
    "stage1_threshold": {
        # Option C: family-aware threshold selection under FPR constraint
        "policy": "target_fpr_family_aware",  # "target_fpr_family_aware" | "target_fpr" | "f1" | "fixed"
        "target_fpr": 0.015,
        "fixed": 0.5,
        "min_family_support": 50,
        "sweep_points": 60,        # threshold sweep points (constrained by FPR)
        "objective_mode": "min_family_recall",  # "min_family_recall" | "p10_family_recall"
        "p10_quantile": 0.10,
    },

    # Cascade gating (thr_low candidates; joint tuned with tau)
    "cascade_gate": {
        "enabled": True,
        "grid_points": 30,
        "max_val_rows_for_sweep": 700_000,
    },

    # Reject/abstain (uncertainty) settings
    "abstain": {
        "enabled": True,
        # policy:
        #   - "abstain": output label=unknown when uncertain
        #   - "reject_to_benign": map unknown -> Benign (still logged as "reject")
        "policy": "abstain",
        "label": "Unknown",

        # Constraints during tuning (validation):
        "max_benign_family_fp_rate": 0.02,   # among true benign, predicted as attack family
        "max_benign_reject_rate": 0.10,      # among true benign, rejected
        "max_overall_reject_rate": 0.01,     # overall rejected
# Sweep controls (D): evaluate multiple constraint bars cheaply on validation.
"sweep_enabled": True,
"max_benign_family_fp_rate_list": [0.01, 0.02, 0.03],
"max_overall_reject_rate_list": [0.01],

        "grid": [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "also_try_quantiles": True,
        "quantiles": [0.05, 0.10, 0.20, 0.30],
        "max_val_rows_for_sweep": 700_000,
    },

    "runs_root": "runs_two_stage_V4_protocolB_LOAO_calib",
    "resume_run_dirs": {"CICIDS2017": None, "CICIoT2023": None},
    "global_seed": 123,
}


CFG["processed_root"] = repo_path(str(CFG["processed_root"]))

PROTOCOL_DEFAULTS = {
    "A": {
        "processed_root": "processed_V5",
        "protocol": "A_stratified",
        "datasets": ["CICIDS2017", "CICIoT2023", "NSL-KDD", "UNSW-NB15"],
        "chunksize_rows": 200_000,
        "loao": {"enabled": False, "apply_to_stage1": False},
        "runs_root": "runs_two_stage_V5_A_core",
    },
    "B": {
        "processed_root": "processed_V5_cicids17_recovery",
        "protocol": "B_day_file",
        "datasets": ["CICIDS2017", "CICIoT2023"],
        "chunksize_rows": 400_000,
        "loao": {"enabled": True, "apply_to_stage1": False},
        "runs_root": "runs_two_stage_V4_protocolB_LOAO_calib",
    },
}


def configure_for_protocol(protocol: str, overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Apply Protocol A/B defaults and caller overrides to the module CFG."""
    key = protocol.strip().upper()
    if key not in PROTOCOL_DEFAULTS:
        raise ValueError(f"Unknown protocol: {protocol!r}. Expected 'A' or 'B'.")
    deep_update(CFG, PROTOCOL_DEFAULTS[key])
    if overrides:
        deep_update(CFG, overrides)
    CFG["processed_root"] = resolve_repo_path(str(CFG["processed_root"]))
    CFG["runs_root"] = resolve_repo_path(str(CFG["runs_root"]))
    return CFG


def run_protocol(protocol: str, overrides: Optional[Dict[str, object]] = None, dry_run: bool = False) -> Dict[str, object] | None:
    """Run the merged engine for one protocol, or print the resolved plan."""
    configure_for_protocol(protocol, overrides=overrides)
    if dry_run:
        progress_print(f"[dry-run] protocol={CFG['protocol']}")
        progress_print(f"[dry-run] processed_root={CFG['processed_root']}")
        progress_print(f"[dry-run] runs_root={CFG['runs_root']}")
        progress_print(f"[dry-run] datasets={CFG['datasets']}")
        progress_print(f"[dry-run] loao={CFG.get('loao', {})}")
        return CFG
    main()
    return None


def run_protocol_a_core(config: Optional[Dict[str, object]] = None, dry_run: bool = False):
    """Run the canonical Protocol A closed-set two-stage core runner."""
    from ids_eval_framework._native import protocol_a_core
    from ids_eval_framework.src.native_runtime import run_native_main

    cfg = config or {}
    runner_cfg = cfg.get("two_stage_engine", {}) or {}
    overrides = runner_cfg.get("legacy_overrides", {})
    return run_native_main(
        protocol_a_core,
        cfg_overrides=overrides,
        dry_run=dry_run,
    )


def run_protocol_b_loao_grid(config: Optional[Dict[str, object]] = None, dry_run: bool = False):
    """Run the support-audited Protocol B/LOAO grid runner."""
    from ids_eval_framework._native import protocol_b_grid, protocol_b_summary
    from ids_eval_framework.src.native_runtime import run_native_main

    cfg = config or {}
    grid_cfg = cfg.get("protocol_b_grid", {}) or {}
    overrides = grid_cfg.get("legacy_overrides", {})
    result = run_native_main(
        protocol_b_grid,
        cfg_overrides=overrides,
        dry_run=dry_run,
    )
    summary_cfg = cfg.get("protocol_b_summary", {}) or {}
    if summary_cfg.get("run_after_grid", True):
        run_native_main(
            protocol_b_summary,
            cfg_overrides=summary_cfg.get("legacy_overrides", {}),
            dry_run=dry_run,
        )
    return result


# ============================================================
# Utils
# ============================================================

def ensure_deps():
    if joblib is None:
        raise RuntimeError("joblib is required. Install: pip install joblib")
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required. Install: pip install xgboost")

def makedirs(p: str):
    os.makedirs(p, exist_ok=True)

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def progress_print(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass

def canonical_col(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"\s+", "_", c)
    return c

def sha1_obj(o: dict) -> str:
    s = json.dumps(o, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]

def safe_joblib_dump(obj, path: str):
    tmp = path + ".tmp"
    joblib.dump(obj, tmp)
    os.replace(tmp, path)

def safe_joblib_load(path: str):
    return joblib.load(path)

def read_part(path: str, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path, columns=usecols)
    elif path.lower().endswith(".csv") or path.lower().endswith(".csv.gz"):
        df = pd.read_csv(path, usecols=usecols)
    else:
        raise RuntimeError(f"Unsupported part: {path}")
    return df

def list_parts(ds_dir: str, split: str) -> List[str]:
    p1 = sorted(glob.glob(os.path.join(ds_dir, split, "*.parquet")))
    p2 = sorted(glob.glob(os.path.join(ds_dir, split, "*.csv")))
    p3 = sorted(glob.glob(os.path.join(ds_dir, split, "*.csv.gz")))
    p4 = sorted(glob.glob(os.path.join(ds_dir, f"{split}_*.parquet")))
    p5 = sorted(glob.glob(os.path.join(ds_dir, f"{split}_*.csv")))
    p6 = sorted(glob.glob(os.path.join(ds_dir, f"{split}_*.csv.gz")))
    return p1 + p2 + p3 + p4 + p5 + p6

def iter_rows_from_parts(parts: List[str], usecols: Optional[List[str]], chunksize: int) -> Iterable[pd.DataFrame]:
    for p in parts:
        if p.lower().endswith(".parquet"):
            yield read_part(p, usecols=usecols)
        else:
            for chunk in pd.read_csv(p, usecols=usecols, chunksize=chunksize):
                yield chunk

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


# ============================================================
# Preprocessor (numeric + limited one-hot categorical) -> sparse CSR
# ============================================================

@dataclass
class Preprocessor:
    num_cols: List[str]
    cat_cols: List[str]
    num_mean: np.ndarray
    num_std: np.ndarray
    cat_maps: Dict[str, Dict[str, int]]   # col -> category->index within that col
    cat_offsets: Dict[str, int]           # col -> global offset
    n_num: int
    n_cat: int

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        df = df.copy()
        df.columns = [canonical_col(c) for c in df.columns]

        if self.n_num > 0:
            Xn = df.reindex(columns=self.num_cols).to_numpy(dtype=np.float64, copy=False)
            if CFG["scale_numeric"]:
                Xn = (Xn - self.num_mean) / self.num_std
            Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
            Xn_sp = csr_matrix(Xn)
        else:
            Xn_sp = csr_matrix((len(df), 0), dtype=np.float64)

        if self.n_cat > 0:
            rows, cols, data = [], [], []
            for col in self.cat_cols:
                vals = df[col].astype(str).fillna(CFG["cat_na_token"]).to_numpy()
                cmap = self.cat_maps[col]
                off = self.cat_offsets[col]
                for i, v in enumerate(vals):
                    j = cmap.get(v)
                    if j is None:
                        j = cmap.get(CFG["cat_na_token"])
                    if j is not None:
                        rows.append(i); cols.append(off + j); data.append(1.0)
            Xc = csr_matrix((data, (rows, cols)), shape=(len(df), self.n_cat), dtype=np.float64)
        else:
            Xc = csr_matrix((len(df), 0), dtype=np.float64)

        return hstack([Xn_sp, Xc], format="csr")


def infer_cols(sample_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    sample_df = sample_df.copy()
    sample_df.columns = [canonical_col(c) for c in sample_df.columns]
    drop = set(CFG["never_feature_cols"]) | {canonical_col(CFG["y_stage1"]), canonical_col(CFG["y_stage2"])}
    cols = [c for c in sample_df.columns if canonical_col(c) not in drop]

    num_cols, cat_cols = [], []
    for c in cols:
        s = sample_df[c]
        if pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            nunq = int(s.astype(str).nunique(dropna=True))
            if nunq <= CFG["max_unique_per_cat_col"]:
                cat_cols.append(c)
            else:
                pass
    return num_cols, cat_cols


def fit_preprocessor(ds_dir: str, run_dir: str) -> Preprocessor:
    parts = list_parts(ds_dir, "train")
    if not parts:
        raise RuntimeError(f"No train parts found in {ds_dir}")

    df0 = read_part(parts[0], usecols=None)
    df0.columns = [canonical_col(c) for c in df0.columns]
    num_cols, cat_cols = infer_cols(df0.head(5000))

    max_stat = int(CFG["max_stat_rows"])
    num_acc = []
    seen = 0

    usecols = list(set(num_cols + cat_cols + [CFG["y_stage1"], CFG["y_stage2"]]))
    usecols = [canonical_col(c) for c in usecols]

    cat_counts: Dict[str, Dict[str, int]] = {c: {} for c in cat_cols}

    for chunk in iter_rows_from_parts(parts, usecols=usecols, chunksize=CFG["chunksize_rows"]):
        chunk.columns = [canonical_col(c) for c in chunk.columns]
        if num_cols:
            Xn = chunk.reindex(columns=num_cols).to_numpy(dtype=np.float64, copy=False)
            num_acc.append(Xn)
        for c in cat_cols:
            vals = chunk[c].astype(str).fillna(CFG["cat_na_token"]).to_numpy()
            cc = cat_counts[c]
            for v in vals:
                cc[v] = cc.get(v, 0) + 1

        seen += len(chunk)
        if seen >= max_stat:
            break

    if num_cols:
        Xn_all = np.vstack(num_acc) if len(num_acc) > 1 else num_acc[0]
        if len(Xn_all) > max_stat:
            rng = np.random.RandomState(CFG["global_seed"])
            idx = rng.choice(len(Xn_all), size=max_stat, replace=False)
            Xn_all = Xn_all[idx]
        mean = np.nanmean(Xn_all, axis=0)
        std = np.nanstd(Xn_all, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    else:
        mean = np.zeros((0,), dtype=np.float64)
        std = np.ones((0,), dtype=np.float64)

    cat_maps, cat_offsets = {}, {}
    off = 0
    for c in cat_cols:
        vc = cat_counts[c]
        items = sorted(vc.items(), key=lambda kv: kv[1], reverse=True)
        top = [k for k, _ in items[:CFG["max_unique_per_cat_col"]]]
        if CFG["cat_na_token"] not in top:
            top.append(CFG["cat_na_token"])
        cmap = {k: i for i, k in enumerate(top)}
        cat_maps[c] = cmap
        cat_offsets[c] = off
        off += len(cmap)

    prep = Preprocessor(
        num_cols=num_cols, cat_cols=cat_cols,
        num_mean=mean.astype(np.float64), num_std=std.astype(np.float64),
        cat_maps=cat_maps, cat_offsets=cat_offsets,
        n_num=len(num_cols), n_cat=off
    )

    with open(os.path.join(run_dir, "preprocessor_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"n_num": prep.n_num, "n_cat": prep.n_cat, "num_cols": prep.num_cols, "cat_cols": prep.cat_cols}, f, indent=2)

    safe_joblib_dump(prep, os.path.join(run_dir, "preprocessor.joblib"))
    return prep


# ============================================================
# Data collection helpers
# ============================================================

def collect_xy(parts: List[str],
               prep: Preprocessor,
               y_col: str,
               max_rows: int,
               seed: int,
               filter_attack: Optional[bool] = None,
               y2_col: Optional[str] = None) -> Tuple[csr_matrix, np.ndarray, Optional[np.ndarray]]:
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    y2_list = [] if y2_col else None

    remaining = max_rows
    usecols = prep.num_cols + prep.cat_cols + [y_col]
    if y2_col:
        usecols.append(y2_col)
    usecols = [canonical_col(c) for c in usecols]

    for chunk in iter_rows_from_parts(parts, usecols=usecols, chunksize=CFG["chunksize_rows"]):
        chunk.columns = [canonical_col(c) for c in chunk.columns]

        if filter_attack is not None:
            ytmp = chunk[y_col].astype(int).to_numpy()
            mask = (ytmp == (1 if filter_attack else 0))
            chunk = chunk.loc[mask]
            if len(chunk) == 0:
                continue

        take = min(len(chunk), remaining)
        if take <= 0:
            break

        if len(chunk) > take:
            idx = rng.choice(len(chunk), size=take, replace=False)
            chunk = chunk.iloc[idx]

        y = chunk[y_col].astype(int).to_numpy()
        if y2_col:
            y2 = chunk[y2_col].astype(str).fillna("").to_numpy()

        X = prep.transform(chunk.drop(columns=[y_col] + ([y2_col] if y2_col else []), errors="ignore"))

        X_list.append(X)
        y_list.append(y)
        if y2_col:
            y2_list.append(y2)

        remaining -= take
        if remaining <= 0:
            break

    if not X_list:
        X = csr_matrix((0, prep.n_num + prep.n_cat), dtype=np.float64)
        y = np.array([], dtype=int)
        y2 = np.array([], dtype=object) if y2_col else None
        return X, y, y2

    X = vstack(X_list, format="csr")
    y = np.concatenate(y_list).astype(int)
    y2 = np.concatenate(y2_list).astype(object) if y2_col else None
    return X, y, y2


# ============================================================
# Calibration
# ============================================================

def fit_platt_on_probs(p: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    if np.unique(y).size < 2:
        return None
    s = logit(np.clip(p, 1e-12, 1 - 1e-12)).astype(np.float64)
    y = y.astype(np.float64)

    def nll(theta):
        a, b = theta
        q = sigmoid(a * s + b)
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    a, b = res.x
    return {"a": float(a), "b": float(b)}

def apply_platt(p: np.ndarray, platt: Optional[Dict[str, float]]) -> np.ndarray:
    if platt is None:
        return p
    a, b = platt["a"], platt["b"]
    s = logit(np.clip(p, 1e-12, 1 - 1e-12))
    return sigmoid(a * s + b)

def fit_temperature_on_probs(p: np.ndarray, y_idx: np.ndarray, K: int) -> float:
    if len(p) == 0:
        return 1.0
    p = np.clip(p, 1e-12, 1.0)
    lp = np.log(p).astype(np.float64)
    labels_fixed = list(range(K))

    def nll(x):
        T = float(np.clip(x[0], 0.05, 50.0))
        scaled = softmax(lp / T, axis=1)
        scaled = np.clip(scaled, 1e-12, 1.0)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        return float(log_loss(y_idx, scaled, labels=labels_fixed))

    res = minimize(nll, x0=np.array([1.0]), method="Nelder-Mead")
    return float(np.clip(res.x[0], 0.05, 50.0))

def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    if T is None or abs(float(T) - 1.0) < 1e-12:
        return p / p.sum(axis=1, keepdims=True)
    lp = np.log(p)
    out = softmax(lp / float(T), axis=1)
    out = np.clip(out, 1e-12, 1.0)
    return out / out.sum(axis=1, keepdims=True)




# ============================================================
# Calibration reporting (Brier / ECE / Reliability)
# ============================================================

def brier_binary(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    p = np.clip(p.astype(np.float64), 1e-12, 1.0 - 1e-12)
    return float(np.mean((p - y_true) ** 2))

def ece_binary(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> Tuple[float, List[dict]]:
    """
    Expected Calibration Error (ECE) for binary classification.
    Returns (ece, bins_stats).
    bins_stats: list of dicts {bin_lo, bin_hi, n, acc, conf, gap}
    """
    y_true = y_true.astype(np.int32)
    p = np.clip(p.astype(np.float64), 1e-12, 1.0 - 1e-12)

    bins = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    ece = 0.0
    stats = []
    N = len(p)
    for i in range(len(bins) - 1):
        lo, hi = float(bins[i]), float(bins[i + 1])
        # Right-inclusive for last bin
        if i == len(bins) - 2:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        n = int(m.sum())
        if n == 0:
            stats.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "acc": None, "conf": None, "gap": None})
            continue
        acc = float(np.mean(y_true[m]))
        conf = float(np.mean(p[m]))
        gap = abs(acc - conf)
        ece += (n / max(1, N)) * gap
        stats.append({"bin_lo": lo, "bin_hi": hi, "n": n, "acc": acc, "conf": conf, "gap": float(gap)})
    return float(ece), stats

def ece_toplabel(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> Tuple[float, List[dict]]:
    """
    Top-label ECE for multiclass (uses confidence=max prob and correctness flag).
    Returns (ece, bins_stats).
    """
    conf = np.clip(conf.astype(np.float64), 1e-12, 1.0)
    correct = correct.astype(np.float64)

    bins = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    ece = 0.0
    stats = []
    N = len(conf)
    for i in range(len(bins) - 1):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == len(bins) - 2:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        n = int(m.sum())
        if n == 0:
            stats.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "acc": None, "conf": None, "gap": None})
            continue
        acc = float(np.mean(correct[m]))
        cbar = float(np.mean(conf[m]))
        gap = abs(acc - cbar)
        ece += (n / max(1, N)) * gap
        stats.append({"bin_lo": lo, "bin_hi": hi, "n": n, "acc": acc, "conf": cbar, "gap": float(gap)})
    return float(ece), stats

def plot_reliability_curve(bin_stats: List[dict], out_png: str, title: str, dpi: int = 160):
    """Generic reliability plot from bin_stats produced by ece_*."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    xs, ys = [], []
    for b in bin_stats:
        if b.get("n", 0) > 0 and b.get("conf") is not None and b.get("acc") is not None:
            xs.append(b["conf"]); ys.append(b["acc"])

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    if xs:
        plt.plot(xs, ys, marker="o")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=int(dpi))
    plt.close()

# ============================================================
# Metrics helpers (Fix 1)
# ============================================================

def macro_f1_fixedK(y_true: np.ndarray, y_pred: np.ndarray, K: int) -> float:
    labels = list(range(K))
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))

def macro_f1_present(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    present = sorted(np.unique(y_true).tolist())
    return float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0))

def present_labels(y_true: np.ndarray) -> List[int]:
    return sorted(np.unique(y_true).tolist())

def missing_labels(y_true: np.ndarray, K: int) -> List[int]:
    pres = set(np.unique(y_true).tolist())
    return [i for i in range(K) if i not in pres]


# ============================================================
# Threshold policies + Option C (family-aware)
# ============================================================

def pick_threshold_target_fpr(y_true_bin: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    benign = p[y_true_bin == 0]
    if len(benign) == 0:
        return 0.5
    q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
    return float(np.quantile(benign, q))

def family_recall_stats(y1_true: np.ndarray,
                        y2_family: np.ndarray,
                        pred_attack: np.ndarray,
                        min_support: int) -> Tuple[Dict[str, float], float, float]:
    # returns (per_family_recall, min_recall, p10_recall) on families with support>=min_support
    y1_true = y1_true.astype(int)
    mask_attack = (y1_true == 1)
    if mask_attack.sum() == 0:
        return {}, 0.0, 0.0

    fam = y2_family[mask_attack].astype(str)
    pa = pred_attack[mask_attack].astype(bool)

    recalls = {}
    supports = {}
    for f in fam:
        supports[f] = supports.get(f, 0) + 1

    for f, s in supports.items():
        if s < min_support or f == "" or f.lower() == "nan":
            continue
        m = (fam == f)
        recalls[f] = float(np.mean(pa[m])) if m.sum() > 0 else 0.0

    if not recalls:
        # if everything is tiny, fall back to attack recall
        attack_recall = float(np.mean(pa)) if len(pa) else 0.0
        return {}, attack_recall, attack_recall

    vals = np.array(list(recalls.values()), dtype=np.float64)
    min_rec = float(np.min(vals))
    p10 = float(np.quantile(vals, 0.10))
    return recalls, min_rec, p10

def choose_thr_high_family_aware(y1_val: np.ndarray,
                                y2_val: np.ndarray,
                                p_attack: np.ndarray,
                                target_fpr: float,
                                min_family_support: int,
                                sweep_points: int,
                                objective_mode: str,
                                p10_q: float) -> Tuple[float, Dict]:
    # Constrained threshold sweep: only consider thresholds that satisfy FPR <= target_fpr.
    benign = p_attack[y1_val == 0]
    if len(benign) == 0:
        return 0.5, {"note": "no benign in val"}

    # Candidate thresholds: from quantiles of benign distribution from (1-target_fpr) up to ~1.0
    q0 = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
    qs = np.linspace(q0, 0.9999, max(10, int(sweep_points)))
    thr_candidates = sorted({float(np.quantile(benign, q)) for q in qs})

    rows = []
    best = None

    for thr in thr_candidates:
        pred_attack = (p_attack >= thr).astype(int)
        fp = int(((y1_val == 0) & (pred_attack == 1)).sum())
        tn = int(((y1_val == 0) & (pred_attack == 0)).sum())
        fpr = fp / max(1, fp + tn)
        if fpr > target_fpr + 1e-12:
            continue

        # overall TPR
        tp = int(((y1_val == 1) & (pred_attack == 1)).sum())
        fn = int(((y1_val == 1) & (pred_attack == 0)).sum())
        tpr = tp / max(1, tp + fn)

        recs, min_rec, p10_rec = family_recall_stats(y1_val, y2_val, pred_attack.astype(bool), min_family_support)
        obj = min_rec if objective_mode == "min_family_recall" else float(np.quantile(list(recs.values()), p10_q) if recs else tpr)

        row = {"thr": float(thr), "fpr": float(fpr), "tpr": float(tpr),
               "min_family_recall": float(min_rec), "p10_family_recall": float(p10_rec),
               "objective": float(obj)}
        rows.append(row)

        if best is None or obj > best["objective"] or (obj == best["objective"] and tpr > best["tpr"]):
            best = row

    if not rows:
        # fallback to target-FPR quantile
        thr = pick_threshold_target_fpr(y1_val, p_attack, target_fpr)
        return float(thr), {"fallback": True, "thr": float(thr)}

    meta = {"rows": rows, "best": best, "objective_mode": objective_mode, "min_family_support": int(min_family_support)}
    return float(best["thr"]), meta


# ============================================================
# System layer (labels + reject metrics)
# ============================================================

def system_labels(families: List[str], use_abstain_label: bool, abstain_label: str) -> List[str]:
    if use_abstain_label:
        return ["Benign"] + list(families) + [abstain_label]
    return ["Benign"] + list(families)

def macro_f1_fixed(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))

def benign_family_fp_rate(y_true_sys: np.ndarray, y_pred_sys: np.ndarray, families: List[str]) -> float:
    benign = (y_true_sys == "Benign")
    if benign.sum() == 0:
        return 0.0
    fam = set(families)
    return float(np.mean([p in fam for p in y_pred_sys[benign]]))

def benign_reject_rate(y_true_sys: np.ndarray, reject_mask: np.ndarray) -> float:
    benign = (y_true_sys == "Benign")
    if benign.sum() == 0:
        return 0.0
    return float(np.mean(reject_mask[benign]))

def overall_reject_rate(reject_mask: np.ndarray) -> float:
    return float(np.mean(reject_mask))

def system_predict_raw(p_attack: np.ndarray,
                       fam_pred_idx: np.ndarray,
                       fam_pmax: np.ndarray,
                       families: List[str],
                       thr: float,
                       tau: float,
                       abstain_label: str) -> np.ndarray:
    pred = np.empty(len(p_attack), dtype=object)
    pred[p_attack < thr] = "Benign"
    idx = np.where(p_attack >= thr)[0]
    if len(idx) > 0:
        low = fam_pmax[idx] < tau
        if np.any(low):
            pred[idx[low]] = abstain_label
        hi = ~low
        if np.any(hi):
            pred[idx[hi]] = np.array([families[i] for i in fam_pred_idx[idx[hi]]], dtype=object)
    return pred

def apply_abstain_policy(pred_raw: np.ndarray, abstain_label: str, policy: str) -> Tuple[np.ndarray, np.ndarray]:
    reject = (pred_raw == abstain_label)
    if policy == "reject_to_benign":
        pred = pred_raw.copy()
        pred[reject] = "Benign"
        return pred, reject
    return pred_raw.copy(), reject


def build_tuning_eval_cache(
    y_sys_true: np.ndarray,
    families: List[str],
    abstain_label: str,
    policy: str,
) -> Dict[str, object]:
    """Pre-encode labels once for repeated validation threshold/tau sweeps."""
    use_abstain_label = policy == "abstain"
    labels = system_labels(families, use_abstain_label, abstain_label)
    unknown_code = len(labels)
    label_to_code = {str(label): i for i, label in enumerate(labels)}
    y_code = np.array([label_to_code.get(str(value), unknown_code) for value in y_sys_true], dtype=np.int32)
    if use_abstain_label:
        unknown_code = label_to_code[str(abstain_label)]
    true_benign = y_code == 0
    true_unknown = y_code == int(unknown_code) if use_abstain_label else np.zeros(len(y_code), dtype=bool)
    return {
        "y_code": y_code,
        "label_count": len(labels),
        "family_count": len(families),
        "unknown_code": int(unknown_code),
        "use_abstain_label": bool(use_abstain_label),
        "true_benign": true_benign,
        "true_unknown": true_unknown,
        "known_all": ~true_unknown,
        "known_attacks": (y_code != 0) & (~true_unknown),
    }


def _macro_f1_from_codes(y_code: np.ndarray, pred_code: np.ndarray, label_count: int) -> float:
    if label_count <= 0:
        return 0.0
    max_code = int(max(label_count, int(y_code.max(initial=0)) + 1, int(pred_code.max(initial=0)) + 1))
    true_counts = np.bincount(y_code, minlength=max_code)
    pred_counts = np.bincount(pred_code, minlength=max_code)
    matched = y_code == pred_code
    tp_counts = np.bincount(y_code[matched], minlength=max_code)

    scores = []
    for label in range(label_count):
        tp = float(tp_counts[label])
        fp = float(pred_counts[label] - tp_counts[label])
        fn = float(true_counts[label] - tp_counts[label])
        denom = (2.0 * tp) + fp + fn
        scores.append(0.0 if denom <= 0.0 else (2.0 * tp) / denom)
    return float(np.mean(scores))


def eval_system_on_val_cached(
    cache: Dict[str, object],
    p_attack: np.ndarray,
    fam_pred_idx: np.ndarray,
    fam_pmax: np.ndarray,
    thr: float,
    tau: float,
) -> Dict[str, float]:
    y_code = cache["y_code"]
    if not isinstance(y_code, np.ndarray):
        raise TypeError("tuning cache y_code must be a numpy array")

    n = int(len(y_code))
    if n == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "benign_family_fp_rate": 0.0,
            "benign_reject_rate": 0.0,
            "overall_reject_rate": 0.0,
            "unknown_detection_rate": None,
            "false_unknown_rate_all_known": None,
            "false_unknown_rate_known_attacks": None,
            "n_true_unknown": None,
        }

    pred_code = np.zeros(n, dtype=np.int32)
    attack_mask = p_attack >= float(thr)
    reject = attack_mask & (fam_pmax < float(tau))
    family_mask = attack_mask & (~reject)
    if np.any(family_mask):
        pred_code[family_mask] = 1 + fam_pred_idx[family_mask].astype(np.int32)
    if bool(cache["use_abstain_label"]) and np.any(reject):
        pred_code[reject] = int(cache["unknown_code"])

    label_count = int(cache["label_count"])
    family_count = int(cache["family_count"])
    true_benign = cache["true_benign"]
    if not isinstance(true_benign, np.ndarray):
        raise TypeError("tuning cache true_benign must be a numpy array")

    accuracy = float(np.mean(y_code == pred_code))
    macro_f1 = _macro_f1_from_codes(y_code, pred_code, label_count)
    benign_n = int(true_benign.sum())
    if benign_n > 0:
        benign_pred = pred_code[true_benign]
        benign_family_fp = (benign_pred >= 1) & (benign_pred <= family_count)
        benign_family_fp_rate_value = float(np.mean(benign_family_fp))
        benign_reject_rate_value = float(np.mean(reject[true_benign]))
    else:
        benign_family_fp_rate_value = 0.0
        benign_reject_rate_value = 0.0

    use_abstain_label = bool(cache["use_abstain_label"])
    if use_abstain_label:
        unknown_code = int(cache["unknown_code"])
        true_unknown = cache["true_unknown"]
        known_all = cache["known_all"]
        known_attacks = cache["known_attacks"]
        if not isinstance(true_unknown, np.ndarray):
            raise TypeError("tuning cache true_unknown must be a numpy array")
        if not isinstance(known_all, np.ndarray) or not isinstance(known_attacks, np.ndarray):
            raise TypeError("tuning cache known masks must be numpy arrays")
        pred_unknown = pred_code == unknown_code
        n_true_unknown = int(true_unknown.sum())
        unknown_detection_rate = float(np.mean(pred_unknown[true_unknown])) if n_true_unknown > 0 else None
        false_unknown_rate_all = float(np.mean(pred_unknown[known_all])) if int(known_all.sum()) > 0 else None
        false_unknown_rate_known_attacks = (
            float(np.mean(pred_unknown[known_attacks])) if int(known_attacks.sum()) > 0 else None
        )
    else:
        n_true_unknown = None
        unknown_detection_rate = None
        false_unknown_rate_all = None
        false_unknown_rate_known_attacks = None

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "benign_family_fp_rate": benign_family_fp_rate_value,
        "benign_reject_rate": benign_reject_rate_value,
        "overall_reject_rate": float(np.mean(reject)),
        "unknown_detection_rate": unknown_detection_rate,
        "false_unknown_rate_all_known": false_unknown_rate_all,
        "false_unknown_rate_known_attacks": false_unknown_rate_known_attacks,
        "n_true_unknown": n_true_unknown,
    }


# ============================================================
# XGBoost model builders
# ============================================================

def build_xgb_binary(params: dict) -> XGBClassifier:
    base = dict(objective="binary:logistic", random_state=CFG["global_seed"], **params)
    return XGBClassifier(**base)

def build_xgb_multi(params: dict, num_class: int) -> XGBClassifier:
    base = dict(objective="multi:softprob", num_class=int(num_class), random_state=CFG["global_seed"], **params)
    return XGBClassifier(**base)


# ============================================================
# Stage-1 grid search (Option C objective)
# ============================================================

def eval_tpr_at_fpr(y_true: np.ndarray, p: np.ndarray, target_fpr: float) -> Tuple[float, float, float]:
    thr = pick_threshold_target_fpr(y_true, p, target_fpr)
    pred = (p >= thr).astype(int)
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fpr = fp / max(1, fp + tn)
    tpr = tp / max(1, tp + fn)
    return float(tpr), float(fpr), float(thr)

def run_stage1_xgb_grid(ds_name: str, ds_dir: str, run_dir: str, prep: Preprocessor) -> Tuple[XGBClassifier, Optional[Dict[str, float]]]:
    """
    Stage-1 (attack gate) training + sweep (Options A–C):
      - A: evaluate multiple operating-point targets (target_fpr_list) + objective knobs
      - B: cost sensitivity via sample_weight modes
      - C: early stopping mode
    We TRAIN a manageable number of models, then CHEAPLY evaluate many threshold/objective combos on val.
    """
    cfg = CFG["stage1_xgb_grid"]
    makedirs(os.path.join(run_dir, cfg["models_dir"]))

    results_path = os.path.join(run_dir, cfg["results_csv"])
    prog_path = os.path.join(run_dir, cfg["progress_json"])

    done_models = set()
    best_combo = None
    if os.path.exists(prog_path):
        state = json.load(open(prog_path, "r", encoding="utf-8"))
        done_models = set(state.get("done_model_hashes", []))
        best_combo = state.get("best_combo", None)

    parts_tr = list_parts(ds_dir, "train")
    parts_va = list_parts(ds_dir, "val")
    if not parts_tr or not parts_va:
        raise RuntimeError("Missing train/val parts.")

    max_tr = int(cfg["max_train_rows"].get(ds_name, 800_000))
    max_va = int(cfg["max_val_rows"].get(ds_name, 400_000))

    # Collect y2_family for train to compute family-aware weights (Option B)
    Xtr, ytr, y2tr = collect_xy(parts_tr, prep, CFG["y_stage1"], max_tr, seed=cfg["seed"], filter_attack=None, y2_col=CFG["y_stage2"])
    Xva, yva, y2va = collect_xy(parts_va, prep, CFG["y_stage1"], max_va, seed=cfg["seed"] + 1, filter_attack=None, y2_col=CFG["y_stage2"])
    if y2tr is None:
        y2tr = np.array([""] * len(ytr), dtype=object)

    # LOAO (optional): remove holdout-family attack rows from Stage-1 TRAINING only (stricter protocol).
    lo = CFG.get("loao", {}) or {}
    if bool(lo.get("enabled", False)) and bool(lo.get("apply_to_stage1", False)):
        holdout = None
        hf = lo.get("holdout_family", None)
        if isinstance(hf, dict):
            holdout = hf.get(ds_name, None)
        elif isinstance(hf, str):
            holdout = hf
        holdout = str(holdout) if holdout else ""
        if holdout:
            y2tr_str = np.array([str(x) for x in y2tr], dtype=object)
            mask_keep = np.ones(len(ytr), dtype=bool)
            mask_keep[(ytr == 1) & (y2tr_str == holdout)] = False
            if mask_keep.sum() > 0 and mask_keep.sum() < len(mask_keep):
                Xtr = Xtr[mask_keep]
                ytr = ytr[mask_keep]
                y2tr = y2tr[mask_keep]
                with open(os.path.join(run_dir, "loao_stage1_filtered.json"), "w", encoding="utf-8") as f:
                    json.dump({"holdout_family": holdout, "removed_rows": int((~mask_keep).sum())}, f, indent=2)


    if y2va is None:
        y2va = np.array([""] * len(yva), dtype=object)

    # Load existing rows if present
    rows = []
    if os.path.exists(results_path):
        try:
            rows = pd.read_csv(results_path).to_dict("records")
        except Exception:
            rows = []

    # Sweeps (A–C)
    target_fprs = [float(x) for x in cfg.get("target_fpr_list", [float(cfg.get("target_fpr", 0.01))])]
    min_sups = [int(x) for x in cfg.get("min_family_support_list", [int(cfg.get("min_family_support", 200))])]
    modes = list(cfg.get("objective_mode_list", [cfg.get("objective_mode", "min_family_recall")]))
    p10_qs = [float(x) for x in cfg.get("p10_quantile_list", [float(cfg.get("p10_quantile", 0.10))])]
    weight_modes = list(cfg.get("weight_mode_list", ["none"]))
    clip_max = float(cfg.get("inv_family_clip_max", 20.0))
    use_es_list = list(cfg.get("use_early_stopping_list", [False]))
    es_n_est = int(cfg.get("early_stop_n_estimators", 3000))
    es_rounds = int(cfg.get("early_stopping_rounds", 50))
    fpr_bar = float(cfg.get("fpr_bar", 0.015))

    # Prefer selecting configs that maximize MIN family recall (do not let p-quantile objectives hide a dead family)
    select_mode = cfg.get("select_objective_mode", None)
    if select_mode is not None:
        select_mode = str(select_mode).strip() or None


    def make_stage1_weights(mode_name: str) -> Optional[np.ndarray]:
        """Return per-row sample_weight for training."""
        if mode_name == "none":
            return None

        w = np.ones(len(ytr), dtype=np.float64)
        mask_attack = (ytr == 1)
        fam = np.array([str(x) for x in y2tr], dtype=object)
        fam_attack = fam[mask_attack]
        fam_attack_clean = np.array([f if (f and f != "nan" and f != "None") else "UNKNOWN" for f in fam_attack], dtype=object)
        if fam_attack_clean.size == 0:
            return None

        uniq, cnt = np.unique(fam_attack_clean, return_counts=True)
        inv = {u: 1.0 / float(c) for u, c in zip(uniq, cnt)}
        w_attack = np.array([inv.get(f, 1.0) for f in fam_attack_clean], dtype=np.float64)

        if mode_name == "inv_family_clipped":
            w_attack = np.minimum(w_attack, clip_max)

        w_attack = w_attack / (np.mean(w_attack) + 1e-12)
        w[mask_attack] = w_attack
        return w

    def predict_proba_safe(mdl: XGBClassifier, X: np.ndarray) -> np.ndarray:
        # xgboost sklearn API differs by version; try to respect early-stopping best_iteration if present
        try:
            bi = getattr(mdl, "best_iteration", None)
            if bi is not None:
                return mdl.predict_proba(X, iteration_range=(0, int(bi) + 1))
        except Exception:
            pass
        return mdl.predict_proba(X)

    # Avoid duplicating combo rows on resume
    seen_combo = set()
    for r in rows:
        mh = str(r.get("model_hash", ""))
        ch = str(r.get("combo_hash", ""))
        if mh and ch:
            seen_combo.add((mh, ch))

    # Train models, then sweep operating points/objectives on val
    for g in cfg["grid"]:
        base_params = dict(g)
        base_params.update({
            "tree_method": cfg.get("tree_method", "hist"),
            "n_jobs": cfg.get("n_jobs", 0),
            "verbosity": cfg.get("verbosity", 0),
            "eval_metric": cfg.get("eval_metric", "logloss"),
        })
        if cfg.get("device"):
            base_params["device"] = cfg.get("device")
        if cfg.get("predictor"):
            base_params["predictor"] = cfg.get("predictor")

        for weight_mode in weight_modes:
            sw = make_stage1_weights(weight_mode)

            for use_es in use_es_list:
                params = dict(base_params)
                if bool(use_es):
                    params["n_estimators"] = es_n_est

                model_hash = sha1_obj({"params": params, "weight_mode": weight_mode, "use_early_stopping": bool(use_es)})
                model_path = os.path.join(run_dir, cfg["models_dir"], f"{model_hash}.joblib")

                if model_hash not in done_models:
                    model = build_xgb_binary(params)
                    if bool(use_es):
                        model.set_params(**{"early_stopping_rounds": es_rounds})
                        model.fit(
                            Xtr, ytr,
                            sample_weight=sw,
                            eval_set=[(Xva, yva)],
                            verbose=True
                        )
                    else:
                        model.fit(Xtr, ytr, sample_weight=sw)

                    safe_joblib_dump(model, model_path)
                    done_models.add(model_hash)

                    with open(prog_path, "w", encoding="utf-8") as f:
                        json.dump({"done_model_hashes": sorted(done_models), "best_combo": best_combo}, f, indent=2)

                model = safe_joblib_load(model_path)
                pva = predict_proba_safe(model, Xva)[:, 1].astype(np.float64)
                auc = float(roc_auc_score(yva, pva)) if np.unique(yva).size >= 2 else float("nan")

                for tgt_fpr in target_fprs:
                    tpr, fpr, thr = eval_tpr_at_fpr(yva, pva, float(tgt_fpr))
                    pred_attack = (pva >= thr)

                    for min_sup in min_sups:
                        recs, min_rec, p10_rec = family_recall_stats(yva, y2va, pred_attack.astype(bool), int(min_sup))
                        bf_rec = float(recs.get("BruteForce", float("nan"))) if isinstance(recs, dict) else float("nan")
                        n_fam = int(len(recs)) if isinstance(recs, dict) else 0

                        for mode in modes:
                            if mode == "min_family_recall":
                                obj = float(min_rec)
                                combo_hash = sha1_obj({"model_hash": model_hash, "tgt_fpr": tgt_fpr, "min_sup": min_sup, "mode": mode})
                                if (model_hash, combo_hash) in seen_combo:
                                    continue

                                row = {
                                    "model_hash": model_hash,
                                    "combo_hash": combo_hash,
                                    "weight_mode": weight_mode,
                                    "use_early_stopping": bool(use_es),
                                    "early_stopping_rounds": int(es_rounds) if bool(use_es) else 0,
                                    "objective_mode": mode,
                                    "p10_quantile": float("nan"),
                                    "min_family_support": int(min_sup),
                                    "target_fpr": float(tgt_fpr),
                                    "objective": float(obj),
                                    "min_family_recall": float(min_rec),
                                    "p10_family_recall": float(p10_rec),
                                    "bruteforce_recall": float(bf_rec),
                                    "n_families_considered": int(n_fam),
                                    "tpr_at_target_fpr": float(tpr),
                                    "fpr_at_target_fpr": float(fpr),
                                    "thr_at_target_fpr": float(thr),
                                    "val_auc": float(auc),
                                    "params": json.dumps(params, sort_keys=True),
                                }
                                rows.append(row)
                                seen_combo.add((model_hash, combo_hash))
                                pd.DataFrame(rows).to_csv(results_path, index=False)

                                ok_bar = float(fpr) <= fpr_bar + 1e-12
                                if ok_bar and (select_mode is None or mode == select_mode):
                                    if (best_combo is None
                                        or obj > best_combo.get("objective", -1)
                                        or (obj == best_combo.get("objective", -1) and bf_rec > best_combo.get("bruteforce_recall", -1))
                                        or (obj == best_combo.get("objective", -1) and bf_rec == best_combo.get("bruteforce_recall", -1) and tpr > best_combo.get("tpr_at_target_fpr", -1))
                                        or (obj == best_combo.get("objective", -1) and bf_rec == best_combo.get("bruteforce_recall", -1) and tpr == best_combo.get("tpr_at_target_fpr", -1) and auc > best_combo.get("val_auc", -1))
                                        or (obj == best_combo.get("objective", -1) and bf_rec == best_combo.get("bruteforce_recall", -1) and tpr == best_combo.get("tpr_at_target_fpr", -1) and auc == best_combo.get("val_auc", -1) and fpr < best_combo.get("fpr_at_target_fpr", 1.0))):
                                        best_combo = {
                                            "hash": model_hash,
                                            "combo_hash": combo_hash,
                                            "objective": float(obj),
                                            "objective_mode": mode,
                                            "p10_quantile": float("nan"),
                                            "min_family_support": int(min_sup),
                                            "target_fpr": float(tgt_fpr),
                                            "min_family_recall": float(min_rec),
                                            "p10_family_recall": float(p10_rec),
                                            "bruteforce_recall": float(bf_rec),
                                            "n_families_considered": int(n_fam),
                                            "tpr_at_target_fpr": float(tpr),
                                            "fpr_at_target_fpr": float(fpr),
                                            "thr_at_target_fpr": float(thr),
                                            "val_auc": float(auc),
                                            "params": params,
                                            "weight_mode": weight_mode,
                                            "use_early_stopping": bool(use_es),
                                            "early_stopping_rounds": int(es_rounds) if bool(use_es) else 0,
                                            "fpr_bar": float(fpr_bar),
                                        }

                                        with open(prog_path, "w", encoding="utf-8") as f:
                                            json.dump({"done_model_hashes": sorted(done_models), "best_combo": best_combo}, f, indent=2)
                            else:
                                for p10_q in p10_qs:
                                    obj = float(np.quantile(list(recs.values()), float(p10_q))) if recs else float(tpr)
                                    combo_hash = sha1_obj({"model_hash": model_hash, "tgt_fpr": tgt_fpr, "min_sup": min_sup, "mode": mode, "p10_q": p10_q})
                                    if (model_hash, combo_hash) in seen_combo:
                                        continue

                                    row = {
                                        "model_hash": model_hash,
                                        "combo_hash": combo_hash,
                                        "weight_mode": weight_mode,
                                        "use_early_stopping": bool(use_es),
                                        "early_stopping_rounds": int(es_rounds) if bool(use_es) else 0,
                                        "objective_mode": mode,
                                        "p10_quantile": float(p10_q),
                                        "min_family_support": int(min_sup),
                                        "target_fpr": float(tgt_fpr),
                                        "objective": float(obj),
                                        "min_family_recall": float(min_rec),
                                        "p10_family_recall": float(p10_rec),
                                        "bruteforce_recall": float(bf_rec),
                                        "n_families_considered": int(n_fam),
                                        "tpr_at_target_fpr": float(tpr),
                                        "fpr_at_target_fpr": float(fpr),
                                        "thr_at_target_fpr": float(thr),
                                        "val_auc": float(auc),
                                        "params": json.dumps(params, sort_keys=True),
                                    }
                                    rows.append(row)
                                    seen_combo.add((model_hash, combo_hash))
                                    pd.DataFrame(rows).to_csv(results_path, index=False)

                                    ok_bar = float(fpr) <= fpr_bar + 1e-12
                                    if ok_bar and (select_mode is None or mode == select_mode):
                                        if (best_combo is None
                                            or obj > best_combo.get("objective", -1)
                                            or (obj == best_combo.get("objective", -1) and bf_rec > best_combo.get("bruteforce_recall", -1))
                                            or (obj == best_combo.get("objective", -1) and bf_rec == best_combo.get("bruteforce_recall", -1) and tpr > best_combo.get("tpr_at_target_fpr", -1))
                                            or (obj == best_combo.get("objective", -1) and bf_rec == best_combo.get("bruteforce_recall", -1) and tpr == best_combo.get("tpr_at_target_fpr", -1) and auc > best_combo.get("val_auc", -1))
                                            or (obj == best_combo.get("objective", -1) and bf_rec == best_combo.get("bruteforce_recall", -1) and tpr == best_combo.get("tpr_at_target_fpr", -1) and auc == best_combo.get("val_auc", -1) and fpr < best_combo.get("fpr_at_target_fpr", 1.0))):
                                            best_combo = {
                                                "hash": model_hash,
                                                "combo_hash": combo_hash,
                                                "objective": float(obj),
                                                "objective_mode": mode,
                                                "p10_quantile": float(p10_q),
                                                "min_family_support": int(min_sup),
                                                "target_fpr": float(tgt_fpr),
                                                "min_family_recall": float(min_rec),
                                                "p10_family_recall": float(p10_rec),
                                                "bruteforce_recall": float(bf_rec),
                                                "n_families_considered": int(n_fam),
                                                "tpr_at_target_fpr": float(tpr),
                                                "fpr_at_target_fpr": float(fpr),
                                                "thr_at_target_fpr": float(thr),
                                                "val_auc": float(auc),
                                                "params": params,
                                                "weight_mode": weight_mode,
                                                "use_early_stopping": bool(use_es),
                                                "early_stopping_rounds": int(es_rounds) if bool(use_es) else 0,
                                                "fpr_bar": float(fpr_bar),
                                            }

                                            with open(prog_path, "w", encoding="utf-8") as f:
                                                json.dump({"done_model_hashes": sorted(done_models), "best_combo": best_combo}, f, indent=2)

    if best_combo is None:
        if not rows:
            raise RuntimeError("Stage-1 sweep produced no result.")
        df = pd.DataFrame(rows)
        df2 = df.sort_values(["objective", "bruteforce_recall", "tpr_at_target_fpr", "val_auc"], ascending=[False, False, False, False])
        best_row = df2.iloc[0].to_dict()
        best_combo = {
            "hash": best_row["model_hash"],
            "combo_hash": best_row["combo_hash"],
            "objective": float(best_row["objective"]),
            "objective_mode": str(best_row["objective_mode"]),
            "p10_quantile": float(best_row.get("p10_quantile", float("nan"))),
            "min_family_support": int(best_row["min_family_support"]),
            "target_fpr": float(best_row["target_fpr"]),
            "min_family_recall": float(best_row["min_family_recall"]),
            "p10_family_recall": float(best_row["p10_family_recall"]),
            "bruteforce_recall": float(best_row["bruteforce_recall"]),
            "n_families_considered": int(best_row["n_families_considered"]),
            "tpr_at_target_fpr": float(best_row["tpr_at_target_fpr"]),
            "fpr_at_target_fpr": float(best_row["fpr_at_target_fpr"]),
            "thr_at_target_fpr": float(best_row["thr_at_target_fpr"]),
            "val_auc": float(best_row["val_auc"]),
            "params": json.loads(best_row["params"]),
            "weight_mode": str(best_row["weight_mode"]),
            "use_early_stopping": bool(best_row["use_early_stopping"]),
            "early_stopping_rounds": int(best_row["early_stopping_rounds"]),
            "fpr_bar": float(fpr_bar),
        }

    with open(os.path.join(run_dir, "stage1_best.json"), "w", encoding="utf-8") as f:
        json.dump(best_combo, f, indent=2)

    best_model = safe_joblib_load(os.path.join(run_dir, cfg["models_dir"], f"{best_combo['hash']}.joblib"))

    # Fit Platt calibration on val probs (binary)
    pva_best = predict_proba_safe(best_model, Xva)[:, 1].astype(np.float64)
    platt = fit_platt_on_probs(pva_best, yva)
    with open(os.path.join(run_dir, "stage1_platt.json"), "w", encoding="utf-8") as f:
        json.dump({"platt": platt}, f, indent=2)

    # Save per-family recall snapshot for best model at chosen thr
    pred_attack = (pva_best >= float(best_combo["thr_at_target_fpr"])).astype(int)
    recs, min_rec, p10_rec = family_recall_stats(yva, y2va, pred_attack.astype(bool), int(best_combo["min_family_support"]))
    with open(os.path.join(run_dir, "stage1_family_recall_at_target_fpr.json"), "w", encoding="utf-8") as f:
        json.dump({
            "threshold": float(best_combo["thr_at_target_fpr"]),
            "target_fpr": float(best_combo["target_fpr"]),
            "fpr_at_target_fpr": float(best_combo["fpr_at_target_fpr"]),
            "tpr_at_target_fpr": float(best_combo["tpr_at_target_fpr"]),
            "objective_mode": str(best_combo["objective_mode"]),
            "p10_quantile": best_combo.get("p10_quantile", None),
            "min_support": int(best_combo["min_family_support"]),
            "recs": recs,
            "min_family_recall": float(min_rec),
            "p10_family_recall": float(p10_rec),
        }, f, indent=2)

    return best_model, platt

def compute_class_weights(y_idx: np.ndarray, K: int) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=K).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / counts
    w = w / np.mean(w)
    return w.astype(np.float64)

def run_stage2_xgb_grid(ds_name: str, ds_dir: str, run_dir: str, prep: Preprocessor) -> Tuple[XGBClassifier, List[str], float]:
    cfg = CFG["stage2_xgb_grid"]
    makedirs(os.path.join(run_dir, cfg["models_dir"]))

    results_path = os.path.join(run_dir, cfg["results_csv"])
    prog_path = os.path.join(run_dir, cfg["progress_json"])

    done = set()
    best = None
    if os.path.exists(prog_path):
        state = json.load(open(prog_path, "r", encoding="utf-8"))
        done = set(state.get("done_hashes", []))
        best = state.get("best", None)

    parts_tr = list_parts(ds_dir, "train")
    parts_va = list_parts(ds_dir, "val")
    if not parts_tr or not parts_va:
        raise RuntimeError("Missing train/val parts.")

    max_tr = int(cfg["max_train_rows"].get(ds_name, 900_000))
    max_va = int(cfg["max_val_rows"].get(ds_name, 450_000))

    Xtr, _, y2tr = collect_xy(parts_tr, prep, CFG["y_stage1"], max_tr, seed=cfg["seed"], filter_attack=True, y2_col=CFG["y_stage2"])
    Xva, _, y2va = collect_xy(parts_va, prep, CFG["y_stage1"], max_va, seed=cfg["seed"] + 1, filter_attack=True, y2_col=CFG["y_stage2"])

    if y2tr is None or len(y2tr) == 0:
        raise RuntimeError("No attack samples found for stage-2 training.")
    if y2va is None or len(y2va) == 0:
        raise RuntimeError("No attack samples found for stage-2 validation.")

    # LOAO (Protocol B-1): remove holdout-family rows from Stage-2 TRAINING only.
    lo = CFG.get("loao", {}) or {}
    holdout = None
    if bool(lo.get("enabled", False)):
        hf = lo.get("holdout_family", None)
        if isinstance(hf, dict):
            holdout = hf.get(ds_name, None)
        elif isinstance(hf, str):
            holdout = hf
    holdout = str(holdout) if holdout else ""
    if holdout:
        y2tr_str = np.array([str(x) for x in y2tr], dtype=object)
        mask_keep = (y2tr_str != holdout)
        removed = int((~mask_keep).sum())
        if removed > 0 and mask_keep.sum() > 0:
            Xtr = Xtr[mask_keep]
            y2tr = y2tr[mask_keep]
        with open(os.path.join(run_dir, "loao_stage2_holdout.json"), "w", encoding="utf-8") as f:
            json.dump({"enabled": True, "holdout_family": holdout, "removed_train_rows": removed}, f, indent=2)
    else:
        with open(os.path.join(run_dir, "loao_stage2_holdout.json"), "w", encoding="utf-8") as f:
            json.dump({"enabled": bool(lo.get("enabled", False)), "holdout_family": None, "removed_train_rows": 0}, f, indent=2)


    families = sorted({str(x) for x in y2tr if str(x) and str(x).lower() != "nan"})
    fam_to_idx = {f: i for i, f in enumerate(families)}
    K = len(families)
    if K < 2:
        raise RuntimeError(f"Stage-2 needs >=2 families, found {K}.")

    ytr = np.array([fam_to_idx[str(x)] for x in y2tr], dtype=int)

    mask_val_ok = np.array([str(x) in fam_to_idx for x in y2va], dtype=bool)
    Xva_ok = Xva[mask_val_ok]
    yva = np.array([fam_to_idx[str(x)] for x in y2va[mask_val_ok]], dtype=int)

    w = compute_class_weights(ytr, K)
    sample_weight = w[ytr]

    rows = []
    if os.path.exists(results_path):
        try:
            rows = pd.read_csv(results_path).to_dict("records")
        except Exception:
            rows = []

    labels_fixed = list(range(K))

    for g in cfg["grid"]:
        params = dict(g)
        params.update({
            "tree_method": cfg.get("tree_method", "hist"),
            "n_jobs": cfg.get("n_jobs", 0),
            "verbosity": cfg.get("verbosity", 0),
            "eval_metric": cfg.get("eval_metric", "mlogloss"),
        })
        if cfg.get("device"):
            params["device"] = cfg.get("device")
        if cfg.get("predictor"):
            params["predictor"] = cfg.get("predictor")
        h = sha1_obj({"K": K, **params})
        if h in done:
            continue

        model = build_xgb_multi(params, num_class=K)
        model.fit(Xtr, ytr, sample_weight=sample_weight)

        pva = model.predict_proba(Xva_ok).astype(np.float64)
        pva = np.clip(pva, 1e-12, 1.0)
        pva = pva / pva.sum(axis=1, keepdims=True)

        pred = np.argmax(pva, axis=1).astype(int)

        # If validation has no attack rows belonging to trained families, avoid metrics that require >=1 sample
        if yva.size == 0:
            mf1_fixed = float("nan")
            mf1_pres = float("nan")
            acc = float("nan")
            nll = float("nan")
        else:
            mf1_fixed = macro_f1_fixedK(yva, pred, K)
            mf1_pres = macro_f1_present(yva, pred)
            acc = float(accuracy_score(yva, pred))
            nll = float(log_loss(yva, pva, labels=labels_fixed))  # Fix 1: explicit labels

        row = {
            "hash": h,
            "val_macro_f1_fixedK": float(mf1_fixed),
            "val_macro_f1_present": float(mf1_pres),
            "val_accuracy": float(acc),
            "val_nll": float(nll),
            "n_val_attacks": int(len(yva)),
            "missing_classes_in_val": json.dumps(missing_labels(yva, K)),
            "params": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_path, index=False)

        safe_joblib_dump(model, os.path.join(run_dir, cfg["models_dir"], f"{h}.joblib"))

        done.add(h)

        # Selection: maximize macro_f1_present (honest split), tie-break by nll then fixedK
        if (best is None
            or mf1_pres > best.get("val_macro_f1_present", -1)
            or (mf1_pres == best.get("val_macro_f1_present", -1) and nll < best.get("val_nll", 1e18))
            or (mf1_pres == best.get("val_macro_f1_present", -1) and nll == best.get("val_nll", 1e18) and mf1_fixed > best.get("val_macro_f1_fixedK", -1))):
            best = {
                "hash": h,
                "val_macro_f1_present": float(mf1_pres),
                "val_macro_f1_fixedK": float(mf1_fixed),
                "val_accuracy": float(acc),
                "val_nll": float(nll),
                "params": params,
                "K": int(K),
            }

        with open(prog_path, "w", encoding="utf-8") as f:
            json.dump({"done_hashes": sorted(done), "best": best}, f, indent=2)

    if best is None:
        raise RuntimeError("Stage-2 grid produced no result.")

    best_model = safe_joblib_load(os.path.join(run_dir, cfg["models_dir"], f"{best['hash']}.joblib"))

    # Temperature scaling on val (attacks only)
    pva_best = best_model.predict_proba(Xva_ok).astype(np.float64)
    T = fit_temperature_on_probs(pva_best, yva, K)

    with open(os.path.join(run_dir, "stage2_temperature.json"), "w", encoding="utf-8") as f:
        json.dump({"T": float(T), "hash": best["hash"], "params": best["params"], "K": int(K)}, f, indent=2)

    with open(os.path.join(run_dir, "stage2_best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    return best_model, families, float(T)


# ============================================================
# Joint tuning (thr_low, tau) + tau strict
# ============================================================

def eval_system_on_val(y_sys_true: np.ndarray,
                       p_attack: np.ndarray,
                       fam_pred_idx: np.ndarray,
                       fam_pmax: np.ndarray,
                       families: List[str],
                       thr: float,
                       tau: float,
                       abstain_label: str,
                       policy: str) -> Dict[str, float]:
    cache = build_tuning_eval_cache(y_sys_true, families, abstain_label, policy)
    return eval_system_on_val_cached(cache, p_attack, fam_pred_idx, fam_pmax, thr, tau)

def pick_tau_strict(run_dir: str,
                    y_sys_true: np.ndarray,
                    p_attack: np.ndarray,
                    fam_pred_idx: np.ndarray,
                    fam_pmax: np.ndarray,
                    families: List[str],
                    thr_high: float) -> float:
    ab = CFG["abstain"]
    if not ab.get("enabled", True):
        return 0.0

    abstain_label = ab["label"]
    policy = ab.get("policy", "reject_to_benign")

    # LOAO / open-set tuning mode: prioritize Unknown detection under false-unknown constraints.
    lo = CFG.get("loao", {}) or {}
    do_open_tune = bool(lo.get("enabled", False)) and (policy == "abstain") and bool(lo.get("optimize_tau_for_unknown", True))
    max_fur_all = float(lo.get("max_false_unknown_rate_all", 0.05))
    max_fur_att = float(lo.get("max_false_unknown_rate_known_attacks", 0.10))

    base_max_bfpr = float(ab.get("max_benign_family_fp_rate", 0.02))
    base_max_br = float(ab.get("max_benign_reject_rate", 0.10))
    base_max_or = float(ab.get("max_overall_reject_rate", 0.01))

    # Sweep constraints (Option D)
    if ab.get("sweep_enabled", False):
        max_bfprs = [float(x) for x in ab.get("max_benign_family_fp_rate_list", [base_max_bfpr])]
        max_ors = [float(x) for x in ab.get("max_overall_reject_rate_list", [base_max_or])]
    else:
        max_bfprs, max_ors = [base_max_bfpr], [base_max_or]

    taus = list(ab.get("grid", [0.0, 0.5, 0.7, 0.8, 0.9]))
    if ab.get("also_try_quantiles", True):
        for q in ab.get("quantiles", [0.05, 0.1, 0.2]):
            taus.append(float(np.quantile(fam_pmax, q)))
    taus = sorted({float(np.clip(t, 0.0, 0.999)) for t in taus})

    rows = []
    best = None
    eval_cache = build_tuning_eval_cache(y_sys_true, families, abstain_label, policy)
    total_candidates = len(max_bfprs) * len(max_ors) * len(taus)
    completed_candidates = 0
    progress_print(
        f"[tune] strict tau sweep candidates={total_candidates} rows={len(y_sys_true)}",
    )

    for max_bfpr in max_bfprs:
        for max_or in max_ors:
            best_local = None

            for tau in taus:
                met = eval_system_on_val_cached(eval_cache, p_attack, fam_pred_idx, fam_pmax, thr_high, float(tau))
                completed_candidates += 1
                if completed_candidates % 25 == 0 or completed_candidates == total_candidates:
                    progress_print(
                        f"[tune] strict tau sweep {completed_candidates}/{total_candidates}",
                    )

                def _getf(d, k, default=0.0):
                    v = d.get(k, default)
                    if v is None:
                        return float(default)
                    try:
                        return float(v)
                    except Exception:
                        return float(default)

                ok = (_getf(met, "benign_family_fp_rate", 1.0) <= max_bfpr + 1e-12) and (_getf(met, "benign_reject_rate", 1.0) <= base_max_br + 1e-12) and (_getf(met, "overall_reject_rate", 1.0) <= max_or + 1e-12)

                # If Unknown exists in validation and open-tune is enabled, enforce false-unknown constraints too.
                if do_open_tune and (met.get("unknown_detection_rate", None) is not None) and (met.get("n_true_unknown", 0) not in (None, 0)):
                    ok = ok and (_getf(met, "false_unknown_rate_all_known", 1.0) <= max_fur_all + 1e-12) and (_getf(met, "false_unknown_rate_known_attacks", 1.0) <= max_fur_att + 1e-12)

                row = {"max_benign_family_fp_rate": float(max_bfpr), "max_overall_reject_rate": float(max_or), "ok": bool(ok), "tau": float(tau), **met}
                rows.append(row)

                if not ok:
                    continue

                # Selection objective
                if do_open_tune and (met.get("unknown_detection_rate", None) is not None) and (met.get("n_true_unknown", 0) not in (None, 0)):
                    if (best_local is None
                        or _getf(met, "unknown_detection_rate", -1) > _getf(best_local, "unknown_detection_rate", -1)
                        or (_getf(met, "unknown_detection_rate", -1) == _getf(best_local, "unknown_detection_rate", -1) and _getf(met, "macro_f1", -1) > _getf(best_local, "macro_f1", -1))
                        or (_getf(met, "unknown_detection_rate", -1) == _getf(best_local, "unknown_detection_rate", -1) and _getf(met, "macro_f1", -1) == _getf(best_local, "macro_f1", -1) and _getf(met, "false_unknown_rate_all_known", 1.0) < _getf(best_local, "false_unknown_rate_all_known", 1.0))):
                        best_local = dict(row)
                else:
                    if (best_local is None) or (_getf(met, "macro_f1", -1) > _getf(best_local, "macro_f1", -1)):
                        best_local = dict(row)

            if best_local is not None:
                if (best is None
                    or float(best_local.get("macro_f1", -1)) > float(best.get("macro_f1", -1))
                    or (float(best_local.get("macro_f1", -1)) == float(best.get("macro_f1", -1)) and float(best_local.get("benign_family_fp_rate", 1.0)) < float(best.get("benign_family_fp_rate", 1.0)))
                    or (float(best_local.get("macro_f1", -1)) == float(best.get("macro_f1", -1)) and float(best_local.get("benign_family_fp_rate", 1.0)) == float(best.get("benign_family_fp_rate", 1.0)) and float(best_local.get("overall_reject_rate", 1.0)) < float(best.get("overall_reject_rate", 1.0)))):
                    best = dict(best_local)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "val_sweep_tau_strict.csv"), index=False)

    if best is None:
        df2 = df.sort_values(["benign_family_fp_rate", "macro_f1"], ascending=[True, False])
        best = df2.iloc[0].to_dict()

    with open(os.path.join(run_dir, "abstain_best_strict.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    return float(best["tau"])

def tune_cascade_thr_low_and_tau(run_dir: str,
                                 y1_val: np.ndarray,
                                 y_sys_true: np.ndarray,
                                 p_attack: np.ndarray,
                                 fam_pred_idx: np.ndarray,
                                 fam_pmax: np.ndarray,
                                 families: List[str],
                                 thr_high: float) -> Tuple[float, float]:
    cg = CFG["cascade_gate"]
    ab = CFG["abstain"]
    if not cg.get("enabled", True):
        return float(thr_high), 0.0

    abstain_label = ab["label"]
    policy = ab.get("policy", "reject_to_benign")

    # LOAO / open-set tuning mode: prioritize Unknown detection under false-unknown constraints.
    lo = CFG.get("loao", {}) or {}
    do_open_tune = bool(lo.get("enabled", False)) and (policy == "abstain") and bool(lo.get("optimize_tau_for_unknown", True))
    max_fur_all = float(lo.get("max_false_unknown_rate_all", 0.05))
    max_fur_att = float(lo.get("max_false_unknown_rate_known_attacks", 0.10))

    base_max_bfpr = float(ab.get("max_benign_family_fp_rate", 0.02))
    base_max_br = float(ab.get("max_benign_reject_rate", 0.10))
    base_max_or = float(ab.get("max_overall_reject_rate", 0.01))

    # Sweep constraints (Option D)
    if ab.get("sweep_enabled", False):
        max_bfprs = [float(x) for x in ab.get("max_benign_family_fp_rate_list", [base_max_bfpr])]
        max_ors = [float(x) for x in ab.get("max_overall_reject_rate_list", [base_max_or])]
    else:
        max_bfprs, max_ors = [base_max_bfpr], [base_max_or]

    benign_p = p_attack[y1_val == 0]
    if len(benign_p) == 0:
        return float(min(0.05, thr_high)), 0.0

    grid_points = max(8, int(cg.get("grid_points", 30)))
    qs = np.linspace(0.50, 0.999, grid_points)
    thr_candidates = sorted({float(min(np.quantile(benign_p, q), thr_high)) for q in qs} | {float(thr_high)})

    taus = list(ab.get("grid", [0.0, 0.5, 0.7, 0.8, 0.9]))
    if ab.get("also_try_quantiles", True):
        for q in ab.get("quantiles", [0.05, 0.1, 0.2]):
            taus.append(float(np.quantile(fam_pmax, q)))
    taus = sorted({float(np.clip(t, 0.0, 0.999)) for t in taus})

    rows, best = [], None
    eval_cache = build_tuning_eval_cache(y_sys_true, families, abstain_label, policy)
    total_candidates = len(max_bfprs) * len(max_ors) * len(thr_candidates) * len(taus)
    completed_candidates = 0
    progress_print(
        f"[tune] cascade sweep candidates={total_candidates} rows={len(y_sys_true)} "
        f"thresholds={len(thr_candidates)} taus={len(taus)}",
    )

    for max_bfpr in max_bfprs:
        for max_or in max_ors:
            best_local = None
            for thr_low in thr_candidates:
                for tau in taus:
                    met = eval_system_on_val_cached(eval_cache, p_attack, fam_pred_idx, fam_pmax, float(thr_low), float(tau))
                    completed_candidates += 1
                    if completed_candidates % 50 == 0 or completed_candidates == total_candidates:
                        progress_print(
                            f"[tune] cascade sweep {completed_candidates}/{total_candidates}",
                        )
                    def _getf(d, k, default=0.0):
                        v = d.get(k, default)
                        if v is None:
                            return float(default)
                        try:
                            return float(v)
                        except Exception:
                            return float(default)

                    ok = (_getf(met, "benign_family_fp_rate", 1.0) <= max_bfpr + 1e-12) and (_getf(met, "benign_reject_rate", 1.0) <= base_max_br + 1e-12) and (_getf(met, "overall_reject_rate", 1.0) <= max_or + 1e-12)
                    if do_open_tune and (met.get("unknown_detection_rate", None) is not None) and (met.get("n_true_unknown", 0) not in (None, 0)):
                        ok = ok and (_getf(met, "false_unknown_rate_all_known", 1.0) <= max_fur_all + 1e-12) and (_getf(met, "false_unknown_rate_known_attacks", 1.0) <= max_fur_att + 1e-12)

                    row = {"max_benign_family_fp_rate": float(max_bfpr), "max_overall_reject_rate": float(max_or), "ok": bool(ok),
                           "thr_low": float(thr_low), "tau": float(tau), **met}
                    rows.append(row)

                    if not ok:
                        continue

                    if do_open_tune and (met.get("unknown_detection_rate", None) is not None) and (met.get("n_true_unknown", 0) not in (None, 0)):
                        if (best_local is None
                            or _getf(met, "unknown_detection_rate", -1) > _getf(best_local, "unknown_detection_rate", -1)
                            or (_getf(met, "unknown_detection_rate", -1) == _getf(best_local, "unknown_detection_rate", -1) and _getf(met, "macro_f1", -1) > _getf(best_local, "macro_f1", -1))
                            or (_getf(met, "unknown_detection_rate", -1) == _getf(best_local, "unknown_detection_rate", -1) and _getf(met, "macro_f1", -1) == _getf(best_local, "macro_f1", -1) and _getf(met, "false_unknown_rate_all_known", 1.0) < _getf(best_local, "false_unknown_rate_all_known", 1.0))):
                            best_local = dict(row)
                    else:
                        if (best_local is None) or (float(met["macro_f1"]) > float(best_local.get("macro_f1", -1))):
                            best_local = dict(row)

            if best_local is not None:
                if best is None:
                    best = dict(best_local)
                else:
                    cond1 = _getf(best_local, "macro_f1", -1) > _getf(best, "macro_f1", -1)
                    cond2 = (_getf(best_local, "macro_f1", -1) == _getf(best, "macro_f1", -1)
                             and _getf(best_local, "benign_family_fp_rate", 1.0) < _getf(best, "benign_family_fp_rate", 1.0))
                    cond3 = (_getf(best_local, "macro_f1", -1) == _getf(best, "macro_f1", -1)
                             and _getf(best_local, "benign_family_fp_rate", 1.0) == _getf(best, "benign_family_fp_rate", 1.0)
                             and _getf(best_local, "overall_reject_rate", 1.0) < _getf(best, "overall_reject_rate", 1.0))
                    if cond1 or cond2 or cond3:
                        best = dict(best_local)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "val_sweep_thr_low_tau_joint.csv"), index=False)

    if best is None:
        df2 = df.sort_values(["benign_family_fp_rate", "macro_f1"], ascending=[True, False])
        best = df2.iloc[0].to_dict()

    with open(os.path.join(run_dir, "cascade_best_thr_low_tau.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    return float(best["thr_low"]), float(best["tau"])

def evaluate_system(ds_dir: str,
                    run_dir: str,
                    split: str,
                    prep: Preprocessor,
                    stage1: XGBClassifier,
                    platt: Optional[Dict[str, float]],
                    stage2: XGBClassifier,
                    families: List[str],
                    T: float,
                    thr_high: float,
                    thr_low: float,
                    tau_strict: float,
                    tau_cascade: float):
    parts = list_parts(ds_dir, split)
    if not parts:
        raise RuntimeError(f"No parts for {split}")

    ab = CFG["abstain"]
    policy = ab.get("policy", "reject_to_benign")
    abstain_label = ab["label"]

    loao_enabled = bool((CFG.get("loao", {}) or {}).get("enabled", False))

    fam_to_idx = {f: i for i, f in enumerate(families)}
    K = len(families)

    y1_all, p_all = [], []
    y_sys_true = []
    pred_strict, pred_cascade = [], []
    pred_strict_tau, pred_cascade_tau = [], []
    rej_strict_tau, rej_cascade_tau = [], []

    # System-level calibration accumulators (known classes only, accepted samples only; strict_tau gate)
    sys_cal_conf = []
    sys_cal_correct = []
    sys_brier_sum = 0.0
    sys_brier_n = 0

    y2_true_idx, y2_pred_idx = [], []

    usecols = prep.num_cols + prep.cat_cols + [CFG["y_stage1"], CFG["y_stage2"]]
    usecols = [canonical_col(c) for c in usecols]

    for chunk in iter_rows_from_parts(parts, usecols=usecols, chunksize=CFG["chunksize_rows"]):
        chunk.columns = [canonical_col(c) for c in chunk.columns]
        y1 = chunk[CFG["y_stage1"]].astype(int).to_numpy()
        y2 = chunk[CFG["y_stage2"]].astype(str).fillna("").to_numpy()

        # System-level ground truth:
        # - Benign -> "Benign"
        # - Attack family in trained families -> that family
        # - Attack family NOT in trained families (e.g., LOAO holdout) -> Unknown
        if loao_enabled and (policy == "abstain"):
            unk = abstain_label
            ysys = np.array(["Benign" if a == 0 else (b if str(b) in fam_to_idx else unk) for a, b in zip(y1, y2)], dtype=object)
        else:
            ysys = np.array(["Benign" if a == 0 else b for a, b in zip(y1, y2)], dtype=object)

        X = prep.transform(chunk.drop(columns=[CFG["y_stage1"], CFG["y_stage2"]], errors="ignore"))

        p_attack = stage1.predict_proba(X)[:, 1].astype(np.float64)
        p_attack = apply_platt(p_attack, platt)

        p2 = stage2.predict_proba(X).astype(np.float64)
        p2 = apply_temperature(p2, T)

        fam_pred = np.argmax(p2, axis=1).astype(int)
        fam_pmax = np.max(p2, axis=1).astype(np.float64)

        # Strict/cascade without rejection (tau=0) — always map abstain to benign for these baselines
        predS_raw = system_predict_raw(p_attack, fam_pred, fam_pmax, families, thr=thr_high, tau=0.0, abstain_label=abstain_label)
        predC_raw = system_predict_raw(p_attack, fam_pred, fam_pmax, families, thr=thr_low,  tau=0.0, abstain_label=abstain_label)
        predS, _ = apply_abstain_policy(predS_raw, abstain_label, policy="reject_to_benign")
        predC, _ = apply_abstain_policy(predC_raw, abstain_label, policy="reject_to_benign")

        # With tau (potential Unknown)
        predS_tau_raw = system_predict_raw(p_attack, fam_pred, fam_pmax, families, thr=thr_high, tau=tau_strict,  abstain_label=abstain_label)
        predC_tau_raw = system_predict_raw(p_attack, fam_pred, fam_pmax, families, thr=thr_low,  tau=tau_cascade, abstain_label=abstain_label)
        predS_tau, rejS = apply_abstain_policy(predS_tau_raw, abstain_label, policy=policy)
        predC_tau, rejC = apply_abstain_policy(predC_tau_raw, abstain_label, policy=policy)

        # Accumulate for stage-1 calibration
        y1_all.append(y1)
        p_all.append(p_attack)

        # System labels
        y_sys_true.extend(list(ysys))
        pred_strict.extend(list(predS))
        pred_cascade.extend(list(predC))
        pred_strict_tau.extend(list(predS_tau))
        pred_cascade_tau.extend(list(predC_tau))
        rej_strict_tau.extend(list(rejS))
        rej_cascade_tau.extend(list(rejC))

        # Stage-2 report (attacks only, known families only)
        mask_attack = (y1 == 1)
        if mask_attack.any():
            y2t = y2[mask_attack]
            idx_true = np.array([fam_to_idx.get(str(f), -1) for f in y2t], dtype=int)
            ok = idx_true >= 0
            if ok.any():
                y2_true_idx.extend(list(idx_true[ok]))
                y2_pred_idx.extend(list(fam_pred[mask_attack][ok]))

        # System-level calibration on known classes (Benign + trained families), accepted samples only.
        calib_cfg = CFG.get("calibration_reporting", {}) or {}
        if bool(calib_cfg.get("enabled", True)):
            idx_true_fam = np.array([fam_to_idx.get(str(f), -1) for f in y2], dtype=int)
            true_idx = np.full(len(y1), -1, dtype=int)
            true_idx[y1 == 0] = 0
            mk = (y1 == 1) & (idx_true_fam >= 0)
            true_idx[mk] = 1 + idx_true_fam[mk]

            accept = (~rejS) if (policy == "abstain") else np.ones(len(y1), dtype=bool)
            use = (true_idx >= 0) & accept
            if np.any(use):
                pb = 1.0 - p_attack
                pf_max = p_attack * fam_pmax
                conf = np.where(pb >= pf_max, pb, pf_max)
                pred_idx = np.where(pb >= pf_max, 0, 1 + fam_pred)

                # Brier (known-only multiclass): sum_k p_k^2 + 1 - 2*p_true
                sum_p2_sq = np.sum(p2 * p2, axis=1)
                sumsq = pb * pb + (p_attack * p_attack) * sum_p2_sq

                p_true = np.zeros(len(y1), dtype=np.float64)
                p_true[true_idx == 0] = pb[true_idx == 0]
                fam_rows = (true_idx > 0)
                if np.any(fam_rows):
                    kk = (true_idx[fam_rows] - 1).astype(int)
                    p_true[fam_rows] = p_attack[fam_rows] * p2[fam_rows, kk]

                brier_samp = sumsq + 1.0 - 2.0 * p_true
                sys_brier_sum += float(np.sum(brier_samp[use]))
                sys_brier_n += int(np.sum(use))

                sys_cal_conf.extend(list(conf[use].astype(np.float64)))
                sys_cal_correct.extend(list((pred_idx[use] == true_idx[use]).astype(np.float64)))

    y1_all = np.concatenate(y1_all) if y1_all else np.array([], dtype=int)
    p_all = np.concatenate(p_all) if p_all else np.array([], dtype=float)
    y_sys_true = np.array(y_sys_true, dtype=object)

    pred_strict = np.array(pred_strict, dtype=object)
    pred_cascade = np.array(pred_cascade, dtype=object)
    pred_strict_tau = np.array(pred_strict_tau, dtype=object)
    pred_cascade_tau = np.array(pred_cascade_tau, dtype=object)

    rej_strict_tau = np.array(rej_strict_tau, dtype=bool)
    rej_cascade_tau = np.array(rej_cascade_tau, dtype=bool)

    # Stage-1 metrics at thr_high
    y1_pred = (p_all >= thr_high).astype(int)
    tn = int(((y1_all == 0) & (y1_pred == 0)).sum())
    fp = int(((y1_all == 0) & (y1_pred == 1)).sum())
    fn = int(((y1_all == 1) & (y1_pred == 0)).sum())
    tp = int(((y1_all == 1) & (y1_pred == 1)).sum())
    fpr = fp / max(1, (fp + tn))
    tpr = tp / max(1, (tp + fn))
    auc = float(roc_auc_score(y1_all, p_all)) if np.unique(y1_all).size >= 2 else float("nan")

    with open(os.path.join(run_dir, f"metrics_stage1_{split}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "roc_auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "threshold": float(thr_high),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }, f, indent=2)

    # Calibration reporting (Stage-1)
    calib_cfg = CFG.get("calibration_reporting", {}) or {}
    if bool(calib_cfg.get("enabled", True)) and len(y1_all) > 0:
        n_bins = int(calib_cfg.get("ece_bins", 15))
        dpi = int(calib_cfg.get("plot_dpi", 160))
        brier = brier_binary(y1_all, p_all)
        ece, bin_stats = ece_binary(y1_all, p_all, n_bins=n_bins)

        with open(os.path.join(run_dir, f"calibration_stage1_{split}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "brier": float(brier),
                "ece": float(ece),
                "n_bins": int(n_bins),
                "n": int(len(y1_all)),
                "bins": bin_stats,
            }, f, indent=2)

        plot_reliability_curve(
            bin_stats,
            out_png=os.path.join(run_dir, f"reliability_stage1_{split}.png"),
            title=f"Stage-1 Reliability ({split})",
            dpi=dpi
        )

    use_abstain_label = (policy == "abstain")
    sys_lbls = system_labels(families, use_abstain_label, abstain_label)

    def dump_sys(name: str, pred: np.ndarray, reject: Optional[np.ndarray] = None) -> Dict[str, object]:
        acc = float(accuracy_score(y_sys_true, pred))
        declared_mf1 = macro_f1_fixed(y_sys_true, pred, labels=sys_lbls)
        supported_labels = ["Benign", *families]
        if str(CFG.get("protocol", "A")).upper().startswith("B"):
            true_unknown_support = int(np.sum(y_sys_true == abstain_label))
            if true_unknown_support <= 0:
                raise ValueError("Protocol B system macro-F1 requires genuine true-Unknown support")
            supported_labels = sys_lbls
        supported_mf1 = macro_f1_fixed(y_sys_true, pred, labels=supported_labels)
        bfpr = benign_family_fp_rate(y_sys_true, pred, families)
        br = 0.0 if reject is None else benign_reject_rate(y_sys_true, reject)
        orr = 0.0 if reject is None else overall_reject_rate(reject)

        payload = {
            "accuracy": acc,
            "macro_f1": supported_mf1,
            "system_macro_f1_supported_labels": supported_mf1,
            "system_macro_f1_declared_output_labels_historical": declared_mf1,
            "system_macro_f1_averaged_labels": supported_labels,
            "benign_family_fp_rate": bfpr,
            "benign_reject_rate": br,
            "overall_reject_rate": orr,
            "policy": policy,
        }

        if use_abstain_label:
            unk = abstain_label
            true_unk = (y_sys_true == unk)
            pred_unk = (pred == unk)
            n_true_unknown = int(true_unk.sum())
            n_known_all = int((~true_unk).sum())
            n_known_attacks = int(((y_sys_true != "Benign") & (~true_unk)).sum())

            payload.update({
                "unknown_detection_rate": float(np.mean(pred_unk[true_unk])) if n_true_unknown > 0 else None,
                "false_unknown_rate_all_known": float(np.mean(pred_unk[~true_unk])) if n_known_all > 0 else None,
                "false_unknown_rate_known_attacks": float(np.mean(pred_unk[(y_sys_true != "Benign") & (~true_unk)])) if n_known_attacks > 0 else None,
                "n_true_unknown": n_true_unknown,
                "n_known_all": n_known_all,
                "n_known_attacks": n_known_attacks,
            })

        with open(os.path.join(run_dir, f"metrics_system_{name}_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        cm = confusion_matrix(y_sys_true, pred, labels=sys_lbls)
        pd.DataFrame(cm, index=sys_lbls, columns=sys_lbls).to_csv(os.path.join(run_dir, f"confusion_matrix_system_{name}_{split}.csv"))

        return payload

    out = {
        "strict": dump_sys("strict", pred_strict, None),
        "cascade": dump_sys("cascade", pred_cascade, None),
        "strict_tau": dump_sys("strict_tau", pred_strict_tau, rej_strict_tau),
        "cascade_tau": dump_sys("cascade_tau", pred_cascade_tau, rej_cascade_tau),
    }

    with open(os.path.join(run_dir, f"system_compare_{split}.json"), "w", encoding="utf-8") as f:
        json.dump({"policy": policy, **out}, f, indent=2)

    # Calibration reporting (System-level, known classes only; accepted samples only)
    if bool(calib_cfg.get("enabled", True)) and sys_brier_n > 0 and len(sys_cal_conf) > 0:
        n_bins = int(calib_cfg.get("ece_bins", 15))
        dpi = int(calib_cfg.get("plot_dpi", 160))
        conf_arr = np.array(sys_cal_conf, dtype=np.float64)
        corr_arr = np.array(sys_cal_correct, dtype=np.float64)
        ece_sys, bin_stats_sys = ece_toplabel(conf_arr, corr_arr, n_bins=n_bins)
        brier_sys = float(sys_brier_sum / max(1, sys_brier_n))

        with open(os.path.join(run_dir, f"calibration_system_known_strict_tau_{split}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "brier_multiclass_known": float(brier_sys),
                "ece_toplabel_known": float(ece_sys),
                "n_bins": int(n_bins),
                "n_accepted_known": int(sys_brier_n),
                "bins": bin_stats_sys,
                "note": "Known classes only (Benign + trained families). Acceptance gate = strict_tau (non-rejected).",
            }, f, indent=2)

        plot_reliability_curve(
            bin_stats_sys,
            out_png=os.path.join(run_dir, f"reliability_system_known_strict_tau_{split}.png"),
            title=f"System Reliability (Known, strict_tau, {split})",
            dpi=dpi
        )

    # Stage-2 report (attacks only) — Fix 1
    if y2_true_idx:
        y2_true_idx = np.array(y2_true_idx, dtype=int)
        y2_pred_idx = np.array(y2_pred_idx, dtype=int)

        mf1_fixed = macro_f1_fixedK(y2_true_idx, y2_pred_idx, K)
        mf1_pres = macro_f1_present(y2_true_idx, y2_pred_idx)
        acc2 = float(accuracy_score(y2_true_idx, y2_pred_idx))

        missing = missing_labels(y2_true_idx, K)
        missing_families = [families[i] for i in missing]

        rep_present = classification_report(
            y2_true_idx, y2_pred_idx,
            labels=present_labels(y2_true_idx),
            output_dict=True, zero_division=0
        )
        pd.DataFrame(rep_present).to_csv(os.path.join(run_dir, f"classification_report_stage2_present_{split}.csv"))

        rep_fixed = classification_report(
            y2_true_idx, y2_pred_idx,
            labels=list(range(K)),
            output_dict=True, zero_division=0
        )
        pd.DataFrame(rep_fixed).to_csv(os.path.join(run_dir, f"classification_report_stage2_fixedK_{split}.csv"))

        cm2 = confusion_matrix(y2_true_idx, y2_pred_idx, labels=list(range(K)))
        pd.DataFrame(cm2, index=families, columns=families).to_csv(os.path.join(run_dir, f"confusion_matrix_stage2_fixedK_{split}.csv"))

        with open(os.path.join(run_dir, f"metrics_stage2_{split}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "macro_f1_fixedK": float(mf1_fixed),
                "macro_f1_present": float(mf1_pres),
                "accuracy": float(acc2),
                "n_attacks": int(len(y2_true_idx)),
                "missing_class_indices": missing,
                "missing_families": missing_families,
            }, f, indent=2)

def main():
    ensure_deps()
    makedirs(CFG["runs_root"])

    for ds_name in CFG["datasets"]:
        ds_dir = os.path.join(CFG["processed_root"], CFG["protocol"], ds_name)
        if not os.path.isdir(ds_dir):
            raise RuntimeError(f"Dataset folder not found: {ds_dir}")

        resume = CFG["resume_run_dirs"].get(ds_name, None)
        if resume:
            run_dir = resume
            makedirs(run_dir)
            progress_print(f"[resume] {ds_name}: {run_dir}")
        else:
            run_dir = os.path.join(CFG["runs_root"], f"{CFG['protocol']}_{ds_name}_{now_stamp()}")
            makedirs(run_dir)

        progress_print(f"\n=== Dataset: {ds_name} ===\nrun_dir={run_dir}")

        # Preprocessor
        prep_path = os.path.join(run_dir, "preprocessor.joblib")
        if os.path.exists(prep_path):
            prep = safe_joblib_load(prep_path)
        else:
            prep = fit_preprocessor(ds_dir, run_dir)

        # Stage-1
        stage1_path = os.path.join(run_dir, "stage1_best.joblib")
        platt_path = os.path.join(run_dir, "stage1_platt.json")
        if os.path.exists(stage1_path):
            stage1 = safe_joblib_load(stage1_path)
            platt = None
            if os.path.exists(platt_path):
                try:
                    platt = json.load(open(platt_path, "r", encoding="utf-8")).get("platt", None)
                except Exception:
                    platt = None
        else:
            stage1, platt = run_stage1_xgb_grid(ds_name, ds_dir, run_dir, prep)
            safe_joblib_dump(stage1, stage1_path)

        # Stage-2
        stage2_path = os.path.join(run_dir, "stage2_best.joblib")
        fam_path = os.path.join(run_dir, "families.json")
        temp_path = os.path.join(run_dir, "stage2_temperature.json")
        if os.path.exists(stage2_path) and os.path.exists(temp_path) and os.path.exists(fam_path):
            stage2 = safe_joblib_load(stage2_path)
            T = float(json.load(open(temp_path, "r", encoding="utf-8"))["T"])
            families = json.load(open(fam_path, "r", encoding="utf-8"))
        else:
            stage2, families, T = run_stage2_xgb_grid(ds_name, ds_dir, run_dir, prep)
            safe_joblib_dump(stage2, stage2_path)
            with open(fam_path, "w", encoding="utf-8") as f:
                json.dump(families, f, indent=2)

        # Build validation arrays for system-level tuning + thr_high selection
        parts_val = list_parts(ds_dir, "val")
        if not parts_val:
            raise RuntimeError("No val parts found.")

        max_val = int(min(CFG["cascade_gate"]["max_val_rows_for_sweep"], CFG["abstain"]["max_val_rows_for_sweep"]))
        seen = 0

        y1_list, y2_list, ysys_list = [], [], []
        p_list, fam_idx_list, pmax_list = [], [], []

        usecols = prep.num_cols + prep.cat_cols + [CFG["y_stage1"], CFG["y_stage2"]]
        usecols = [canonical_col(c) for c in usecols]

        for chunk in iter_rows_from_parts(parts_val, usecols=usecols, chunksize=CFG["chunksize_rows"]):
            chunk.columns = [canonical_col(c) for c in chunk.columns]
            y1 = chunk[CFG["y_stage1"]].astype(int).to_numpy()
            y2 = chunk[CFG["y_stage2"]].astype(str).fillna("").to_numpy()            # System-level labels used for tuning (val):
            # If LOAO is enabled, map families not in trained families -> Unknown.
            loao_enabled = bool((CFG.get("loao", {}) or {}).get("enabled", False))
            if loao_enabled and (CFG.get("abstain", {}).get("policy", "reject_to_benign") == "abstain"):
                unk = CFG["abstain"]["label"]
                fam_set = set(families)
                ysys = np.array(["Benign" if a == 0 else (b if str(b) in fam_set else unk) for a, b in zip(y1, y2)], dtype=object)
            else:
                ysys = np.array(["Benign" if a == 0 else b for a, b in zip(y1, y2)], dtype=object)


            X = prep.transform(chunk.drop(columns=[CFG["y_stage1"], CFG["y_stage2"]], errors="ignore"))
            p_attack = stage1.predict_proba(X)[:, 1].astype(np.float64)
            p_attack = apply_platt(p_attack, platt)

            p2 = stage2.predict_proba(X).astype(np.float64)
            p2 = apply_temperature(p2, T)

            fam_pred = np.argmax(p2, axis=1).astype(int)
            pmax = np.max(p2, axis=1).astype(np.float64)

            y1_list.append(y1)
            y2_list.append(y2)
            ysys_list.append(ysys)
            p_list.append(p_attack)
            fam_idx_list.append(fam_pred)
            pmax_list.append(pmax)

            seen += len(y1)
            if seen >= max_val:
                break

        y1v = np.concatenate(y1_list) if y1_list else np.array([], dtype=int)
        y2v = np.concatenate(y2_list) if y2_list else np.array([], dtype=object)
        ysysv = np.concatenate(ysys_list) if ysys_list else np.array([], dtype=object)
        pv = np.concatenate(p_list) if p_list else np.array([], dtype=float)
        fam_pred_v = np.concatenate(fam_idx_list) if fam_idx_list else np.array([], dtype=int)
        pmaxv = np.concatenate(pmax_list) if pmax_list else np.array([], dtype=float)

        # Strict thr_high selection (Option C)
        pol = CFG["stage1_threshold"].get("policy", "target_fpr_family_aware")
        tgt_fpr = float(CFG["stage1_threshold"].get("target_fpr", 0.01))

        if pol == "target_fpr_family_aware":
            thr_high, meta = choose_thr_high_family_aware(
                y1_val=y1v, y2_val=y2v, p_attack=pv,
                target_fpr=tgt_fpr,
                min_family_support=int(CFG["stage1_threshold"].get("min_family_support", 200)),
                sweep_points=int(CFG["stage1_threshold"].get("sweep_points", 60)),
                objective_mode=str(CFG["stage1_threshold"].get("objective_mode", "min_family_recall")),
                p10_q=float(CFG["stage1_threshold"].get("p10_quantile", 0.10)),
            )
            with open(os.path.join(run_dir, "stage1_threshold_strict_family_aware_sweep.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        elif pol == "target_fpr":
            thr_high = pick_threshold_target_fpr(y1v, pv, tgt_fpr)
        elif pol == "fixed":
            thr_high = float(CFG["stage1_threshold"].get("fixed", 0.5))
        else:
            # fallback
            thr_high = pick_threshold_target_fpr(y1v, pv, tgt_fpr)

        with open(os.path.join(run_dir, "stage1_threshold_strict.json"), "w", encoding="utf-8") as f:
            json.dump({"thr_high": float(thr_high), "policy": CFG["stage1_threshold"], "val_rows_used": int(seen)}, f, indent=2)

        # JOINT tuning for cascade (thr_low, tau_cascade)
        thr_low, tau_cascade = tune_cascade_thr_low_and_tau(run_dir, y1v, ysysv, pv, fam_pred_v, pmaxv, families, float(thr_high))
        with open(os.path.join(run_dir, "stage1_threshold_cascade.json"), "w", encoding="utf-8") as f:
            json.dump({"thr_low": float(thr_low), "policy": CFG["cascade_gate"], "val_rows_used": int(seen)}, f, indent=2)

        # strict tau (baseline)
        tau_strict = pick_tau_strict(run_dir, ysysv, pv, fam_pred_v, pmaxv, families, float(thr_high))

        with open(os.path.join(run_dir, "abstain_selected.json"), "w", encoding="utf-8") as f:
            json.dump({
                "policy": CFG["abstain"].get("policy", "reject_to_benign"),
                "label": CFG["abstain"]["label"],
                "thr_high": float(thr_high),
                "thr_low": float(thr_low),
                "tau_strict": float(tau_strict),
                "tau_cascade": float(tau_cascade),
                "cfg": CFG["abstain"],
            }, f, indent=2)

        # Evaluate val/test
        evaluate_system(ds_dir, run_dir, "val",  prep, stage1, platt, stage2, families, T, thr_high, thr_low, tau_strict, tau_cascade)
        evaluate_system(ds_dir, run_dir, "test", prep, stage1, platt, stage2, families, T, thr_high, thr_low, tau_strict, tau_cascade)

        with open(os.path.join(run_dir, "SUMMARY.txt"), "w", encoding="utf-8") as f:
            f.write(f"Dataset: {ds_name}\n")
            f.write(f"run_dir: {run_dir}\n")
            f.write(f"policy: {CFG['abstain'].get('policy','reject_to_benign')}\n")
            f.write(f"thr_high: {thr_high}\nthr_low: {thr_low}\n")
            f.write(f"tau_strict: {tau_strict}\ntau_cascade: {tau_cascade}\n")
            f.write("See system_compare_val.json and system_compare_test.json\n")
            f.write("Stage-2 metrics: metrics_stage2_{val,test}.json include macro_f1_fixedK and macro_f1_present + missing families.\n")

        progress_print(f"[done] {ds_name}: {run_dir}")


if __name__ == "__main__":
    main()
