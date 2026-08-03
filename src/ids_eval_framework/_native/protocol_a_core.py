#!/usr/bin/env python3
"""
7.ProtocolA_CoreRunner.py
=========================

Purpose
-------
Run fresh Protocol A closed-set baselines for the two primary thesis datasets and
emit one canonical summary table for the thesis core package.

Why this exists
---------------
`3.TrainEvalBase_V4.1A.py` already contains the rich preprocessing, calibration,
thresholding, and evaluation logic that we want to preserve. However, that script
is still effectively XGBoost-only and writes one-off run folders without a compact
cross-run summary table.

This runner keeps the trusted evaluation stack but adds:
    - fresh Protocol A runs under a dedicated V5 artifact root
    - RandomForest support for the same closed-set workflow
    - a canonical summary export for the thesis core package
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from ids_eval_framework.src.paths import repo_path


CFG: Dict[str, object] = {
    "processed_root": repo_path("processed_V5"),
    "protocol": "A_stratified",
    "datasets": ["CICIDS2017", "CICIoT2023"],
    "model_families": ["xgb", "rf"],
    "runs_root": "runs_two_stage_V5_A_core",
    "helper_script": "ids_eval_framework.src.two_stage_engine",
    "random_seed": 123,
    "n_jobs": 24,
    "xgb_tree_method": "hist",
    "xgb_device": "cuda",
    "xgb_predictor": None,
    "max_train_rows": {"CICIDS2017": 900_000, "CICIoT2023": 1_500_000},
    "max_val_rows": {"CICIDS2017": 500_000, "CICIoT2023": 650_000},
    "rf_stage1_grid": {
        "weight_modes": ["none", "class_weight_balanced"],
        "target_fpr": 0.015,
        "min_family_support": 100,
        "objective_mode": "min_family_recall",
        "p10_quantile": 0.10,
        "sweep_points": 60,
        "params_grid": [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 600, "max_depth": 30, "min_samples_leaf": 2, "max_features": 0.5},
        ],
        "results_csv": "stage1_rf_grid_results.csv",
        "progress_json": "stage1_rf_grid_progress.json",
        "models_dir": "stage1_rf_grid_models",
    },
    "rf_stage2_grid": {
        "params_grid": [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
            {"n_estimators": 600, "max_depth": 25, "min_samples_leaf": 2, "max_features": 0.5},
        ],
        "results_csv": "stage2_rf_grid_results.csv",
        "progress_json": "stage2_rf_grid_progress.json",
        "models_dir": "stage2_rf_grid_models",
    },
    "summary_dirname": "summary",
    "summary_csv": "protocol_a_core_summary.csv",
}


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def progress_print(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def stable_short_hash(obj: Dict[str, object], n: int = 10) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:n]


def load_helper_module(path: str):
    from ids_eval_framework.src import two_stage_engine

    return two_stage_engine


def configure_helper_for_protocol_a(helper) -> None:
    helper.CFG["processed_root"] = CFG["processed_root"]
    helper.CFG["protocol"] = CFG["protocol"]
    helper.CFG["datasets"] = list(CFG["datasets"])
    helper.CFG["runs_root"] = CFG["runs_root"]
    helper.CFG["resume_run_dirs"] = {ds: None for ds in CFG["datasets"]}
    helper.CFG["global_seed"] = int(CFG["random_seed"])
    helper.CFG["loao"]["enabled"] = False
    helper.CFG["loao"]["apply_to_stage1"] = False
    helper.CFG["stage1_xgb_grid"]["weight_mode_list"] = ["none"]
    xgb_tree_method = str(CFG.get("xgb_tree_method", "hist"))
    xgb_device = CFG.get("xgb_device", None)
    xgb_predictor = CFG.get("xgb_predictor", None)
    helper.CFG["stage1_xgb_grid"]["tree_method"] = xgb_tree_method
    helper.CFG["stage1_xgb_grid"]["predictor"] = xgb_predictor
    if xgb_device:
        helper.CFG["stage1_xgb_grid"]["device"] = str(xgb_device)
    else:
        helper.CFG["stage1_xgb_grid"].pop("device", None)
    helper.CFG["stage1_xgb_grid"]["n_jobs"] = int(CFG.get("n_jobs", 8))
    helper.CFG["stage2_xgb_grid"]["tree_method"] = xgb_tree_method
    helper.CFG["stage2_xgb_grid"]["predictor"] = xgb_predictor
    if xgb_device:
        helper.CFG["stage2_xgb_grid"]["device"] = str(xgb_device)
    else:
        helper.CFG["stage2_xgb_grid"].pop("device", None)
    helper.CFG["stage2_xgb_grid"]["n_jobs"] = int(CFG.get("n_jobs", 8))
    helper.CFG["stage1_xgb_grid"]["max_train_rows"] = dict(CFG["max_train_rows"])
    helper.CFG["stage1_xgb_grid"]["max_val_rows"] = dict(CFG["max_val_rows"])
    helper.CFG["stage2_xgb_grid"]["max_train_rows"] = dict(CFG["max_train_rows"])
    helper.CFG["stage2_xgb_grid"]["max_val_rows"] = dict(CFG["max_val_rows"])


def run_name(dataset: str, model_family: str) -> str:
    return f"{CFG['protocol']}__{dataset}__{model_family}__{now_stamp()}"


def run_prefix(dataset: str, model_family: str) -> str:
    return f"{CFG['protocol']}__{dataset}__{model_family}__"


def is_complete_protocol_a_run(run_dir: str) -> bool:
    required = [
        "run_identity.json",
        "system_compare_val.json",
        "system_compare_test.json",
        "metrics_stage1_test.json",
        "metrics_stage2_test.json",
    ]
    return all(os.path.exists(os.path.join(run_dir, name)) for name in required)


def choose_run_dir(runs_root: str, dataset: str, model_family: str) -> Tuple[str, str]:
    prefix = run_prefix(dataset, model_family)
    matches: List[str] = []
    if os.path.isdir(runs_root):
        for name in os.listdir(runs_root):
            run_dir = os.path.join(runs_root, name)
            if os.path.isdir(run_dir) and name.startswith(prefix):
                matches.append(run_dir)
    matches = sorted(matches, reverse=True)

    complete_matches = [p for p in matches if is_complete_protocol_a_run(p)]
    if complete_matches:
        return complete_matches[0], "skip_complete"

    if matches:
        return matches[0], "resume_partial"

    return os.path.join(runs_root, run_name(dataset, model_family)), "new"


def build_rf_stage1_model(params: Dict[str, object], weight_mode: str, seed: int):
    model_params = dict(params)
    model_params["random_state"] = int(seed)
    model_params.setdefault("n_jobs", int(CFG.get("n_jobs", 8)))
    if weight_mode == "class_weight_balanced":
        model_params["class_weight"] = "balanced_subsample"
    return RandomForestClassifier(**model_params)


def build_rf_stage2_model(params: Dict[str, object], seed: int):
    model_params = dict(params)
    model_params["random_state"] = int(seed)
    model_params.setdefault("n_jobs", int(CFG.get("n_jobs", 8)))
    return RandomForestClassifier(**model_params)


def predict_binary_proba(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    if p.ndim != 2 or p.shape[1] < 2:
        raise RuntimeError("Binary model returned an unexpected probability shape.")
    return p[:, 1].astype(np.float64)


def predict_multi_proba(model, X) -> np.ndarray:
    p = model.predict_proba(X).astype(np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def run_stage1_rf_grid(helper, ds_name: str, ds_dir: str, run_dir: str, prep) -> Tuple[object, Optional[Dict[str, float]]]:
    cfg = dict(CFG["rf_stage1_grid"])
    models_dir = os.path.join(run_dir, cfg["models_dir"])
    safe_mkdir(models_dir)

    results_path = os.path.join(run_dir, cfg["results_csv"])
    progress_path = os.path.join(run_dir, cfg["progress_json"])

    rows: List[Dict[str, object]] = []
    done_hashes = set()
    best_combo: Optional[Dict[str, object]] = None
    if os.path.exists(results_path):
        try:
            rows = pd.read_csv(results_path).to_dict("records")
        except Exception:
            rows = []
    if os.path.exists(progress_path):
        try:
            state = json.load(open(progress_path, "r", encoding="utf-8"))
            done_hashes = set(state.get("done_model_hashes", []))
            best_combo = state.get("best_combo")
        except Exception:
            done_hashes = set()
            best_combo = None

    parts_tr = helper.list_parts(ds_dir, "train")
    parts_va = helper.list_parts(ds_dir, "val")
    if not parts_tr or not parts_va:
        raise RuntimeError("Missing train/val parts for RF stage-1 grid.")

    Xtr, ytr, y2tr = helper.collect_xy(
        parts_tr,
        prep,
        helper.CFG["y_stage1"],
        int(CFG["max_train_rows"][ds_name]),
        seed=int(CFG["random_seed"]),
        filter_attack=None,
        y2_col=helper.CFG["y_stage2"],
    )
    Xva, yva, y2va = helper.collect_xy(
        parts_va,
        prep,
        helper.CFG["y_stage1"],
        int(CFG["max_val_rows"][ds_name]),
        seed=int(CFG["random_seed"]) + 1,
        filter_attack=None,
        y2_col=helper.CFG["y_stage2"],
    )
    if len(ytr) == 0 or len(yva) == 0:
        raise RuntimeError("RF stage-1 grid saw empty train or val arrays.")

    for weight_mode in cfg["weight_modes"]:
        for params in cfg["params_grid"]:
            combo = {"weight_mode": weight_mode, "params": params}
            combo_hash = stable_short_hash(combo, n=12)
            model_path = os.path.join(models_dir, f"{combo_hash}.joblib")
            if combo_hash in done_hashes and os.path.exists(model_path):
                continue

            model = build_rf_stage1_model(params, weight_mode, seed=int(CFG["random_seed"]))
            model.fit(Xtr, ytr)
            pva = predict_binary_proba(model, Xva)

            thr, meta = helper.choose_thr_high_family_aware(
                y1_val=yva,
                y2_val=y2va,
                p_attack=pva,
                target_fpr=float(cfg["target_fpr"]),
                min_family_support=int(cfg["min_family_support"]),
                sweep_points=int(cfg["sweep_points"]),
                objective_mode=str(cfg["objective_mode"]),
                p10_q=float(cfg["p10_quantile"]),
            )
            best_meta = dict(meta.get("best", {})) if isinstance(meta, dict) else {}

            row = {
                "hash": combo_hash,
                "weight_mode": weight_mode,
                "thr_at_target_fpr": float(thr),
                "target_fpr": float(cfg["target_fpr"]),
                "objective": float(best_meta.get("objective", np.nan)),
                "tpr_at_target_fpr": float(best_meta.get("tpr", np.nan)),
                "fpr_at_target_fpr": float(best_meta.get("fpr", np.nan)),
                "min_family_recall": float(best_meta.get("min_family_recall", np.nan)),
                "p10_family_recall": float(best_meta.get("p10_family_recall", np.nan)),
                "params": json.dumps(params, sort_keys=True),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(results_path, index=False)
            helper.safe_joblib_dump(model, model_path)
            done_hashes.add(combo_hash)

            if best_combo is None:
                best_combo = row
            else:
                better = False
                if row["objective"] > best_combo.get("objective", -1):
                    better = True
                elif row["objective"] == best_combo.get("objective", -1):
                    better = row["tpr_at_target_fpr"] > best_combo.get("tpr_at_target_fpr", -1)
                if better:
                    best_combo = row

            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"done_model_hashes": sorted(done_hashes), "best_combo": best_combo}, f, indent=2)

    if best_combo is None:
        raise RuntimeError("RF stage-1 grid produced no result.")

    best_model = helper.safe_joblib_load(os.path.join(models_dir, f"{best_combo['hash']}.joblib"))
    pva_best = predict_binary_proba(best_model, Xva)
    platt = helper.fit_platt_on_probs(pva_best, yva)
    with open(os.path.join(run_dir, "stage1_best.json"), "w", encoding="utf-8") as f:
        json.dump(best_combo, f, indent=2)
    with open(os.path.join(run_dir, "stage1_platt.json"), "w", encoding="utf-8") as f:
        json.dump({"platt": platt}, f, indent=2)

    pred_attack = (pva_best >= float(best_combo["thr_at_target_fpr"])).astype(int)
    recs, min_rec, p10_rec = helper.family_recall_stats(
        yva,
        y2va,
        pred_attack.astype(bool),
        int(cfg["min_family_support"]),
    )
    with open(os.path.join(run_dir, "stage1_family_recall_at_target_fpr.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": float(best_combo["thr_at_target_fpr"]),
                "target_fpr": float(best_combo["target_fpr"]),
                "fpr_at_target_fpr": float(best_combo["fpr_at_target_fpr"]),
                "tpr_at_target_fpr": float(best_combo["tpr_at_target_fpr"]),
                "objective_mode": str(cfg["objective_mode"]),
                "p10_quantile": float(cfg["p10_quantile"]),
                "min_support": int(cfg["min_family_support"]),
                "recs": recs,
                "min_family_recall": float(min_rec),
                "p10_family_recall": float(p10_rec),
                "weight_mode": str(best_combo["weight_mode"]),
            },
            f,
            indent=2,
        )
    return best_model, platt


def run_stage2_rf_grid(helper, ds_name: str, ds_dir: str, run_dir: str, prep) -> Tuple[object, List[str], float]:
    cfg = dict(CFG["rf_stage2_grid"])
    models_dir = os.path.join(run_dir, cfg["models_dir"])
    safe_mkdir(models_dir)

    results_path = os.path.join(run_dir, cfg["results_csv"])
    progress_path = os.path.join(run_dir, cfg["progress_json"])

    rows: List[Dict[str, object]] = []
    done_hashes = set()
    best: Optional[Dict[str, object]] = None
    if os.path.exists(results_path):
        try:
            rows = pd.read_csv(results_path).to_dict("records")
        except Exception:
            rows = []
    if os.path.exists(progress_path):
        try:
            state = json.load(open(progress_path, "r", encoding="utf-8"))
            done_hashes = set(state.get("done_hashes", []))
            best = state.get("best")
        except Exception:
            done_hashes = set()
            best = None

    parts_tr = helper.list_parts(ds_dir, "train")
    parts_va = helper.list_parts(ds_dir, "val")
    Xtr, _, y2tr = helper.collect_xy(
        parts_tr,
        prep,
        helper.CFG["y_stage1"],
        int(CFG["max_train_rows"][ds_name]),
        seed=int(CFG["random_seed"]),
        filter_attack=True,
        y2_col=helper.CFG["y_stage2"],
    )
    Xva, _, y2va = helper.collect_xy(
        parts_va,
        prep,
        helper.CFG["y_stage1"],
        int(CFG["max_val_rows"][ds_name]),
        seed=int(CFG["random_seed"]) + 1,
        filter_attack=True,
        y2_col=helper.CFG["y_stage2"],
    )
    if y2tr is None or len(y2tr) == 0:
        raise RuntimeError("RF stage-2 grid found no attack samples in training.")
    if y2va is None or len(y2va) == 0:
        raise RuntimeError("RF stage-2 grid found no attack samples in validation.")

    families = sorted({str(x) for x in y2tr if str(x) and str(x).lower() != "nan"})
    fam_to_idx = {fam: i for i, fam in enumerate(families)}
    if len(families) < 2:
        raise RuntimeError(f"Stage-2 RF needs >=2 families, found {len(families)}.")

    ytr = np.array([fam_to_idx[str(x)] for x in y2tr], dtype=int)
    mask_val_ok = np.array([str(x) in fam_to_idx for x in y2va], dtype=bool)
    Xva_ok = Xva[mask_val_ok]
    yva = np.array([fam_to_idx[str(x)] for x in y2va[mask_val_ok]], dtype=int)
    class_weight = helper.compute_class_weights(ytr, len(families))
    sample_weight = class_weight[ytr]
    labels_fixed = list(range(len(families)))

    for params in cfg["params_grid"]:
        combo_hash = stable_short_hash(params, n=12)
        model_path = os.path.join(models_dir, f"{combo_hash}.joblib")
        if combo_hash in done_hashes and os.path.exists(model_path):
            continue

        model = build_rf_stage2_model(params, seed=int(CFG["random_seed"]) + 10)
        model.fit(Xtr, ytr, sample_weight=sample_weight)

        pva = predict_multi_proba(model, Xva_ok)
        pred = np.argmax(pva, axis=1).astype(int)
        mf1_fixed = helper.macro_f1_fixedK(yva, pred, len(families))
        mf1_pres = helper.macro_f1_present(yva, pred)
        acc = float(accuracy_score(yva, pred))
        nll = float(log_loss(yva, pva, labels=labels_fixed))

        row = {
            "hash": combo_hash,
            "val_macro_f1_fixedK": float(mf1_fixed),
            "val_macro_f1_present": float(mf1_pres),
            "val_accuracy": float(acc),
            "val_nll": float(nll),
            "n_val_attacks": int(len(yva)),
            "params": json.dumps(params, sort_keys=True),
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(results_path, index=False)
        helper.safe_joblib_dump(model, model_path)
        done_hashes.add(combo_hash)

        if best is None:
            best = row
        else:
            better = False
            if row["val_macro_f1_present"] > best.get("val_macro_f1_present", -1):
                better = True
            elif row["val_macro_f1_present"] == best.get("val_macro_f1_present", -1):
                if row["val_nll"] < best.get("val_nll", 1e18):
                    better = True
                elif row["val_nll"] == best.get("val_nll", 1e18):
                    better = row["val_macro_f1_fixedK"] > best.get("val_macro_f1_fixedK", -1)
            if better:
                best = row

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"done_hashes": sorted(done_hashes), "best": best}, f, indent=2)

    if best is None:
        raise RuntimeError("RF stage-2 grid produced no result.")

    best_model = helper.safe_joblib_load(os.path.join(models_dir, f"{best['hash']}.joblib"))
    pva_best = predict_multi_proba(best_model, Xva_ok)
    T = helper.fit_temperature_on_probs(pva_best, yva, len(families))
    with open(os.path.join(run_dir, "stage2_best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    with open(os.path.join(run_dir, "stage2_temperature.json"), "w", encoding="utf-8") as f:
        json.dump({"T": float(T), "hash": best["hash"], "params": json.loads(best["params"]), "K": int(len(families))}, f, indent=2)
    return best_model, families, float(T)


def run_protocol_a_eval(helper, ds_name: str, model_family: str, run_dir: str) -> None:
    ds_dir = os.path.join(str(CFG["processed_root"]), str(CFG["protocol"]), ds_name)
    if not os.path.isdir(ds_dir):
        raise RuntimeError(f"Dataset folder not found: {ds_dir}")

    safe_mkdir(run_dir)
    identity = {
        "dataset": ds_name,
        "protocol": str(CFG["protocol"]),
        "model_family": model_family,
        "processed_dir": ds_dir,
        "run_name": os.path.basename(run_dir),
        "run_started_at": now_stamp(),
    }
    with open(os.path.join(run_dir, "run_identity.json"), "w", encoding="utf-8") as f:
        json.dump(identity, f, indent=2)

    prep_path = os.path.join(run_dir, "preprocessor.joblib")
    if os.path.exists(prep_path):
        prep = helper.safe_joblib_load(prep_path)
    else:
        prep = helper.fit_preprocessor(ds_dir, run_dir)

    stage1_path = os.path.join(run_dir, "stage1_best.joblib")
    stage2_path = os.path.join(run_dir, "stage2_best.joblib")
    fam_path = os.path.join(run_dir, "families.json")
    temp_path = os.path.join(run_dir, "stage2_temperature.json")
    platt_path = os.path.join(run_dir, "stage1_platt.json")

    if model_family == "xgb":
        if not os.path.exists(stage1_path):
            stage1, platt = helper.run_stage1_xgb_grid(ds_name, ds_dir, run_dir, prep)
            helper.safe_joblib_dump(stage1, stage1_path)
        else:
            stage1 = helper.safe_joblib_load(stage1_path)
            platt = json.load(open(platt_path, "r", encoding="utf-8")).get("platt")

        if not (os.path.exists(stage2_path) and os.path.exists(temp_path) and os.path.exists(fam_path)):
            stage2, families, T = helper.run_stage2_xgb_grid(ds_name, ds_dir, run_dir, prep)
            helper.safe_joblib_dump(stage2, stage2_path)
            with open(fam_path, "w", encoding="utf-8") as f:
                json.dump(families, f, indent=2)
        else:
            stage2 = helper.safe_joblib_load(stage2_path)
            families = json.load(open(fam_path, "r", encoding="utf-8"))
            T = float(json.load(open(temp_path, "r", encoding="utf-8"))["T"])
    else:
        if not os.path.exists(stage1_path):
            stage1, platt = run_stage1_rf_grid(helper, ds_name, ds_dir, run_dir, prep)
            helper.safe_joblib_dump(stage1, stage1_path)
        else:
            stage1 = helper.safe_joblib_load(stage1_path)
            platt = json.load(open(platt_path, "r", encoding="utf-8")).get("platt")

        if not (os.path.exists(stage2_path) and os.path.exists(temp_path) and os.path.exists(fam_path)):
            stage2, families, T = run_stage2_rf_grid(helper, ds_name, ds_dir, run_dir, prep)
            helper.safe_joblib_dump(stage2, stage2_path)
            with open(fam_path, "w", encoding="utf-8") as f:
                json.dump(families, f, indent=2)
        else:
            stage2 = helper.safe_joblib_load(stage2_path)
            families = json.load(open(fam_path, "r", encoding="utf-8"))
            T = float(json.load(open(temp_path, "r", encoding="utf-8"))["T"])

    parts_val = helper.list_parts(ds_dir, "val")
    if not parts_val:
        raise RuntimeError("No val parts found for Protocol A evaluation.")

    max_val = int(min(helper.CFG["cascade_gate"]["max_val_rows_for_sweep"], helper.CFG["abstain"]["max_val_rows_for_sweep"]))
    seen = 0
    y1_list, y2_list, ysys_list = [], [], []
    p_list, fam_idx_list, pmax_list = [], [], []
    usecols = prep.num_cols + prep.cat_cols + [helper.CFG["y_stage1"], helper.CFG["y_stage2"]]
    usecols = [helper.canonical_col(c) for c in usecols]

    for chunk in helper.iter_rows_from_parts(parts_val, usecols=usecols, chunksize=helper.CFG["chunksize_rows"]):
        chunk.columns = [helper.canonical_col(c) for c in chunk.columns]
        y1 = chunk[helper.CFG["y_stage1"]].astype(int).to_numpy()
        y2 = chunk[helper.CFG["y_stage2"]].astype(str).fillna("").to_numpy()
        ysys = np.array(["Benign" if a == 0 else b for a, b in zip(y1, y2)], dtype=object)

        X = prep.transform(chunk.drop(columns=[helper.CFG["y_stage1"], helper.CFG["y_stage2"]], errors="ignore"))
        p_attack = predict_binary_proba(stage1, X)
        p_attack = helper.apply_platt(p_attack, platt)

        p2 = predict_multi_proba(stage2, X)
        p2 = helper.apply_temperature(p2, T)
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

    thr_high, meta = helper.choose_thr_high_family_aware(
        y1_val=y1v,
        y2_val=y2v,
        p_attack=pv,
        target_fpr=float(helper.CFG["stage1_threshold"].get("target_fpr", 0.015)),
        min_family_support=int(helper.CFG["stage1_threshold"].get("min_family_support", 50)),
        sweep_points=int(helper.CFG["stage1_threshold"].get("sweep_points", 60)),
        objective_mode=str(helper.CFG["stage1_threshold"].get("objective_mode", "min_family_recall")),
        p10_q=float(helper.CFG["stage1_threshold"].get("p10_quantile", 0.10)),
    )
    with open(os.path.join(run_dir, "stage1_threshold_strict_family_aware_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(run_dir, "stage1_threshold_strict.json"), "w", encoding="utf-8") as f:
        json.dump({"thr_high": float(thr_high), "policy": helper.CFG["stage1_threshold"], "val_rows_used": int(seen)}, f, indent=2)

    thr_low, tau_cascade = helper.tune_cascade_thr_low_and_tau(run_dir, y1v, ysysv, pv, fam_pred_v, pmaxv, families, float(thr_high))
    with open(os.path.join(run_dir, "stage1_threshold_cascade.json"), "w", encoding="utf-8") as f:
        json.dump({"thr_low": float(thr_low), "policy": helper.CFG["cascade_gate"], "val_rows_used": int(seen)}, f, indent=2)

    tau_strict = helper.pick_tau_strict(run_dir, ysysv, pv, fam_pred_v, pmaxv, families, float(thr_high))
    with open(os.path.join(run_dir, "abstain_selected.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": helper.CFG["abstain"].get("policy", "abstain"),
                "label": helper.CFG["abstain"]["label"],
                "thr_high": float(thr_high),
                "thr_low": float(thr_low),
                "tau_strict": float(tau_strict),
                "tau_cascade": float(tau_cascade),
                "cfg": helper.CFG["abstain"],
            },
            f,
            indent=2,
        )

    helper.evaluate_system(ds_dir, run_dir, "val", prep, stage1, platt, stage2, families, T, thr_high, thr_low, tau_strict, tau_cascade)
    helper.evaluate_system(ds_dir, run_dir, "test", prep, stage1, platt, stage2, families, T, thr_high, thr_low, tau_strict, tau_cascade)

    with open(os.path.join(run_dir, "SUMMARY.txt"), "w", encoding="utf-8") as f:
        f.write(f"Dataset: {ds_name}\n")
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"model_family: {model_family}\n")
        f.write(f"protocol: {CFG['protocol']}\n")
        f.write(f"thr_high: {thr_high}\nthr_low: {thr_low}\n")
        f.write(f"tau_strict: {tau_strict}\ntau_cascade: {tau_cascade}\n")
        f.write("See system_compare_val.json and system_compare_test.json\n")
        f.write("Stage-2 metrics: metrics_stage2_{val,test}.json include macro_f1_fixedK and macro_f1_present + missing families.\n")


def collect_protocol_a_summary_rows(runs_root: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for name in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        identity_path = os.path.join(run_dir, "run_identity.json")
        system_path = os.path.join(run_dir, "system_compare_test.json")
        stage1_metrics_path = os.path.join(run_dir, "metrics_stage1_test.json")
        stage2_metrics_path = os.path.join(run_dir, "metrics_stage2_test.json")
        stage1_cal_path = os.path.join(run_dir, "calibration_stage1_test.json")
        system_cal_path = os.path.join(run_dir, "calibration_system_known_strict_tau_test.json")
        if not all(os.path.exists(p) for p in [identity_path, system_path, stage1_metrics_path, stage2_metrics_path]):
            continue

        identity = json.load(open(identity_path, "r", encoding="utf-8"))
        system_compare = json.load(open(system_path, "r", encoding="utf-8"))
        stage1_metrics = json.load(open(stage1_metrics_path, "r", encoding="utf-8"))
        stage2_metrics = json.load(open(stage2_metrics_path, "r", encoding="utf-8"))
        stage1_cal = json.load(open(stage1_cal_path, "r", encoding="utf-8")) if os.path.exists(stage1_cal_path) else {}
        system_cal = json.load(open(system_cal_path, "r", encoding="utf-8")) if os.path.exists(system_cal_path) else {}
        stage1_best = json.load(open(os.path.join(run_dir, "stage1_best.json"), "r", encoding="utf-8")) if os.path.exists(os.path.join(run_dir, "stage1_best.json")) else {}

        for policy_variant in ["strict", "strict_tau"]:
            sys_row = dict(system_compare.get(policy_variant, {}))
            rows.append(
                {
                    "dataset": identity.get("dataset"),
                    "model_family": identity.get("model_family"),
                    "policy_variant": policy_variant,
                    "protocol": identity.get("protocol"),
                    "run_name": identity.get("run_name"),
                    "run_dir": run_dir,
                    "stage1_weight_mode": stage1_best.get("weight_mode"),
                    "stage1_roc_auc": stage1_metrics.get("roc_auc"),
                    "stage1_fpr": stage1_metrics.get("fpr"),
                    "stage1_tpr": stage1_metrics.get("tpr"),
                    "stage2_macro_f1_fixedK": stage2_metrics.get("macro_f1_fixedK"),
                    "stage2_macro_f1_present": stage2_metrics.get("macro_f1_present"),
                    "stage2_accuracy": stage2_metrics.get("accuracy"),
                    "system_macro_f1_supported_labels": sys_row.get(
                        "system_macro_f1_supported_labels", sys_row.get("macro_f1")
                    ),
                    "system_macro_f1_declared_output_labels_historical": sys_row.get(
                        "system_macro_f1_declared_output_labels_historical"
                    ),
                    "system_accuracy": sys_row.get("accuracy"),
                    "benign_family_fp_rate": sys_row.get("benign_family_fp_rate"),
                    "overall_reject_rate": sys_row.get("overall_reject_rate"),
                    "stage1_brier": stage1_cal.get("brier"),
                    "stage1_ece": stage1_cal.get("ece"),
                    "system_brier_multiclass_known": system_cal.get("brier_multiclass_known"),
                    "system_ece_toplabel_known": system_cal.get("ece_toplabel_known"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["dataset", "model_family", "policy_variant", "run_name"], ascending=[True, True, True, False]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["dataset", "model_family", "policy_variant"], keep="first").reset_index(drop=True)
    return df


def write_protocol_a_summary(runs_root: str) -> str:
    out_dir = os.path.join(runs_root, str(CFG["summary_dirname"]))
    safe_mkdir(out_dir)
    df = collect_protocol_a_summary_rows(runs_root)
    out_csv = os.path.join(out_dir, str(CFG["summary_csv"]))
    df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    runs_root = str(CFG["runs_root"])
    safe_mkdir(runs_root)

    helper = load_helper_module(str(CFG["helper_script"]))
    configure_helper_for_protocol_a(helper)
    helper.ensure_deps()

    for ds_name in CFG["datasets"]:
        for model_family in CFG["model_families"]:
            run_dir, action = choose_run_dir(runs_root, ds_name, model_family)
            progress_print(f"\n=== Protocol A core run: dataset={ds_name} model={model_family} ===")
            progress_print(f"run_dir={run_dir}")
            if action == "skip_complete":
                progress_print(f"[skip] found completed run for {ds_name} / {model_family}")
                continue
            if action == "resume_partial":
                progress_print(f"[resume] found partial run for {ds_name} / {model_family}")
            run_protocol_a_eval(helper, ds_name, model_family, run_dir)
            progress_print(f"[done] {ds_name} / {model_family}: {run_dir}")

    out_csv = write_protocol_a_summary(runs_root)
    progress_print(f"Wrote Protocol A summary: {out_csv}")


if __name__ == "__main__":
    main()
