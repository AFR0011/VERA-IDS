#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ids_eval_framework._native.full_scope_utils import resolve_repo_path, safe_mkdir, split_variant_name_for_dataset


CFG: Dict[str, object] = {
    "step5_root": "protocolB_grid_runs step 5 - open-set baselines",
    "failure_modes_csv": os.path.join("thesis_core_pack", "protocol_b_failure_modes.csv"),
    "runs_root": "protocolB_grid_runs step 6 - sink-aware rejection",
    "summary_dirname": "summary",
    "selected_cases": [],
    "sink_tau_quantiles": [0.20, 0.35, 0.50, 0.65, 0.80],
    "margin_grid": [0.00, 0.02, 0.05, 0.08, 0.10, 0.15],
    "bootstrap_resamples": 200,
    "random_seed": 123,
    "case_parallel_workers": None,
    "parallel_backend": "process",
}


CASE_COMPARISON_CSV = "comparison_rows_case.csv"


def safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass


def apply_env_overrides() -> None:
    step5_root = os.environ.get("IDS_STEP6_STEP5_ROOT")
    if step5_root:
        CFG["step5_root"] = step5_root

    runs_root = os.environ.get("IDS_STEP6_RUNS_ROOT")
    if runs_root:
        CFG["runs_root"] = runs_root

    selected_cases_json = os.environ.get("IDS_STEP6_SELECTED_CASES_JSON")
    if selected_cases_json:
        parsed = json.loads(selected_cases_json)
        if not isinstance(parsed, list):
            raise ValueError("IDS_STEP6_SELECTED_CASES_JSON must decode to a JSON list.")
        CFG["selected_cases"] = [str(x) for x in parsed]

    workers = os.environ.get("IDS_STEP6_CASE_WORKERS")
    if workers is not None and str(workers).strip():
        CFG["case_parallel_workers"] = int(workers)
    backend = os.environ.get("IDS_STEP6_PARALLEL_BACKEND")
    if backend is not None and str(backend).strip():
        CFG["parallel_backend"] = str(backend).strip().lower()

    bootstrap = os.environ.get("IDS_STEP6_BOOTSTRAP_RESAMPLES")
    if bootstrap is not None and str(bootstrap).strip():
        CFG["bootstrap_resamples"] = int(bootstrap)


def sanitize_token(value: object) -> str:
    text = str(value).strip()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("._") or "NA"


def normalize_case_key(dataset: str, holdout: str) -> str:
    return f"{dataset}::{holdout}"


def case_run_dir(root: str, dataset: str, holdout: str, model_profile: object = "") -> str:
    profile = str(model_profile or "").strip()
    profile_token = f"__{sanitize_token(profile)}" if profile else ""
    return os.path.join(root, f"{dataset}__holdout_{sanitize_token(holdout)}{profile_token}__sink_aware")


def step5_case_run_dir(step5_root: str, dataset: str, holdout: str, model_profile: object = "") -> str:
    profile = str(model_profile or "").strip()
    profile_token = f"__{sanitize_token(profile)}" if profile else ""
    run_dir = os.path.join(step5_root, f"{dataset}__holdout_{sanitize_token(holdout)}{profile_token}__winner_replay")
    if not os.path.exists(run_dir):
        run_dir = os.path.join(step5_root, f"{dataset}__holdout_{sanitize_token(holdout)}__winner_replay")
    return run_dir


def case_is_complete(case_out: str) -> bool:
    needed = ["frontier_table.csv", "selected_candidate.json", "summary.json"]
    return all(os.path.exists(os.path.join(case_out, name)) for name in needed)


def comparison_cache_path(case_out: str) -> str:
    return os.path.join(case_out, CASE_COMPARISON_CSV)


def resolve_execution_settings(num_cases: int) -> Dict[str, object]:
    cpu_total = max(1, int(os.cpu_count() or 1))
    workers_cfg = CFG.get("case_parallel_workers")
    if workers_cfg is None:
        case_parallel_workers = max(1, min(num_cases or 1, min(16, cpu_total)))
    else:
        case_parallel_workers = max(1, min(num_cases or 1, int(workers_cfg)))
    return {
        "cpu_total": cpu_total,
        "case_parallel_workers": int(case_parallel_workers),
        "parallel_backend": str(CFG.get("parallel_backend", "process")).strip().lower(),
    }


def load_scores(run_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def parse_json_dict(text: object) -> Dict[str, object]:
    if isinstance(text, dict):
        return dict(text)
    s = str(text).strip()
    return dict(json.loads(s)) if s else {}


def load_step5_case_records(step5_root: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for entry in os.scandir(step5_root):
        if not entry.is_dir():
            continue
        if not entry.name.endswith("__winner_replay"):
            continue
        ref_path = os.path.join(entry.path, "winner_reference.json")
        scores_path = os.path.join(entry.path, "test_scores.csv.gz")
        selected_path = os.path.join(entry.path, "selected_methods.csv")
        if not os.path.exists(ref_path) or not os.path.exists(scores_path) or not os.path.exists(selected_path):
            continue
        with open(ref_path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        model_profile = str(rec.get("model_profile", rec.get("model_family", "")))
        rows.append({
            "dataset": str(rec["dataset"]),
            "holdout_family": str(rec["holdout_family"]),
            "model_profile": model_profile,
            "_profile_scoped_dir": bool(model_profile and sanitize_token(model_profile) in entry.name),
        })
    if not rows:
        raise RuntimeError(f"No usable step-5 winner_replay case folders were found under {step5_root}.")
    return (
        pd.DataFrame(rows)
        .sort_values(["dataset", "holdout_family", "model_profile", "_profile_scoped_dir"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["dataset", "holdout_family", "model_profile"])
        .drop(columns=["_profile_scoped_dir"], errors="ignore")
        .reset_index(drop=True)
    )


def system_metrics(y_true: np.ndarray, y_pred: np.ndarray, reject: np.ndarray, known_families: Sequence[str], unknown_label: str) -> Dict[str, object]:
    true_unk = y_true == unknown_label
    pred_unk = y_pred == unknown_label
    known_all = ~true_unk
    known_att = (y_true != "Benign") & known_all
    benign = y_true == "Benign"
    fam_set = set(known_families)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=["Benign", *known_families, unknown_label], zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "benign_family_fp_rate": float(np.mean([p in fam_set for p in y_pred[benign]])) if benign.sum() else 0.0,
        "benign_reject_rate": float(np.mean(reject[benign])) if benign.sum() else 0.0,
        "overall_reject_rate": float(np.mean(reject)) if len(reject) else 0.0,
        "unknown_detection_rate": float(np.mean(pred_unk[true_unk])) if true_unk.sum() else 0.0,
        "false_unknown_rate_all_known": float(np.mean(pred_unk[known_all])) if known_all.sum() else 0.0,
        "false_unknown_rate_known_attacks": float(np.mean(pred_unk[known_att])) if known_att.sum() else 0.0,
        "n_true_unknown": int(true_unk.sum()),
        "n_known_all": int(known_all.sum()),
        "n_known_attacks": int(known_att.sum()),
    }


def predict_sink_aware(scores: pd.DataFrame, thr_high: float, global_tau: float, sink_family: str, sink_tau: float, margin_floor: float, unknown_label: str) -> Tuple[np.ndarray, np.ndarray]:
    p_attack = pd.to_numeric(scores["p_attack"], errors="coerce").to_numpy(dtype=np.float64)
    fam_pred = scores["fam_pred_family"].astype(str).to_numpy(dtype=object)
    fam_pmax = pd.to_numeric(scores["fam_pmax"], errors="coerce").to_numpy(dtype=np.float64)
    margin = pd.to_numeric(scores["top2_margin"], errors="coerce").to_numpy(dtype=np.float64)
    pred = np.empty(len(scores), dtype=object)
    pred[p_attack < thr_high] = "Benign"
    reject = np.zeros(len(scores), dtype=bool)
    idx = np.where(p_attack >= thr_high)[0]
    if len(idx) == 0:
        return pred, reject
    pred[idx] = fam_pred[idx]
    global_reject = fam_pmax[idx] < global_tau
    sink_reject = (fam_pred[idx] == sink_family) & ((fam_pmax[idx] < sink_tau) | (margin[idx] < margin_floor))
    reject_idx = idx[global_reject | sink_reject]
    pred[reject_idx] = unknown_label
    reject[reject_idx] = True
    return pred, reject


def encode_label_codes(values: np.ndarray, labels: Sequence[str], fallback_label: str) -> np.ndarray:
    categories = pd.Categorical(pd.Series(values, dtype="string"), categories=list(labels))
    codes = np.asarray(categories.codes, dtype=np.int16).copy()
    fallback_idx = int(list(labels).index(fallback_label))
    codes[codes < 0] = fallback_idx
    return codes


def macro_f1_from_codes(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> float:
    cm = np.bincount(y_true.astype(np.int64) * n_labels + y_pred.astype(np.int64), minlength=n_labels * n_labels).reshape(n_labels, n_labels)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    denom = (2.0 * tp) + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(np.mean(f1))


def bootstrap_diff(y_true: np.ndarray, pred_a: np.ndarray, reject_a: np.ndarray, pred_b: np.ndarray, reject_b: np.ndarray, known_families: Sequence[str], unknown_label: str, seed: int, resamples: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = int(len(y_true))
    labels = ["Benign", *known_families, unknown_label]
    n_labels = len(labels)
    unknown_idx = labels.index(unknown_label)
    y_true_codes = encode_label_codes(y_true, labels, unknown_label)
    pred_a_codes = encode_label_codes(pred_a, labels, unknown_label)
    pred_b_codes = encode_label_codes(pred_b, labels, unknown_label)
    true_unknown = y_true_codes == unknown_idx
    pred_a_unknown = pred_a_codes == unknown_idx
    pred_b_unknown = pred_b_codes == unknown_idx
    macro_diff: List[float] = []
    udr_diff: List[float] = []
    for _ in range(int(resamples)):
        sample_idx = rng.integers(0, n, size=n, dtype=np.int64)
        macro_a = macro_f1_from_codes(y_true_codes[sample_idx], pred_a_codes[sample_idx], n_labels)
        macro_b = macro_f1_from_codes(y_true_codes[sample_idx], pred_b_codes[sample_idx], n_labels)
        sampled_unknown = true_unknown[sample_idx]
        n_unknown = int(sampled_unknown.sum())
        if n_unknown:
            udr_a = float(np.mean(pred_a_unknown[sample_idx][sampled_unknown]))
            udr_b = float(np.mean(pred_b_unknown[sample_idx][sampled_unknown]))
        else:
            udr_a = 0.0
            udr_b = 0.0
        macro_diff.append(macro_a - macro_b)
        udr_diff.append(udr_a - udr_b)
    return {
        "delta_macro_f1_mean": float(np.mean(macro_diff)),
        "delta_macro_f1_ci_low": float(np.quantile(macro_diff, 0.025)),
        "delta_macro_f1_ci_high": float(np.quantile(macro_diff, 0.975)),
        "delta_unknown_detection_mean": float(np.mean(udr_diff)),
        "delta_unknown_detection_ci_low": float(np.quantile(udr_diff, 0.025)),
        "delta_unknown_detection_ci_high": float(np.quantile(udr_diff, 0.975)),
    }


def build_comparison_rows(
    rec: Dict[str, object],
    failure: pd.DataFrame,
    step5_root: str,
    best: Dict[str, object],
    met_test: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    dataset = str(rec["dataset"])
    holdout = str(rec["holdout_family"])
    rec_profile = str(rec.get("model_profile", "") or "")
    run_dir = step5_case_run_dir(step5_root, dataset, holdout, rec_profile)

    val_scores = load_scores(run_dir, "val_scores.csv.gz")
    test_scores = load_scores(run_dir, "test_scores.csv.gz")
    selected_methods = pd.read_csv(os.path.join(run_dir, "selected_methods.csv"))
    control = selected_methods.loc[(selected_methods["split"] == "val") & (selected_methods["method"] == "control_tau")].iloc[0]
    control_test = selected_methods.loc[(selected_methods["split"] == "test") & (selected_methods["method"] == "control_tau")].iloc[0]
    control_params = parse_json_dict(control["selection_param_json"])
    global_tau = float(control_params["tau"])
    thr_high = float(control["thr_high"])
    split_variant = str(control["split_variant"])
    model_family = str(control["model_family"])
    model_profile = str(control.get("model_profile", model_family))
    stage1_weight_mode = str(control["stage1_weight_mode"])
    stage2_weight_mode = str(control.get("stage2_weight_mode", "unknown"))
    sink_family = str(failure.loc[(failure["dataset"] == dataset) & (failure["holdout_family"] == holdout), "top_sink_family"].iloc[0])
    unknown_label = "Unknown"
    known_families = sorted(set(val_scores["fam_pred_family"].astype(str).tolist()) | set(test_scores["fam_pred_family"].astype(str).tolist()))
    known_families = [fam for fam in known_families if fam not in {"Benign", unknown_label}]

    pred_test, reject_test = predict_sink_aware(
        test_scores,
        thr_high,
        global_tau,
        sink_family,
        float(best["sink_tau"]),
        float(best["margin_floor"]),
        unknown_label,
    )
    if met_test is None:
        met_test = system_metrics(test_scores["y_true_sys"].astype(str).to_numpy(dtype=object), pred_test, reject_test, known_families, unknown_label)

    comparison_rows: List[Dict[str, object]] = []
    comparison_rows.append({
        "dataset": dataset,
        "holdout_family": holdout,
        "split_variant": split_variant,
        "model_family": model_family,
        "model_profile": model_profile,
        "stage1_weight_mode": stage1_weight_mode,
        "stage2_weight_mode": stage2_weight_mode,
        "method": "control_tau",
        "sink_family": sink_family,
        "global_tau": global_tau,
        "sink_tau": np.nan,
        "margin_floor": np.nan,
        "macro_f1": float(control_test["macro_f1"]),
        "accuracy": float(control_test["accuracy"]),
        "unknown_detection_rate": float(control_test["unknown_detection_rate"]),
        "false_unknown_rate_all_known": float(control_test["false_unknown_rate_all_known"]),
        "false_unknown_rate_known_attacks": float(control_test["false_unknown_rate_known_attacks"]),
        "benign_family_fp_rate": float(control_test["benign_family_fp_rate"]),
        "overall_reject_rate": float(control_test["overall_reject_rate"]),
    })
    comparison_rows.append({
        "dataset": dataset,
        "holdout_family": holdout,
        "split_variant": split_variant,
        "model_family": model_family,
        "model_profile": model_profile,
        "stage1_weight_mode": stage1_weight_mode,
        "stage2_weight_mode": stage2_weight_mode,
        "method": "sink_aware_reject",
        "sink_family": sink_family,
        "global_tau": global_tau,
        "sink_tau": float(best["sink_tau"]),
        "margin_floor": float(best["margin_floor"]),
        **met_test,
    })

    control_pred, control_reject = predict_sink_aware(test_scores, thr_high, global_tau, sink_family, global_tau, -1.0, unknown_label)
    diff = bootstrap_diff(
        test_scores["y_true_sys"].astype(str).to_numpy(dtype=object),
        pred_test,
        reject_test,
        control_pred,
        control_reject,
        known_families,
        unknown_label,
        int(CFG["random_seed"]),
        int(CFG["bootstrap_resamples"]),
    )
    comparison_rows.append({
        "dataset": dataset,
        "holdout_family": holdout,
        "split_variant": split_variant,
        "model_family": model_family,
        "model_profile": model_profile,
        "stage1_weight_mode": stage1_weight_mode,
        "stage2_weight_mode": stage2_weight_mode,
        "method": "sink_aware_delta_vs_control",
        "sink_family": sink_family,
        **diff,
    })
    return comparison_rows


def load_completed_case(rec: Dict[str, object], failure: pd.DataFrame, step5_root: str, out_root: str) -> Dict[str, List[Dict[str, object]]]:
    dataset = str(rec["dataset"])
    holdout = str(rec["holdout_family"])
    model_profile = str(rec.get("model_profile", "") or "")
    case_out = case_run_dir(out_root, dataset, holdout, model_profile)
    frontier_rows = pd.read_csv(os.path.join(case_out, "frontier_table.csv")).to_dict("records")
    comparison_path = comparison_cache_path(case_out)
    if os.path.exists(comparison_path):
        comparison_rows = pd.read_csv(comparison_path).to_dict("records")
    else:
        safe_print(f"[sink-aware] rebuild comparison cache -> {os.path.basename(case_out)}")
        with open(os.path.join(case_out, "selected_candidate.json"), "r", encoding="utf-8") as f:
            best = json.load(f)
        met_test = None
        summary_path = os.path.join(case_out, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                met_test = dict(json.load(f).get("test_metrics", {}) or {})
        comparison_rows = build_comparison_rows(rec, failure, step5_root, best, met_test=met_test or None)
        pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    return {"frontier_rows": frontier_rows, "comparison_rows": comparison_rows}


def run_case(rec: Dict[str, object], failure: pd.DataFrame, step5_root: str, out_root: str) -> Dict[str, List[Dict[str, object]]]:
    dataset = str(rec["dataset"])
    holdout = str(rec["holdout_family"])
    rec_profile = str(rec.get("model_profile", "") or "")
    run_dir = step5_case_run_dir(step5_root, dataset, holdout, rec_profile)
    safe_print(f"[sink-aware] load scores -> {dataset} / {holdout} / {rec_profile}")
    val_scores = load_scores(run_dir, "val_scores.csv.gz")
    test_scores = load_scores(run_dir, "test_scores.csv.gz")
    selected_methods = pd.read_csv(os.path.join(run_dir, "selected_methods.csv"))
    control = selected_methods.loc[(selected_methods["split"] == "val") & (selected_methods["method"] == "control_tau")].iloc[0]
    control_test = selected_methods.loc[(selected_methods["split"] == "test") & (selected_methods["method"] == "control_tau")].iloc[0]
    control_params = parse_json_dict(control["selection_param_json"])
    global_tau = float(control_params["tau"])
    thr_high = float(control["thr_high"])
    split_variant = str(control["split_variant"])
    model_family = str(control["model_family"])
    model_profile = str(control.get("model_profile", model_family))
    stage1_weight_mode = str(control["stage1_weight_mode"])
    stage2_weight_mode = str(control.get("stage2_weight_mode", "unknown"))
    sink_family = str(failure.loc[(failure["dataset"] == dataset) & (failure["holdout_family"] == holdout), "top_sink_family"].iloc[0])
    unknown_label = "Unknown"
    known_families = sorted(set(val_scores["fam_pred_family"].astype(str).tolist()) | set(test_scores["fam_pred_family"].astype(str).tolist()))
    known_families = [fam for fam in known_families if fam not in {"Benign", unknown_label}]

    sink_vals = pd.to_numeric(
        val_scores.loc[val_scores["fam_pred_family"].astype(str) == sink_family, "fam_pmax"], errors="coerce"
    ).dropna()
    sink_tau_grid = [global_tau]
    if not sink_vals.empty:
        sink_tau_grid.extend([max(global_tau, float(np.quantile(sink_vals.to_numpy(dtype=np.float64), q))) for q in CFG["sink_tau_quantiles"]])

    frontier_rows: List[Dict[str, object]] = []
    best = None
    safe_print(f"[sink-aware] search frontier -> {dataset} / {holdout} / {rec_profile}")
    for sink_tau in sorted(set(sink_tau_grid)):
        for margin_floor in [float(x) for x in CFG["margin_grid"]]:
            pred_val, reject_val = predict_sink_aware(val_scores, thr_high, global_tau, sink_family, float(sink_tau), float(margin_floor), unknown_label)
            met = system_metrics(val_scores["y_true_sys"].astype(str).to_numpy(dtype=object), pred_val, reject_val, known_families, unknown_label)
            ok = (
                float(met["benign_family_fp_rate"]) <= 0.02 + 1e-12
                and float(met["benign_reject_rate"]) <= 0.10 + 1e-12
                and float(met["overall_reject_rate"]) <= 0.10 + 1e-12
                and float(met["false_unknown_rate_all_known"]) <= 0.05 + 1e-12
                and float(met["false_unknown_rate_known_attacks"]) <= 0.10 + 1e-12
            )
            row = {
                "dataset": dataset,
                "holdout_family": holdout,
                "split_variant": split_variant,
                "model_family": model_family,
                "model_profile": model_profile,
                "stage1_weight_mode": stage1_weight_mode,
                "stage2_weight_mode": stage2_weight_mode,
                "sink_family": sink_family,
                "global_tau": global_tau,
                "sink_tau": float(sink_tau),
                "margin_floor": float(margin_floor),
                "ok": bool(ok),
                **met,
            }
            frontier_rows.append(row)
            if not ok:
                continue
            if best is None or float(row["unknown_detection_rate"]) > float(best["unknown_detection_rate"]) or (float(row["unknown_detection_rate"]) == float(best["unknown_detection_rate"]) and float(row["macro_f1"]) > float(best["macro_f1"])):
                best = dict(row)

    if best is None:
        case_frontier = pd.DataFrame(frontier_rows)
        best = case_frontier.sort_values(["unknown_detection_rate", "macro_f1"], ascending=[False, False]).iloc[0].to_dict()

    case_out = case_run_dir(out_root, dataset, holdout, model_profile)
    safe_mkdir(case_out)
    pd.DataFrame(frontier_rows).to_csv(os.path.join(case_out, "frontier_table.csv"), index=False)
    with open(os.path.join(case_out, "selected_candidate.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    safe_print(f"[sink-aware] test/bootstrap -> {dataset} / {holdout} / {model_profile}")
    pred_test, reject_test = predict_sink_aware(test_scores, thr_high, global_tau, sink_family, float(best["sink_tau"]), float(best["margin_floor"]), unknown_label)
    met_test = system_metrics(test_scores["y_true_sys"].astype(str).to_numpy(dtype=object), pred_test, reject_test, known_families, unknown_label)
    with open(os.path.join(case_out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"selected_candidate": best, "test_metrics": met_test}, f, indent=2)

    comparison_rows: List[Dict[str, object]] = []
    comparison_rows.append({
        "dataset": dataset,
        "holdout_family": holdout,
        "split_variant": split_variant,
        "model_family": model_family,
        "model_profile": model_profile,
        "stage1_weight_mode": stage1_weight_mode,
        "stage2_weight_mode": stage2_weight_mode,
        "method": "control_tau",
        "sink_family": sink_family,
        "global_tau": global_tau,
        "sink_tau": np.nan,
        "margin_floor": np.nan,
        "macro_f1": float(control_test["macro_f1"]),
        "accuracy": float(control_test["accuracy"]),
        "unknown_detection_rate": float(control_test["unknown_detection_rate"]),
        "false_unknown_rate_all_known": float(control_test["false_unknown_rate_all_known"]),
        "false_unknown_rate_known_attacks": float(control_test["false_unknown_rate_known_attacks"]),
        "benign_family_fp_rate": float(control_test["benign_family_fp_rate"]),
        "overall_reject_rate": float(control_test["overall_reject_rate"]),
    })
    comparison_rows.append({
        "dataset": dataset,
        "holdout_family": holdout,
        "split_variant": split_variant,
        "model_family": model_family,
        "model_profile": model_profile,
        "stage1_weight_mode": stage1_weight_mode,
        "stage2_weight_mode": stage2_weight_mode,
        "method": "sink_aware_reject",
        "sink_family": sink_family,
        "global_tau": global_tau,
        "sink_tau": float(best["sink_tau"]),
        "margin_floor": float(best["margin_floor"]),
        **met_test,
    })

    control_pred, control_reject = predict_sink_aware(test_scores, thr_high, global_tau, sink_family, global_tau, -1.0, unknown_label)
    diff = bootstrap_diff(
        test_scores["y_true_sys"].astype(str).to_numpy(dtype=object),
        pred_test,
        reject_test,
        control_pred,
        control_reject,
        known_families,
        unknown_label,
        int(CFG["random_seed"]),
        int(CFG["bootstrap_resamples"]),
    )
    comparison_rows.append({
        "dataset": dataset,
        "holdout_family": holdout,
        "split_variant": split_variant,
        "model_family": model_family,
        "model_profile": model_profile,
        "stage1_weight_mode": stage1_weight_mode,
        "stage2_weight_mode": stage2_weight_mode,
        "method": "sink_aware_delta_vs_control",
        "sink_family": sink_family,
        **diff,
    })
    pd.DataFrame(comparison_rows).to_csv(comparison_cache_path(case_out), index=False)
    return {"frontier_rows": frontier_rows, "comparison_rows": comparison_rows}


def process_case(case_idx: int, rec: Dict[str, object], failure: pd.DataFrame, step5_root: str, out_root: str) -> Dict[str, List[Dict[str, object]]]:
    dataset = str(rec["dataset"])
    holdout = str(rec["holdout_family"])
    model_profile = str(rec.get("model_profile", "") or "")
    case_out = case_run_dir(out_root, dataset, holdout, model_profile)
    run_name = os.path.basename(case_out)
    error_path = os.path.join(case_out, "error.json")
    try:
        if os.path.exists(error_path):
            os.remove(error_path)
        if case_is_complete(case_out):
            safe_print(f"[sink-aware {case_idx}] skip completed -> {run_name}")
            return load_completed_case(rec, failure, step5_root, out_root)
        safe_print(f"[sink-aware {case_idx}] run -> {run_name}")
        result = run_case(rec, failure, step5_root, out_root)
        safe_print(f"[sink-aware {case_idx}] done -> {run_name}")
        return result
    except Exception as exc:
        safe_mkdir(case_out)
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "case_idx": int(case_idx),
                    "dataset": dataset,
                    "holdout_family": holdout,
                    "model_profile": model_profile,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                f,
                indent=2,
            )
        safe_print(f"[sink-aware {case_idx}] ERROR -> {run_name}: {type(exc).__name__}: {exc}")
        raise


def main() -> None:
    apply_env_overrides()
    step5_root = resolve_repo_path(str(CFG["step5_root"]))
    out_root = resolve_repo_path(str(CFG["runs_root"]))
    summary_root = os.path.join(out_root, str(CFG["summary_dirname"]))
    safe_mkdir(out_root)
    safe_mkdir(summary_root)

    failure = pd.read_csv(resolve_repo_path(str(CFG["failure_modes_csv"])))
    selected = {str(x) for x in list(CFG["selected_cases"])}
    step5_plan = load_step5_case_records(step5_root)
    failure_keys = {
        normalize_case_key(str(row["dataset"]), str(row["holdout_family"]))
        for row in failure[["dataset", "holdout_family"]].drop_duplicates().to_dict("records")
    }
    exec_cfg = resolve_execution_settings(len(selected) if selected else len(step5_plan))
    with open(os.path.join(out_root, "execution_settings.json"), "w", encoding="utf-8") as f:
        json.dump(exec_cfg, f, indent=2)

    frontier_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []

    payloads = []
    for rec in step5_plan.to_dict("records"):
        dataset = str(rec["dataset"])
        holdout = str(rec["holdout_family"])
        case_key = normalize_case_key(dataset, holdout)
        if selected and case_key not in selected:
            continue
        if case_key not in failure_keys:
            safe_print(f"[skip] no sink failure-mode row for {case_key}")
            continue
        payloads.append(rec)
    payloads.sort(
        key=lambda rec: (
            case_is_complete(
                case_run_dir(
                    out_root,
                    str(rec["dataset"]),
                    str(rec["holdout_family"]),
                    str(rec.get("model_profile", "") or ""),
                )
            ),
            str(rec["dataset"]),
            str(rec["holdout_family"]),
            str(rec.get("model_profile", "") or ""),
        )
    )

    results: List[Dict[str, List[Dict[str, object]]]] = []
    indexed_payloads = list(enumerate(payloads, start=1))
    safe_print(
        "[sink-aware] planned cases="
        f"{len(indexed_payloads)} workers={int(exec_cfg['case_parallel_workers'])} "
        f"backend={exec_cfg.get('parallel_backend', 'process')}"
    )
    if int(exec_cfg["case_parallel_workers"]) <= 1:
        for case_idx, rec in indexed_payloads:
            results.append(process_case(case_idx, rec, failure, step5_root, out_root))
    elif str(exec_cfg.get("parallel_backend", "process")).strip().lower() == "thread":
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(exec_cfg["case_parallel_workers"])) as pool:
            futures = [
                pool.submit(process_case, case_idx, rec, failure, step5_root, out_root)
                for case_idx, rec in indexed_payloads
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(exec_cfg["case_parallel_workers"]),
            mp_context=mp_ctx,
        ) as pool:
            futures = [
                pool.submit(process_case, case_idx, rec, failure, step5_root, out_root)
                for case_idx, rec in indexed_payloads
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    for result in results:
        frontier_rows.extend(result["frontier_rows"])
        comparison_rows.extend(result["comparison_rows"])

    frontier_df = pd.DataFrame(frontier_rows)
    frontier_df.to_csv(os.path.join(out_root, "aggregate_results.csv"), index=False)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(summary_root, "sink_aware_comparison.csv"), index=False)
    frontier_df.to_csv(os.path.join(summary_root, "sink_aware_frontier.csv"), index=False)
    safe_print(f"[sink-aware] wrote aggregate -> {os.path.join(out_root, 'aggregate_results.csv')}")


if __name__ == "__main__":
    main()
