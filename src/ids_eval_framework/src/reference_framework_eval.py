"""Reference-paper model profiles evaluated through the thesis framework."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from ids_eval_framework.src import two_stage_engine as engine
from ids_eval_framework.src.native_runtime import run_native_main
from ids_eval_framework.src.paths import resolve_repo_path


PROFILE_TO_PAPER = {
    "adewole2025_xgb_profile": "adewole2025_xgb",
    "neto2023_rf_profile": "neto2023_rf",
}


DEFAULT_PROFILE_CONFIG: dict[str, dict[str, Any]] = {
    "adewole2025_xgb_profile": {
        "paper": "adewole2025_xgb",
        "model_family": "xgb",
        "stage1_weight_modes": ["none"],
        "stage2_weight_modes": ["none"],
        "stage1_params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.3,
            "subsample": 0.5,
        },
        "stage2_params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.3,
            "subsample": 0.5,
        },
    },
    "neto2023_rf_profile": {
        "paper": "neto2023_rf",
        "model_family": "rf",
        "stage1_weight_modes": ["none"],
        "stage2_weight_modes": ["none"],
        "stage1_params": {
            "n_estimators": 100,
        },
        "stage2_params": {
            "n_estimators": 100,
        },
    },
}


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_codes_path(path_text: str | os.PathLike[str]) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(resolve_repo_path(path))


def ref_cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict((config or {}).get("reference_framework_eval", {}) or {})


def configured_profiles(config: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    cfg = ref_cfg(config)
    profiles = deepcopy(DEFAULT_PROFILE_CONFIG)
    for name, override in dict(cfg.get("profiles", {}) or {}).items():
        base = dict(profiles.get(name, {}))
        base.update(dict(override or {}))
        profiles[name] = base
    selected = cfg.get("enabled_profiles")
    if selected:
        keep = {str(x) for x in selected}
        profiles = {k: v for k, v in profiles.items() if k in keep}
    return profiles


def out_root(config: Mapping[str, Any] | None, *, smoke: bool = False) -> Path:
    cfg = ref_cfg(config)
    if smoke:
        smoke_root = ((cfg.get("smoke", {}) or {}).get("out_root"))
        if smoke_root:
            return resolve_codes_path(smoke_root)
    return resolve_codes_path(cfg.get("out_root", "ids_eval_framework/outputs/11_reference_framework_eval"))


PROTOCOL_A_COMPLETION_ARTIFACTS = (
    "reference_profile_metadata.json",
    "metrics_stage1_test.json",
    "metrics_stage2_test.json",
    "system_compare_test.json",
    "abstain_selected.json",
    "stage1_best.joblib",
    "stage2_best.joblib",
    "preprocessor.joblib",
)


def protocol_a_row_caps(config: Mapping[str, Any] | None, *, smoke: bool = False) -> tuple[int, int, int]:
    cfg = ref_cfg(config)
    proto_cfg = dict(cfg.get("protocol_a", {}) or {})
    smoke_cfg = dict((cfg.get("smoke", {}) or {}).get("protocol_a", {}) or {}) if smoke else {}
    max_train = int(smoke_cfg.get("max_train_rows", proto_cfg.get("max_train_rows", 1_000_000)))
    max_val = int(smoke_cfg.get("max_val_rows", proto_cfg.get("max_val_rows", 600_000)))
    max_test = int(smoke_cfg.get("max_test_rows", proto_cfg.get("max_test_rows", 1_000_000)))
    return max_train, max_val, max_test


def protocol_a_run_is_complete(run_dir: Path) -> bool:
    return run_dir.is_dir() and all((run_dir / name).exists() for name in PROTOCOL_A_COMPLETION_ARTIFACTS)


def metadata_int_matches(metadata: Mapping[str, Any], key: str, expected: int) -> bool:
    if key not in metadata or metadata.get(key) in (None, ""):
        return True
    try:
        return int(metadata[key]) == int(expected)
    except (TypeError, ValueError):
        return False


def metadata_matches_protocol_a_run(
    metadata: Mapping[str, Any],
    profile_name: str,
    profile: Mapping[str, Any],
    dataset: str,
    max_train: int,
    max_val: int,
    max_test: int,
) -> bool:
    if str(metadata.get("model_profile", "")) != str(profile_name):
        return False
    if str(metadata.get("dataset", "")) != str(dataset):
        return False
    if str(metadata.get("model_family", "")) != str(profile.get("model_family", "")):
        return False
    return (
        metadata_int_matches(metadata, "n_train_row_cap", max_train)
        and metadata_int_matches(metadata, "n_val_row_cap", max_val)
        and metadata_int_matches(metadata, "n_test_row_cap", max_test)
    )


def completed_protocol_a_run(
    config: Mapping[str, Any] | None,
    profile_name: str,
    profile: Mapping[str, Any],
    dataset: str,
    *,
    smoke: bool = False,
) -> dict[str, Any] | None:
    root = out_root(config, smoke=smoke) / "protocol_a"
    if not root.exists():
        return None
    max_train, max_val, max_test = protocol_a_row_caps(config, smoke=smoke)
    candidates = [p for p in root.glob(f"{profile_name}__{dataset}__*") if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        if not protocol_a_run_is_complete(run_dir):
            continue
        metadata = read_json(run_dir / "reference_profile_metadata.json")
        if metadata_matches_protocol_a_run(metadata, profile_name, profile, dataset, max_train, max_val, max_test):
            out = dict(metadata)
            out["run_dir"] = str(run_dir)
            return out
    return None


def build_model(profile_name: str, profile: Mapping[str, Any], stage: str, n_classes: int, seed: int, n_jobs: int):
    family = str(profile["model_family"])
    params = dict(profile.get(f"{stage}_params", {}) or {})
    if family == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is required for adewole2025_xgb_profile.")
        base = {
            "random_state": int(seed),
            "n_jobs": int(n_jobs),
            "tree_method": "hist",
            "verbosity": 0,
        }
        base.update(params)
        if stage == "stage1":
            base.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            base.update({"objective": "multi:softprob", "num_class": int(n_classes), "eval_metric": "mlogloss"})
        return XGBClassifier(**{k: v for k, v in base.items() if v is not None})
    if family == "rf":
        base = {
            "n_estimators": 100,
            "random_state": int(seed),
            "n_jobs": int(n_jobs),
            "class_weight": None,
        }
        base.update(params)
        return RandomForestClassifier(**base)
    raise ValueError(f"Unsupported model_family for {profile_name}: {family}")


def system_truth(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return np.array(["Benign" if int(a) == 0 else str(b) for a, b in zip(y1, y2)], dtype=object)


def fit_protocol_a_reference_profile(
    profile_name: str,
    profile: Mapping[str, Any],
    dataset: str,
    config: Mapping[str, Any],
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    cfg = ref_cfg(config)
    proto_cfg = dict(cfg.get("protocol_a", {}) or {})
    root = out_root(config, smoke=smoke) / "protocol_a"
    safe_mkdir(root)

    processed_root = resolve_codes_path(proto_cfg.get("processed_root", "ids_eval_framework/outputs/02_prepared_data/processed_V5"))
    protocol = str(proto_cfg.get("protocol", "A_stratified"))
    ds_dir = processed_root / protocol / dataset
    if not ds_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {ds_dir}")

    run_dir = safe_mkdir(root / f"{profile_name}__{dataset}__{now_stamp()}")
    overrides = {
        "processed_root": str(processed_root),
        "protocol": protocol,
        "datasets": [dataset],
        "runs_root": str(root),
        "chunksize_rows": int(proto_cfg.get("chunksize_rows", 200_000)),
        "stage1_threshold": dict(proto_cfg.get("stage1_threshold", {}) or {}),
        "cascade_gate": dict(proto_cfg.get("cascade_gate", {}) or {}),
        "abstain": dict(proto_cfg.get("abstain", {}) or {}),
        "calibration_reporting": dict(proto_cfg.get("calibration_reporting", {}) or {}),
    }
    engine.configure_for_protocol("A", overrides=overrides)

    seed = int(proto_cfg.get("seed", 123))
    n_jobs = int(proto_cfg.get("n_jobs", 8))
    max_train, max_val, max_test = protocol_a_row_caps(config, smoke=smoke)
    engine.CFG["stage1_xgb_grid"]["max_train_rows"] = {dataset: max_train}
    engine.CFG["stage1_xgb_grid"]["max_val_rows"] = {dataset: max_val}
    engine.CFG["stage2_xgb_grid"]["max_train_rows"] = {dataset: max_train}
    engine.CFG["stage2_xgb_grid"]["max_val_rows"] = {dataset: max_val}

    prep = engine.fit_preprocessor(str(ds_dir), str(run_dir))
    parts_train = engine.list_parts(str(ds_dir), "train")
    parts_val = engine.list_parts(str(ds_dir), "val")
    if not parts_train or not parts_val:
        raise RuntimeError(f"Missing train/val parts under {ds_dir}")

    X1_train, y1_train, y2_train = engine.collect_xy(
        parts_train,
        prep,
        engine.CFG["y_stage1"],
        max_train,
        seed=seed,
        filter_attack=None,
        y2_col=engine.CFG["y_stage2"],
    )
    X_val, y1_val, y2_val = engine.collect_xy(
        parts_val,
        prep,
        engine.CFG["y_stage1"],
        max_val,
        seed=seed + 1,
        filter_attack=None,
        y2_col=engine.CFG["y_stage2"],
    )
    if y2_train is None or y2_val is None:
        raise RuntimeError("Stage labels were not loaded for Protocol A reference profile.")

    stage1 = build_model(profile_name, profile, "stage1", 2, seed, n_jobs)
    stage1.fit(X1_train, y1_train)
    p_val_raw = stage1.predict_proba(X_val)[:, 1].astype(np.float64)
    platt = engine.fit_platt_on_probs(p_val_raw, y1_val)
    engine.safe_joblib_dump(stage1, str(run_dir / "stage1_best.joblib"))
    with (run_dir / "stage1_platt.json").open("w", encoding="utf-8") as f:
        json.dump({"platt": platt, "profile": profile_name}, f, indent=2)

    X2_train, _, y2_attack_train = engine.collect_xy(
        parts_train,
        prep,
        engine.CFG["y_stage1"],
        max_train,
        seed=seed + 2,
        filter_attack=True,
        y2_col=engine.CFG["y_stage2"],
    )
    X2_val, _, y2_attack_val = engine.collect_xy(
        parts_val,
        prep,
        engine.CFG["y_stage1"],
        max_val,
        seed=seed + 3,
        filter_attack=True,
        y2_col=engine.CFG["y_stage2"],
    )
    if y2_attack_train is None or len(y2_attack_train) == 0:
        raise RuntimeError("No attack rows for Protocol A Stage 2 reference training.")
    families = sorted({str(x) for x in y2_attack_train if str(x) and str(x).lower() != "nan"})
    fam_to_idx = {fam: i for i, fam in enumerate(families)}
    y2_idx = np.array([fam_to_idx[str(x)] for x in y2_attack_train], dtype=int)
    stage2 = build_model(profile_name, profile, "stage2", len(families), seed + 10, n_jobs)
    stage2.fit(X2_train, y2_idx)

    mask_val = np.array([str(x) in fam_to_idx for x in y2_attack_val], dtype=bool)
    X2_val_ok = X2_val[mask_val]
    y2_val_idx = np.array([fam_to_idx[str(x)] for x in y2_attack_val[mask_val]], dtype=int)
    p2_val = stage2.predict_proba(X2_val_ok).astype(np.float64)
    temp = engine.fit_temperature_on_probs(p2_val, y2_val_idx, len(families))
    engine.safe_joblib_dump(stage2, str(run_dir / "stage2_best.joblib"))
    with (run_dir / "families.json").open("w", encoding="utf-8") as f:
        json.dump(families, f, indent=2)
    with (run_dir / "stage2_temperature.json").open("w", encoding="utf-8") as f:
        json.dump({"T": float(temp), "profile": profile_name, "K": len(families)}, f, indent=2)

    p_attack_val = engine.apply_platt(p_val_raw, platt)
    p2_all_val = engine.apply_temperature(stage2.predict_proba(X_val).astype(np.float64), temp)
    fam_pred_val = np.argmax(p2_all_val, axis=1).astype(int)
    fam_pmax_val = np.max(p2_all_val, axis=1).astype(np.float64)
    ysys_val = system_truth(y1_val, y2_val)

    threshold_cfg = engine.CFG["stage1_threshold"]
    if threshold_cfg.get("policy", "target_fpr_family_aware") == "target_fpr_family_aware":
        thr_high, meta = engine.choose_thr_high_family_aware(
            y1_val,
            y2_val,
            p_attack_val,
            float(threshold_cfg.get("target_fpr", 0.01)),
            int(threshold_cfg.get("min_family_support", 50)),
            int(threshold_cfg.get("sweep_points", 60)),
            str(threshold_cfg.get("objective_mode", "min_family_recall")),
            float(threshold_cfg.get("p10_quantile", 0.10)),
        )
        with (run_dir / "stage1_threshold_strict_family_aware_sweep.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    else:
        thr_high = engine.pick_threshold_target_fpr(y1_val, p_attack_val, float(threshold_cfg.get("target_fpr", 0.01)))

    with (run_dir / "stage1_threshold_strict.json").open("w", encoding="utf-8") as f:
        json.dump({"thr_high": float(thr_high), "policy": threshold_cfg, "val_rows_used": int(len(y1_val))}, f, indent=2)

    thr_low, tau_cascade = engine.tune_cascade_thr_low_and_tau(
        str(run_dir),
        y1_val,
        ysys_val,
        p_attack_val,
        fam_pred_val,
        fam_pmax_val,
        families,
        float(thr_high),
    )
    tau_strict = engine.pick_tau_strict(
        str(run_dir),
        ysys_val,
        p_attack_val,
        fam_pred_val,
        fam_pmax_val,
        families,
        float(thr_high),
    )
    with (run_dir / "abstain_selected.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": engine.CFG["abstain"].get("policy", "reject_to_benign"),
                "label": engine.CFG["abstain"]["label"],
                "thr_high": float(thr_high),
                "thr_low": float(thr_low),
                "tau_strict": float(tau_strict),
                "tau_cascade": float(tau_cascade),
                "profile": profile_name,
            },
            f,
            indent=2,
        )

    engine.evaluate_system(str(ds_dir), str(run_dir), "val", prep, stage1, platt, stage2, families, temp, thr_high, thr_low, tau_strict, tau_cascade)
    engine.evaluate_system(str(ds_dir), str(run_dir), "test", prep, stage1, platt, stage2, families, temp, thr_high, thr_low, tau_strict, tau_cascade)

    summary = summarize_protocol_a_run(run_dir, profile_name, profile, dataset, max_train, max_val, max_test)
    with (run_dir / "reference_profile_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def summarize_protocol_a_run(
    run_dir: Path,
    profile_name: str,
    profile: Mapping[str, Any],
    dataset: str,
    max_train: int,
    max_val: int,
    max_test: int,
) -> dict[str, Any]:
    stage1 = read_json(run_dir / "metrics_stage1_test.json")
    stage2 = read_json(run_dir / "metrics_stage2_test.json")
    system = read_json(run_dir / "system_compare_test.json")
    strict_tau = dict(system.get("strict_tau", {}) or {})
    cascade_tau = dict(system.get("cascade_tau", {}) or {})
    return {
        "surface": "protocol_a_reference_profile",
        "claim_status": "descriptive_reference_profile_full_framework",
        "paper": str(profile.get("paper", PROFILE_TO_PAPER.get(profile_name, profile_name))),
        "model_profile": profile_name,
        "model_family": str(profile["model_family"]),
        "dataset": dataset,
        "run_dir": str(run_dir),
        "n_train_row_cap": int(max_train),
        "n_val_row_cap": int(max_val),
        "n_test_row_cap": int(max_test),
        "system_variant": "strict_tau",
        "system_accuracy": strict_tau.get("accuracy"),
        "system_macro_f1_supported_labels": strict_tau.get("system_macro_f1_supported_labels", strict_tau.get("macro_f1")),
        "system_macro_f1_declared_output_labels_historical": strict_tau.get(
            "system_macro_f1_declared_output_labels_historical"
        ),
        "system_benign_family_fp_rate": strict_tau.get("benign_family_fp_rate"),
        "system_overall_reject_rate": strict_tau.get("overall_reject_rate"),
        "cascade_tau_accuracy": cascade_tau.get("accuracy"),
        "cascade_tau_macro_f1_supported_labels": cascade_tau.get(
            "system_macro_f1_supported_labels", cascade_tau.get("macro_f1")
        ),
        "cascade_tau_macro_f1_declared_output_labels_historical": cascade_tau.get(
            "system_macro_f1_declared_output_labels_historical"
        ),
        "stage1_auc": stage1.get("roc_auc"),
        "stage1_fpr": stage1.get("fpr"),
        "stage1_tpr": stage1.get("tpr"),
        "stage1_threshold": stage1.get("threshold"),
        "stage2_macro_f1_fixedK": stage2.get("macro_f1_fixedK"),
        "stage2_macro_f1_present": stage2.get("macro_f1_present"),
        "stage2_accuracy": stage2.get("accuracy"),
    }


def run_protocol_a_reference_profiles(
    config: Mapping[str, Any] | None,
    *,
    dry_run: bool = False,
    smoke: bool = False,
    resume: bool = True,
    profiles: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
) -> pd.DataFrame:
    cfg = ref_cfg(config)
    proto_cfg = dict(cfg.get("protocol_a", {}) or {})
    profile_map = configured_profiles(config)
    selected_profiles = [p for p in profile_map if profiles is None or p in set(profiles)]
    rows: list[dict[str, Any]] = []
    for profile_name in selected_profiles:
        profile = profile_map[profile_name]
        profile_datasets = list((proto_cfg.get("datasets_by_profile", {}) or {}).get(profile_name, proto_cfg.get("datasets", [])))
        if not profile_datasets:
            profile_datasets = ["CICIoT2023"] if str(profile.get("paper")) == "neto2023_rf" else ["CICIDS2017", "CICIoT2023"]
        for dataset in profile_datasets:
            if datasets is not None and dataset not in set(datasets):
                continue
            if dry_run:
                rows.append(
                    {
                        "surface": "protocol_a_reference_profile",
                        "model_profile": profile_name,
                        "paper": profile.get("paper"),
                        "dataset": dataset,
                        "planned": True,
                    }
                )
            else:
                if resume:
                    completed = completed_protocol_a_run(config, profile_name, profile, dataset, smoke=smoke)
                    if completed is not None:
                        print(f"[resume] protocol_a skip completed -> {profile_name} / {dataset}: {completed['run_dir']}")
                        rows.append(completed)
                        continue
                rows.append(fit_protocol_a_reference_profile(profile_name, profile, dataset, config or {}, smoke=smoke))
    df = pd.DataFrame(rows)
    summary_dir = safe_mkdir(out_root(config, smoke=smoke) / "protocol_a" / "summary")
    df.to_csv(summary_dir / "protocol_a_reference_profile_summary.csv", index=False)
    return df


def profile_overrides_for_protocol_b(config: Mapping[str, Any] | None, selected: Sequence[str] | None = None) -> dict[str, Any]:
    cfg = ref_cfg(config)
    b_cfg = dict(cfg.get("protocol_b", {}) or {})
    datasets_by_profile = dict(b_cfg.get("datasets_by_profile", {}) or {})
    profiles = configured_profiles(config)
    if selected is not None:
        keep = set(selected)
        profiles = {k: v for k, v in profiles.items() if k in keep}
    out: dict[str, Any] = {}
    for name, profile in profiles.items():
        rec = {
            "paper": profile.get("paper", PROFILE_TO_PAPER.get(name, name)),
            "model_family": profile["model_family"],
            "stage1_weight_modes": list(profile.get("stage1_weight_modes", ["none"])),
            "stage2_weight_modes": list(profile.get("stage2_weight_modes", ["none"])),
            "stage1_param_grid": [dict(profile.get("stage1_params", {}) or {})],
            "stage2_param_grid": [dict(profile.get("stage2_params", {}) or {})],
        }
        if name in datasets_by_profile:
            rec["datasets"] = list(datasets_by_profile[name] or [])
        out[name] = rec
    return out


def protocol_b_overrides(config: Mapping[str, Any] | None, *, smoke: bool = False, profiles: Sequence[str] | None = None) -> dict[str, Any]:
    cfg = ref_cfg(config)
    b_cfg = dict(cfg.get("protocol_b", {}) or {})
    overrides = deepcopy(b_cfg.get("legacy_overrides", {}) or {})
    root = out_root(config, smoke=smoke)
    overrides["runs_root"] = str(root / "protocol_b")
    overrides["model_profiles"] = profile_overrides_for_protocol_b(config, profiles)
    overrides["model_families"] = []
    overrides["stage1_weight_modes"] = ["none"]
    overrides["stage2_weight_modes"] = ["none"]
    overrides.setdefault("xgb_binary_defaults", {})
    overrides.setdefault("xgb_multi_defaults", {})
    overrides["xgb_binary_defaults"].update({"tree_method": "hist", "device": None, "predictor": None, "verbosity": 0})
    overrides["xgb_multi_defaults"].update({"tree_method": "hist", "device": None, "predictor": None, "verbosity": 0})
    if smoke:
        smoke_cfg = dict((cfg.get("smoke", {}) or {}).get("protocol_b", {}) or {})
        for key in ("max_train_rows", "max_val_rows", "max_test_rows", "max_manifests", "n_jobs"):
            if key in smoke_cfg:
                overrides[key] = smoke_cfg[key]
    return overrides


def run_protocol_b_reference_profiles(
    config: Mapping[str, Any] | None,
    *,
    dry_run: bool = False,
    smoke: bool = False,
    profiles: Sequence[str] | None = None,
) -> None:
    from ids_eval_framework._native import protocol_b_grid, protocol_b_summary

    overrides = protocol_b_overrides(config, smoke=smoke, profiles=profiles)
    root = out_root(config, smoke=smoke)
    run_native_main(
        protocol_b_grid,
        cfg_overrides=overrides,
        dry_run=dry_run,
    )
    summary_overrides = {
        "aggregate_csv": str(root / "protocol_b" / "aggregate_results.csv"),
        "out_dir": str(root / "protocol_b" / "summary"),
        "baseline_best_per_holdout_csv": "",
        "best_per_holdout_group_cols": ["dataset", "holdout_family", "model_profile"],
    }
    run_native_main(
        protocol_b_summary,
        cfg_overrides=summary_overrides,
        dry_run=dry_run,
    )


def open_set_overrides(config: Mapping[str, Any] | None, *, smoke: bool = False) -> dict[str, Any]:
    cfg = ref_cfg(config)
    o_cfg = dict(cfg.get("open_set", {}) or {})
    root = out_root(config, smoke=smoke)
    overrides = deepcopy(o_cfg.get("legacy_overrides", {}) or {})
    overrides.update(
        {
            "best_csv": str(root / "protocol_b" / "summary" / "best_per_holdout.csv"),
            "runs_root": str(root / "open_set"),
            "grid_script": "ids_eval_framework._native.protocol_b_grid",
            "base_helper_script": "ids_eval_framework.src.two_stage_engine",
        }
    )
    # Threaded case parallelism is portable on Windows and keeps worker imports
    # within the installed package.
    overrides.setdefault("parallel_backend", "thread")
    overrides.setdefault("case_parallel_workers", 4)
    overrides.setdefault("threads_per_worker", 4)
    # Open-set replay builds CPU-backed pandas/NumPy matrices; keep XGBoost on CPU
    # to avoid CUDA booster vs CPU input fallback warnings and extra memory churn.
    overrides.setdefault("xgb_device", "cpu")
    if "audit_roots" not in overrides:
        overrides["audit_roots"] = [
            "ids_eval_framework/outputs/04_protocol_b_support_audit/protocolB_support_audit_out",
            "ids_eval_framework/outputs/04_protocol_b_support_audit/protocolB_support_audit_out_cicids17_recovery",
        ]
    if smoke:
        smoke_cfg = dict((cfg.get("smoke", {}) or {}).get("open_set", {}) or {})
        overrides.update(smoke_cfg)
    return overrides


def run_open_set_reference_profiles(config: Mapping[str, Any] | None, *, dry_run: bool = False, smoke: bool = False) -> None:
    from ids_eval_framework._native import protocol_b_open_set

    run_native_main(
        protocol_b_open_set,
        cfg_overrides=open_set_overrides(config, smoke=smoke),
        dry_run=dry_run,
    )


def sink_aware_overrides(config: Mapping[str, Any] | None, *, smoke: bool = False) -> dict[str, Any]:
    cfg = ref_cfg(config)
    s_cfg = dict(cfg.get("sink_aware", {}) or {})
    root = out_root(config, smoke=smoke)
    overrides = deepcopy(s_cfg.get("legacy_overrides", {}) or {})
    overrides.update(
        {
            "step5_root": str(root / "open_set"),
            "runs_root": str(root / "sink_aware"),
        }
    )
    # Sink-aware replay is resume-aware; use enough threaded workers to overlap the
    # I/O-heavy score loads and bootstrap/cache rebuilds on high-core workstations.
    overrides.setdefault("parallel_backend", "thread")
    overrides.setdefault("case_parallel_workers", 6)
    overrides.setdefault(
        "failure_modes_csv",
        "ids_eval_framework/outputs/supplementary/exploratory/thesis_full_scope_pack/protocol_b_failure_modes.csv",
    )
    if smoke:
        smoke_cfg = dict((cfg.get("smoke", {}) or {}).get("sink_aware", {}) or {})
        overrides.update(smoke_cfg)
    return overrides


def run_sink_aware_reference_profiles(config: Mapping[str, Any] | None, *, dry_run: bool = False, smoke: bool = False) -> None:
    from ids_eval_framework._native import protocol_b_sink_aware

    failure_path = resolve_codes_path(sink_aware_overrides(config, smoke=smoke).get("failure_modes_csv", ""))
    if not dry_run and not failure_path.exists():
        print(f"[skip] sink-aware failure modes not found: {failure_path}")
        return
    run_native_main(
        protocol_b_sink_aware,
        cfg_overrides=sink_aware_overrides(config, smoke=smoke),
        dry_run=dry_run,
    )


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def add_surface(df: pd.DataFrame, surface: str, claim_status: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["surface"] = surface
    out["claim_status"] = claim_status
    return out


def paper_for_profile(profile: object) -> str:
    return PROFILE_TO_PAPER.get(str(profile), str(profile))


def comparison_sources(config: Mapping[str, Any] | None, *, smoke: bool = False) -> dict[str, Path]:
    root = out_root(config, smoke=smoke)
    return {
        "protocol_a": root / "protocol_a" / "summary" / "protocol_a_reference_profile_summary.csv",
        "protocol_b": root / "protocol_b" / "summary" / "best_per_holdout.csv",
        "open_set": root / "open_set" / "summary" / "open_set_baseline_comparison.csv",
        "sink_aware": root / "sink_aware" / "summary" / "sink_aware_comparison.csv",
    }


def build_reference_framework_comparison(config: Mapping[str, Any] | None, *, smoke: bool = False) -> Path:
    root = out_root(config, smoke=smoke)
    comp_dir = safe_mkdir(root / "comparison")
    paths = comparison_sources(config, smoke=smoke)
    reference_values = list(ref_cfg(config).get("closed_set_reference_values", []) or [])
    paper = pd.DataFrame(reference_values)
    pa = safe_read_csv(paths["protocol_a"])
    pb = safe_read_csv(paths["protocol_b"])
    os_df = safe_read_csv(paths["open_set"])
    sink = safe_read_csv(paths["sink_aware"])

    if not paper.empty:
        paper = paper.copy()
        paper["surface"] = "published_closed_set_reference_value"
        paper["claim_status"] = "published_reference_value_not_reproduced_here"
    if not pb.empty and "paper" not in pb.columns:
        pb = pb.copy()
        pb["paper"] = pb.get("model_profile", "").map(paper_for_profile)
    if not os_df.empty and "paper" not in os_df.columns:
        os_df = os_df.copy()
        os_df["paper"] = os_df.get("model_profile", "").map(paper_for_profile)
    if not sink.empty and "paper" not in sink.columns:
        sink = sink.copy()
        sink["paper"] = sink.get("model_profile", "").map(paper_for_profile)

    rows: list[dict[str, Any]] = []
    for _, row in paper.iterrows() if not paper.empty else []:
        rows.append(
            {
                "surface": "published_closed_set_reference_value",
                "claim_status": row.get("claim_status"),
                "paper": row.get("paper"),
                "model_profile": "",
                "dataset": row.get("dataset"),
                "task_or_holdout": row.get("task"),
                "accuracy": row.get("accuracy"),
                "macro_f1": row.get("macro_f1"),
            }
        )
    for _, row in pa.iterrows() if not pa.empty else []:
        rows.append(
            {
                "surface": "protocol_a_reference_profile",
                "claim_status": row.get("claim_status"),
                "paper": row.get("paper"),
                "model_profile": row.get("model_profile"),
                "dataset": row.get("dataset"),
                "task_or_holdout": "closed_set_system",
                "accuracy": row.get("system_accuracy"),
                "macro_f1": row.get("system_macro_f1_supported_labels"),
                "macro_f1_declared_output_labels_historical": row.get(
                    "system_macro_f1_declared_output_labels_historical"
                ),
                "stage1_auc": row.get("stage1_auc"),
                "stage2_macro_f1": row.get("stage2_macro_f1_present"),
                "overall_reject_rate": row.get("system_overall_reject_rate"),
            }
        )
    for _, row in pb.iterrows() if not pb.empty else []:
        rows.append(
            {
                "surface": "protocol_b_reference_profile",
                "claim_status": "descriptive_reference_profile_full_framework",
                "paper": row.get("paper"),
                "model_profile": row.get("model_profile"),
                "dataset": row.get("dataset"),
                "task_or_holdout": row.get("holdout_family"),
                "accuracy": row.get("accuracy"),
                "macro_f1": row.get("macro_f1"),
                "stage1_auc": row.get("stage1_auc_val"),
                "stage2_macro_f1": row.get("stage2_macro_f1_val"),
                "unknown_detection_rate": row.get("unknown_detection_rate"),
                "false_unknown_rate_all_known": row.get("false_unknown_rate_all_known"),
                "benign_family_fp_rate": row.get("benign_family_fp_rate"),
                "overall_reject_rate": row.get("overall_reject_rate"),
            }
        )
    for _, row in os_df.iterrows() if not os_df.empty else []:
        if str(row.get("split", "test")) != "test":
            continue
        rows.append(
            {
                "surface": "open_set_reference_profile",
                "claim_status": "descriptive_reference_profile_open_set",
                "paper": row.get("paper"),
                "model_profile": row.get("model_profile"),
                "dataset": row.get("dataset"),
                "task_or_holdout": row.get("holdout_family"),
                "method": row.get("method"),
                "accuracy": row.get("accuracy"),
                "macro_f1": row.get("macro_f1"),
                "unknown_detection_rate": row.get("unknown_detection_rate"),
                "false_unknown_rate_all_known": row.get("false_unknown_rate_all_known"),
                "benign_family_fp_rate": row.get("benign_family_fp_rate"),
                "overall_reject_rate": row.get("overall_reject_rate"),
            }
        )
    for _, row in sink.iterrows() if not sink.empty else []:
        rows.append(
            {
                "surface": "sink_aware_reference_profile",
                "claim_status": "descriptive_reference_profile_sink_aware",
                "paper": row.get("paper"),
                "model_profile": row.get("model_profile"),
                "dataset": row.get("dataset"),
                "task_or_holdout": row.get("holdout_family"),
                "method": row.get("method"),
                "accuracy": row.get("accuracy"),
                "macro_f1": row.get("macro_f1"),
                "unknown_detection_rate": row.get("unknown_detection_rate"),
                "false_unknown_rate_all_known": row.get("false_unknown_rate_all_known"),
                "benign_family_fp_rate": row.get("benign_family_fp_rate"),
                "overall_reject_rate": row.get("overall_reject_rate"),
                "delta_macro_f1_mean": row.get("delta_macro_f1_mean"),
                "delta_unknown_detection_mean": row.get("delta_unknown_detection_mean"),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(comp_dir / "accuracy_vs_full_framework_summary.csv", index=False)

    drops: list[dict[str, Any]] = []
    if not summary.empty:
        closed = summary.loc[summary["surface"] == "published_closed_set_reference_value"].copy()
        full = summary.loc[summary["surface"].isin(["protocol_a_reference_profile", "protocol_b_reference_profile"])].copy()
        for _, row in full.iterrows():
            candidates = closed.loc[
                (closed["paper"].astype(str) == str(row.get("paper")))
                & (closed["dataset"].astype(str) == str(row.get("dataset")))
            ].copy()
            if not candidates.empty:
                candidates["_task_rank"] = candidates["task_or_holdout"].astype(str).map({"family": 0, "binary": 1}).fillna(2)
                base = candidates.sort_values("_task_rank").iloc[0]
                drops.append(
                    {
                        "paper": row.get("paper"),
                        "model_profile": row.get("model_profile"),
                        "dataset": row.get("dataset"),
                        "full_framework_surface": row.get("surface"),
                        "task_or_holdout": row.get("task_or_holdout"),
                        "closed_set_task_used": base.get("task_or_holdout"),
                        "closed_set_accuracy": base.get("accuracy"),
                        "full_framework_accuracy": row.get("accuracy"),
                        "accuracy_delta_full_minus_closed": pd.to_numeric(pd.Series([row.get("accuracy")]), errors="coerce").iloc[0]
                        - pd.to_numeric(pd.Series([base.get("accuracy")]), errors="coerce").iloc[0],
                        "closed_set_macro_f1": base.get("macro_f1"),
                        "full_framework_macro_f1_supported_labels": row.get("macro_f1"),
                        "full_framework_macro_f1_declared_output_labels_historical": row.get(
                            "macro_f1_declared_output_labels_historical"
                        ),
                    }
                )
    pd.DataFrame(drops).to_csv(comp_dir / "metric_drop_table.csv", index=False)

    stress_cols = [
        "paper",
        "model_profile",
        "dataset",
        "holdout_family",
        "unknown_detection_rate",
        "false_unknown_rate_all_known",
        "false_unknown_rate_known_attacks",
        "benign_family_fp_rate",
        "overall_reject_rate",
        "macro_f1",
        "accuracy",
        "stage1_auc_val",
        "stage2_macro_f1_val",
        "run_name",
        "run_dir",
    ]
    stress = pb.reindex(columns=[c for c in stress_cols if c in pb.columns]) if not pb.empty else pd.DataFrame(columns=stress_cols)
    stress.to_csv(comp_dir / "protocol_b_holdout_stress_table.csv", index=False)

    notes = [
        "# Reference Framework Interpretation Notes",
        "",
        "These outputs compare closed-set paper-style accuracy against full-framework reference-profile evaluation.",
        "",
        "- `paper_style_closed_set` is the framework-compatible reproduction-pack surface.",
        "- `protocol_a_reference_profile` and `protocol_b_reference_profile` use the paper model family/settings inside the thesis two-stage framework.",
        "- Open-set and sink-aware rows are framework stress tests, not claims that the original papers evaluated those protocols.",
        "- Metric drops are descriptive contrasts across evaluation surfaces, not byte-identical third-party reruns.",
    ]
    (comp_dir / "reference_framework_interpretation_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    with (comp_dir / "comparison_sources.json").open("w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in paths.items()}, f, indent=2, sort_keys=True)
    return comp_dir


def run_reference_framework_eval(
    config: Mapping[str, Any] | None,
    *,
    dry_run: bool = False,
    smoke: bool = False,
    resume: bool = True,
    profiles: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    skip_protocol_a: bool = False,
    skip_protocol_b: bool = False,
    skip_open_set: bool = False,
    skip_sink_aware: bool = False,
    skip_comparison: bool = False,
) -> None:
    root = safe_mkdir(out_root(config, smoke=smoke))
    print(f"[reference-framework] out_root={root}")
    if not skip_protocol_a:
        df = run_protocol_a_reference_profiles(
            config,
            dry_run=dry_run,
            smoke=smoke,
            resume=resume,
            profiles=profiles,
            datasets=datasets,
        )
        if dry_run:
            print(df.to_string(index=False))
    if not skip_protocol_b:
        run_protocol_b_reference_profiles(config, dry_run=dry_run, smoke=smoke, profiles=profiles)
    if not skip_open_set:
        run_open_set_reference_profiles(config, dry_run=dry_run, smoke=smoke)
    if not skip_sink_aware:
        run_sink_aware_reference_profiles(config, dry_run=dry_run, smoke=smoke)
    if not skip_comparison and not dry_run:
        build_reference_framework_comparison(config, smoke=smoke)
