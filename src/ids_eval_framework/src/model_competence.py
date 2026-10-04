"""Staged Protocol-A competence audit for strong conventional tabular learners.

The lane answers one narrow question: are the primary RF/XGB systems obviously weak
relative to ordinary strong alternatives under the same VERA tasks and operating logic?
It is intentionally not a state-of-the-art leaderboard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

from ids_eval_framework.src import two_stage_engine as engine
from ids_eval_framework.src.paths import resolve_repo_path


def competence_cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict((config or {}).get("model_competence", {}) or {})


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(resolve_repo_path(str(path)))


def stable_hash(value: Mapping[str, Any], n: int = 12) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:n]


def safe_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def system_truth(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return np.array(["Benign" if int(a) == 0 else str(b) for a, b in zip(y1, y2)], dtype=object)


def build_model(
    model_family: str,
    params: Mapping[str, Any],
    *,
    stage: str,
    n_classes: int,
    seed: int,
    n_jobs: int,
):
    p = dict(params)
    if model_family == "extra_trees":
        p.setdefault("n_jobs", int(n_jobs))
        p["random_state"] = int(seed)
        return ExtraTreesClassifier(**p)

    if model_family == "lgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is required for the model-competence lane.")
        p.setdefault("n_jobs", int(n_jobs))
        p.setdefault("verbosity", -1)
        p["random_state"] = int(seed)
        if stage == "stage1":
            p.setdefault("objective", "binary")
        else:
            p.setdefault("objective", "multiclass")
            p.setdefault("num_class", int(n_classes))
        return LGBMClassifier(**p)

    if model_family == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is required for the model-competence lane.")
        p.setdefault("thread_count", int(n_jobs))
        p.setdefault("verbose", False)
        p.setdefault("allow_writing_files", False)
        p["random_seed"] = int(seed)
        p.setdefault("loss_function", "Logloss" if stage == "stage1" else "MultiClass")
        return CatBoostClassifier(**p)

    raise ValueError(f"Unsupported competence model family: {model_family}")


def predict_multi_proba(model, X, n_classes: int) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", np.arange(proba.shape[1])), dtype=int)
    if proba.shape[1] == int(n_classes) and np.array_equal(classes, np.arange(n_classes)):
        out = proba
    else:
        out = np.zeros((proba.shape[0], int(n_classes)), dtype=np.float64)
        for column, cls in enumerate(classes):
            if 0 <= int(cls) < int(n_classes):
                out[:, int(cls)] = proba[:, column]
    out = np.clip(out, 1e-12, 1.0)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def _stage1_candidate_result(
    model,
    X_val,
    y1_val: np.ndarray,
    y2_val: np.ndarray,
    threshold_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    raw = np.asarray(model.predict_proba(X_val), dtype=np.float64)[:, 1]
    platt = engine.fit_platt_on_probs(raw, y1_val)
    calibrated = engine.apply_platt(raw, platt)
    thr, meta = engine.choose_thr_high_family_aware(
        y1_val,
        y2_val,
        calibrated,
        float(threshold_cfg.get("target_fpr", 0.015)),
        int(threshold_cfg.get("min_family_support", 50)),
        int(threshold_cfg.get("sweep_points", 60)),
        str(threshold_cfg.get("objective_mode", "min_family_recall")),
        float(threshold_cfg.get("p10_quantile", 0.10)),
    )
    best = dict(meta.get("best", {}) or {}) if isinstance(meta, dict) else {}
    if not best:
        pred = calibrated >= float(thr)
        recs, min_rec, p10 = engine.family_recall_stats(
            y1_val,
            y2_val,
            pred,
            int(threshold_cfg.get("min_family_support", 50)),
        )
        benign = y1_val == 0
        attacks = y1_val == 1
        best = {
            "objective": float(min_rec),
            "min_family_recall": float(min_rec),
            "p10_family_recall": float(p10),
            "fpr": float(np.mean(pred[benign])) if np.any(benign) else 0.0,
            "tpr": float(np.mean(pred[attacks])) if np.any(attacks) else 0.0,
            "n_families": len(recs),
        }
    return {
        "threshold": float(thr),
        "objective": float(best.get("objective", np.nan)),
        "tpr": float(best.get("tpr", np.nan)),
        "fpr": float(best.get("fpr", np.nan)),
        "min_family_recall": float(best.get("min_family_recall", np.nan)),
        "p10_family_recall": float(best.get("p10_family_recall", np.nan)),
    }, platt


def select_stage1(
    family: str,
    candidates: Sequence[Mapping[str, Any]],
    X_train,
    y_train: np.ndarray,
    X_val,
    y1_val: np.ndarray,
    y2_val: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
    threshold_cfg: Mapping[str, Any],
) -> tuple[Any, dict[str, float], dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], Any, dict[str, float], dict[str, Any]] | None = None
    for idx, params in enumerate(candidates):
        model = build_model(family, params, stage="stage1", n_classes=2, seed=seed, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        metrics, platt = _stage1_candidate_result(model, X_val, y1_val, y2_val, threshold_cfg)
        row = {
            "candidate_index": idx,
            "candidate_hash": stable_hash(params),
            "params": json.dumps(dict(params), sort_keys=True),
            **metrics,
        }
        rows.append(row)
        objective = float(metrics["objective"])
        tpr = float(metrics["tpr"])
        score = (
            objective if np.isfinite(objective) else -np.inf,
            tpr if np.isfinite(tpr) else -np.inf,
        )
        if best is None or score > best[0]:
            best = (score, model, platt, row)
    if best is None:
        raise RuntimeError(f"No Stage-1 candidates were evaluated for {family}.")
    return best[1], best[2], best[3], pd.DataFrame(rows)


def select_stage2(
    family: str,
    candidates: Sequence[Mapping[str, Any]],
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    *,
    n_classes: int,
    seed: int,
    n_jobs: int,
) -> tuple[Any, float, dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], Any, float, dict[str, Any]] | None = None
    for idx, params in enumerate(candidates):
        model = build_model(
            family,
            params,
            stage="stage2",
            n_classes=n_classes,
            seed=seed + 10,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)
        raw = predict_multi_proba(model, X_val, n_classes)
        temperature = float(engine.fit_temperature_on_probs(raw, y_val, n_classes))
        calibrated = engine.apply_temperature(raw, temperature)
        pred = np.argmax(calibrated, axis=1).astype(int)
        macro = float(engine.macro_f1_present(y_val, pred))
        nll = float(log_loss(y_val, calibrated, labels=list(range(n_classes))))
        row = {
            "candidate_index": idx,
            "candidate_hash": stable_hash(params),
            "params": json.dumps(dict(params), sort_keys=True),
            "val_macro_f1_present": macro,
            "val_nll": nll,
        }
        rows.append(row)
        score = (macro, -nll)
        if best is None or score > best[0]:
            best = (score, model, temperature, row)
    if best is None:
        raise RuntimeError(f"No Stage-2 candidates were evaluated for {family}.")
    return best[1], best[2], best[3], pd.DataFrame(rows)


def configure_engine(cfg: Mapping[str, Any], dataset: str, seed: int) -> None:
    overrides = {
        "processed_root": str(resolve_path(cfg["processed_root"])),
        "protocol": str(cfg.get("protocol", "A_stratified")),
        "datasets": [dataset],
        "chunksize_rows": int(cfg.get("chunksize_rows", 200_000)),
        "global_seed": int(seed),
        "stage1_threshold": dict(cfg.get("stage1_threshold", {}) or {}),
        "cascade_gate": dict(cfg.get("cascade_gate", {}) or {}),
        "abstain": dict(cfg.get("abstain", {}) or {}),
        "calibration_reporting": dict(cfg.get("calibration_reporting", {}) or {}),
    }
    engine.configure_for_protocol("A", overrides=overrides)


def summarize_run(run_dir: Path, family: str, dataset: str, seed: int) -> dict[str, Any]:
    s1 = read_json(run_dir / "metrics_stage1_test.json")
    s2 = read_json(run_dir / "metrics_stage2_test.json")
    system = read_json(run_dir / "system_compare_test.json")
    strict = dict(system.get("strict", {}) or {})
    strict_tau = dict(system.get("strict_tau", {}) or {})
    return {
        "surface": "model_competence_staged_protocol_a",
        "claim_status": "competence_sanity_check_not_sota_ranking",
        "dataset": dataset,
        "seed": int(seed),
        "model_family": family,
        "run_dir": str(run_dir),
        "stage1_auc": s1.get("roc_auc"),
        "stage1_fpr": s1.get("fpr"),
        "stage1_tpr": s1.get("tpr"),
        "stage2_macro_f1_fixedK": s2.get("macro_f1_fixedK"),
        "stage2_macro_f1_present": s2.get("macro_f1_present"),
        "stage2_accuracy": s2.get("accuracy"),
        "system_macro_f1_supported_labels": strict.get(
            "system_macro_f1_supported_labels", strict.get("macro_f1")
        ),
        "system_accuracy": strict.get("accuracy"),
        "system_benign_family_fp_rate": strict.get("benign_family_fp_rate"),
        "system_overall_reject_rate": strict.get("overall_reject_rate"),
        "strict_tau_macro_f1_supported_labels": strict_tau.get(
            "system_macro_f1_supported_labels", strict_tau.get("macro_f1")
        ),
        "strict_tau_accuracy": strict_tau.get("accuracy"),
        "strict_tau_benign_family_fp_rate": strict_tau.get("benign_family_fp_rate"),
        "strict_tau_overall_reject_rate": strict_tau.get("overall_reject_rate"),
    }


def run_dataset_seed(
    cfg: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
    model_families: Sequence[str],
    smoke: bool,
) -> list[dict[str, Any]]:
    configure_engine(cfg, dataset, seed)
    out_root = resolve_path(cfg.get("out_root", "outputs/12_model_competence"))
    seed_root = out_root / dataset / f"seed_{seed}"
    shared = seed_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)

    processed_root = resolve_path(cfg["processed_root"])
    ds_dir = processed_root / str(cfg.get("protocol", "A_stratified")) / dataset
    if not ds_dir.exists():
        raise FileNotFoundError(f"Processed dataset directory not found: {ds_dir}")

    max_train = int(cfg["max_train_rows"][dataset])
    max_val = int(cfg["max_val_rows"][dataset])
    if smoke:
        smoke_cfg = dict(cfg.get("smoke", {}) or {})
        max_train = min(max_train, int(smoke_cfg.get("max_train_rows", max_train)))
        max_val = min(max_val, int(smoke_cfg.get("max_val_rows", max_val)))

    prep_path = shared / "preprocessor.joblib"
    if prep_path.exists():
        prep = engine.safe_joblib_load(str(prep_path))
    else:
        prep = engine.fit_preprocessor(str(ds_dir), str(shared))
        engine.safe_joblib_dump(prep, str(prep_path))

    train_parts = engine.list_parts(str(ds_dir), "train")
    val_parts = engine.list_parts(str(ds_dir), "val")
    X1_train, y1_train, _ = engine.collect_xy(
        train_parts, prep, engine.CFG["y_stage1"], max_train,
        seed=seed, filter_attack=None, y2_col=engine.CFG["y_stage2"],
    )
    X_val, y1_val, y2_val = engine.collect_xy(
        val_parts, prep, engine.CFG["y_stage1"], max_val,
        seed=seed + 1, filter_attack=None, y2_col=engine.CFG["y_stage2"],
    )
    X2_train, _, y2_train = engine.collect_xy(
        train_parts, prep, engine.CFG["y_stage1"], max_train,
        seed=seed + 2, filter_attack=True, y2_col=engine.CFG["y_stage2"],
    )
    X2_val, _, y2_val_attack = engine.collect_xy(
        val_parts, prep, engine.CFG["y_stage1"], max_val,
        seed=seed + 3, filter_attack=True, y2_col=engine.CFG["y_stage2"],
    )
    if y2_val is None or y2_train is None or y2_val_attack is None:
        raise RuntimeError("Stage labels are unavailable in the prepared dataset.")

    families = sorted({str(x) for x in y2_train if str(x) and str(x).lower() != "nan"})
    fam_to_idx = {name: idx for idx, name in enumerate(families)}
    y2_train_idx = np.array([fam_to_idx[str(x)] for x in y2_train], dtype=int)
    valid = np.array([str(x) in fam_to_idx for x in y2_val_attack], dtype=bool)
    X2_val = X2_val[valid]
    y2_val_idx = np.array([fam_to_idx[str(x)] for x in y2_val_attack[valid]], dtype=int)

    results: list[dict[str, Any]] = []
    grids = dict(cfg.get("grids", {}) or {})
    n_jobs = int(cfg.get("n_jobs", 8))
    for family in model_families:
        run_dir = seed_root / family
        summary_path = run_dir / "summary.json"
        metadata_path = run_dir / "run_metadata.json"
        expected_metadata = {
            "smoke": bool(smoke),
            "max_train_rows": int(max_train),
            "max_val_rows": int(max_val),
            "dataset": str(dataset),
            "seed": int(seed),
            "model_family": str(family),
        }
        existing_metadata = read_json(metadata_path)
        if (
            summary_path.exists()
            and existing_metadata
            and all(existing_metadata.get(k) == v for k, v in expected_metadata.items())
        ):
            results.append(read_json(summary_path))
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        family_grid = dict(grids.get(family, {}) or {})
        stage1_candidates = list(family_grid.get("stage1", []) or [])
        stage2_candidates = list(family_grid.get("stage2", []) or [])
        if not stage1_candidates or not stage2_candidates:
            raise ValueError(f"Missing Stage-1/Stage-2 grid for {family}.")

        stage1, platt, stage1_best, stage1_table = select_stage1(
            family, stage1_candidates, X1_train, y1_train, X_val, y1_val, y2_val,
            seed=seed, n_jobs=n_jobs, threshold_cfg=dict(cfg.get("stage1_threshold", {}) or {}),
        )
        stage2, temperature, stage2_best, stage2_table = select_stage2(
            family, stage2_candidates, X2_train, y2_train_idx, X2_val, y2_val_idx,
            n_classes=len(families), seed=seed, n_jobs=n_jobs,
        )

        stage1_table.to_csv(run_dir / "stage1_candidate_validation.csv", index=False)
        stage2_table.to_csv(run_dir / "stage2_candidate_validation.csv", index=False)
        safe_json(run_dir / "stage1_selected.json", stage1_best)
        safe_json(run_dir / "stage2_selected.json", stage2_best)
        engine.safe_joblib_dump(stage1, str(run_dir / "stage1_best.joblib"))
        engine.safe_joblib_dump(stage2, str(run_dir / "stage2_best.joblib"))
        safe_json(run_dir / "stage1_platt.json", {"platt": platt})
        safe_json(run_dir / "stage2_temperature.json", {"T": float(temperature), "K": len(families)})
        safe_json(run_dir / "families.json", {"families": families})

        p_attack_val = engine.apply_platt(
            np.asarray(stage1.predict_proba(X_val), dtype=np.float64)[:, 1], platt
        )
        p2_val_all = engine.apply_temperature(
            predict_multi_proba(stage2, X_val, len(families)), temperature
        )
        fam_pred_val = np.argmax(p2_val_all, axis=1).astype(int)
        fam_pmax_val = np.max(p2_val_all, axis=1).astype(np.float64)
        ysys_val = system_truth(y1_val, y2_val)
        thr_high = float(stage1_best["threshold"])
        thr_low, tau_cascade = engine.tune_cascade_thr_low_and_tau(
            str(run_dir), y1_val, ysys_val, p_attack_val, fam_pred_val,
            fam_pmax_val, families, thr_high,
        )
        tau_strict = engine.pick_tau_strict(
            str(run_dir), ysys_val, p_attack_val, fam_pred_val,
            fam_pmax_val, families, thr_high,
        )
        safe_json(
            run_dir / "abstain_selected.json",
            {
                "policy": engine.CFG["abstain"].get("policy", "abstain"),
                "label": engine.CFG["abstain"].get("label", "Unknown"),
                "thr_high": thr_high,
                "thr_low": float(thr_low),
                "tau_strict": float(tau_strict),
                "tau_cascade": float(tau_cascade),
            },
        )

        engine.evaluate_system(
            str(ds_dir), str(run_dir), "val", prep, stage1, platt, stage2,
            families, temperature, thr_high, thr_low, tau_strict, tau_cascade,
        )
        engine.evaluate_system(
            str(ds_dir), str(run_dir), "test", prep, stage1, platt, stage2,
            families, temperature, thr_high, thr_low, tau_strict, tau_cascade,
        )
        summary = summarize_run(run_dir, family, dataset, seed)
        safe_json(summary_path, summary)
        safe_json(metadata_path, expected_metadata)
        results.append(summary)
    return results


def collect_primary_vera_rows(
    cfg: Mapping[str, Any],
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Load existing repeated RF/XGB Protocol-A evidence for matched comparison."""
    root_value = cfg.get("primary_seed_reliability_root")
    if not root_value:
        return []
    root = resolve_path(str(root_value))
    wanted_datasets = set(str(x) for x in datasets)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        source = root / f"seed_{int(seed)}" / "runs" / "summary" / "protocol_a_core_summary.csv"
        if not source.exists():
            continue
        frame = pd.read_csv(source)
        if "policy_variant" in frame.columns:
            frame = frame[frame["policy_variant"].astype(str) == "strict"].copy()
        if "model_family" in frame.columns:
            frame = frame[frame["model_family"].astype(str).isin(["rf", "xgb"])].copy()
        if "dataset" in frame.columns:
            frame = frame[frame["dataset"].astype(str).isin(wanted_datasets)].copy()
        for _, row in frame.iterrows():
            rows.append({
                "surface": "primary_vera_protocol_a",
                "claim_status": "existing_primary_reference",
                "dataset": str(row.get("dataset", "")),
                "seed": int(seed),
                "model_family": str(row.get("model_family", "")),
                "run_dir": str(row.get("run_dir", "")),
                "stage1_auc": row.get("stage1_roc_auc"),
                "stage1_fpr": row.get("stage1_fpr"),
                "stage1_tpr": row.get("stage1_tpr"),
                "stage2_macro_f1_fixedK": row.get("stage2_macro_f1_fixedK"),
                "stage2_macro_f1_present": row.get("stage2_macro_f1_present"),
                "stage2_accuracy": row.get("stage2_accuracy"),
                "system_macro_f1_supported_labels": row.get("system_macro_f1_supported_labels"),
                "system_accuracy": row.get("system_accuracy"),
                "system_benign_family_fp_rate": row.get("benign_family_fp_rate"),
                "system_overall_reject_rate": row.get("overall_reject_rate"),
                "strict_tau_macro_f1_supported_labels": None,
                "strict_tau_accuracy": None,
                "strict_tau_benign_family_fp_rate": None,
                "strict_tau_overall_reject_rate": None,
            })
    return rows


def build_summary(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    metric_cols = [
        "stage1_auc",
        "stage1_fpr",
        "stage2_macro_f1_present",
        "system_macro_f1_supported_labels",
        "system_accuracy",
        "system_benign_family_fp_rate",
        "strict_tau_macro_f1_supported_labels",
        "strict_tau_accuracy",
        "strict_tau_benign_family_fp_rate",
    ]
    out_rows: list[dict[str, Any]] = []
    for (dataset, family), group in frame.groupby(["dataset", "model_family"], dropna=False):
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            out_rows.append({
                "dataset": dataset,
                "model_family": family,
                "metric": metric,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return pd.DataFrame(out_rows)


def run_model_competence(
    config: Mapping[str, Any] | None,
    *,
    dry_run: bool = False,
    smoke: bool = False,
    datasets: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    model_families: Sequence[str] | None = None,
    processed_root: str | Path | None = None,
) -> Path:
    cfg = competence_cfg(config)
    if not cfg:
        raise ValueError("Missing model_competence configuration.")
    if processed_root is not None:
        cfg["processed_root"] = str(processed_root)
    selected_datasets = list(datasets or cfg.get("datasets", []))
    selected_seeds = [int(x) for x in (seeds or cfg.get("seeds", []))]
    selected_models = list(model_families or cfg.get("model_families", []))
    if smoke:
        smoke_cfg = dict(cfg.get("smoke", {}) or {})
        selected_datasets = list(smoke_cfg.get("datasets", selected_datasets))
        selected_seeds = [int(x) for x in smoke_cfg.get("seeds", selected_seeds)]

    smoke_cfg = dict(cfg.get("smoke", {}) or {})
    out_root_value = smoke_cfg.get("out_root") if smoke else None
    out_root = resolve_path(out_root_value or cfg.get("out_root", "outputs/12_model_competence"))
    resolved_processed_root = resolve_path(cfg["processed_root"])
    if dry_run:
        print(f"[dry-run] out_root={out_root}")
        print(
            f"[dry-run] processed_root={resolved_processed_root} "
            f"exists={resolved_processed_root.exists()}"
        )
        print(f"[dry-run] datasets={selected_datasets}")
        print(f"[dry-run] seeds={selected_seeds}")
        print(f"[dry-run] models={selected_models}")
        for family in selected_models:
            grid = dict((cfg.get("grids", {}) or {}).get(family, {}) or {})
            print(
                f"[dry-run] {family}: stage1_candidates={len(grid.get('stage1', []))} "
                f"stage2_candidates={len(grid.get('stage2', []))}"
            )
        return out_root

    if not resolved_processed_root.exists():
        raise FileNotFoundError(
            "Prepared Protocol-A root not found: "
            f"{resolved_processed_root}. "
            "The public repository intentionally excludes prepared benchmark rows. "
            "Point this run at your existing processed_V5 directory with "
            "--processed-root <path>, or recreate the prepared data before running."
        )

    all_rows: list[dict[str, Any]] = collect_primary_vera_rows(
        cfg,
        datasets=selected_datasets,
        seeds=selected_seeds,
    )
    for dataset in selected_datasets:
        for seed in selected_seeds:
            all_rows.extend(
                run_dataset_seed(
                    cfg,
                    dataset=dataset,
                    seed=seed,
                    model_families=selected_models,
                    smoke=smoke,
                )
            )

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(summary_dir / "model_competence_runs.csv", index=False)
    build_summary(all_rows).to_csv(summary_dir / "model_competence_summary.csv", index=False)
    return out_root
