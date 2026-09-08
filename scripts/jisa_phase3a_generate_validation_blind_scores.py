#!/usr/bin/env python3
"""Generate heldout-validation-blind Protocol-B score artifacts for JISA Phase 3A.

The primary Phase-3 comparison isolates validation visibility. Historical
preprocessing behavior is therefore reproduced as-is: the preprocessor is fitted on
the full Protocol-B training partition before Stage-1/Stage-2 LOAO filtering. This
unsupervised feature exposure is recorded explicitly and is not described as fully
inductive unknown-family blindness.

For each seed/holdout, the runner reconstructs the historical top Stage-2 equivalence
set, resolves any tie with Stage-1 AUROC computed only on benign plus represented-
family validation traffic, selects the Stage-1 operating threshold on that same
heldout-free validation surface, freezes all choices, and then writes heldout-free
validation scores plus full test scores for later rejector replay.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework._native import protocol_b_grid as grid_mod  # noqa: E402
from ids_eval_framework._native import protocol_b_open_set as open_mod  # noqa: E402
from ids_eval_framework.src import two_stage_engine as helper  # noqa: E402
from ids_eval_framework.src.paths import load_config  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "jisa_phase3_validation_blind.yml"
DEFAULT_PREFLIGHT = REPO_ROOT / ".release-audit" / "jisa_phase3_preflight" / "preflight.json"
DEFAULT_TIE_AUDIT = REPO_ROOT / ".release-audit" / "jisa_phase3_tie_audit" / "tie_audit.json"
DATASET = "CICIoT2023"
TOL = 1e-12


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return {}
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        out = ast.literal_eval(text)
    if not isinstance(out, dict):
        raise ValueError(f"Expected mapping, got: {value!r}")
    return dict(out)


def sanitize(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9.-]+", "_", str(value))).strip("._") or "NA"


def candidate_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "model_profile",
        "model_family",
        "apply_loao_stage1",
        "stage1_weight_mode",
        "stage2_weight_mode",
        "stage1_params",
        "stage2_params",
    ]
    return [c for c in preferred if c in df.columns]


def candidate_id(row: pd.Series, cols: list[str]) -> str:
    parts: list[str] = []
    for col in cols:
        value = row[col]
        if isinstance(value, float) and math.isnan(value):
            value = ""
        parts.append(f"{col}={value}")
    return "|".join(parts)


def short_label(row: pd.Series) -> str:
    profile = str(row.get("model_profile", "")).strip()
    family = str(row.get("model_family", "")).strip()
    weight = str(row.get("stage1_weight_mode", "")).strip()
    if profile and profile not in {"nan", family}:
        return profile
    if family and weight and weight != "nan":
        return f"{family}_{weight}"
    return profile or family or "unknown"


def stage2_signature(row: pd.Series) -> str:
    return "|".join(
        [
            f"model_family={row.get('model_family', '')}",
            f"stage2_weight_mode={row.get('stage2_weight_mode', '')}",
            f"stage2_params={row.get('stage2_params', '')}",
        ]
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_local_path(raw: str, anchors: Iterable[Path] = ()) -> Path:
    p = Path(str(raw))
    if p.is_absolute() and p.exists():
        return p.resolve()
    candidates = [REPO_ROOT / p, REPO_ROOT.parent / "Codes" / p, REPO_ROOT.parent / p]
    for anchor in anchors:
        candidates.append(anchor / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / p).resolve()


def selected_aggregate(seed: int, preflight: dict[str, Any]) -> Path:
    mapping = preflight.get("selected_seed_aggregates", {})
    raw = mapping.get(str(seed)) if isinstance(mapping, dict) else None
    if not raw:
        raise RuntimeError(f"No Phase-3 preflight aggregate for seed {seed}")
    path = Path(str(raw))
    if not path.exists():
        raise RuntimeError(f"Phase-3 aggregate missing: {path}")
    return path.resolve()


def load_holdout_candidates(aggregate_path: Path, holdout: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(aggregate_path)
    df = df[(df["dataset"].astype(str) == DATASET) & (df["holdout_family"].astype(str) == str(holdout))].copy()
    if df.empty:
        raise RuntimeError(f"No {DATASET}/{holdout} rows in {aggregate_path}")
    cols = candidate_columns(df)
    if not cols:
        raise RuntimeError("Cannot construct candidate identity from historical aggregate.")
    df["_candidate_id"] = df.apply(lambda r: candidate_id(r, cols), axis=1)
    df = df.sort_index().drop_duplicates(subset=["_candidate_id"], keep="last").copy()
    df["stage2_macro_f1_val"] = pd.to_numeric(df["stage2_macro_f1_val"], errors="coerce")
    df = df[df["stage2_macro_f1_val"].notna()].copy()
    df = df.sort_values(["stage2_macro_f1_val", "_candidate_id"], ascending=[False, True]).reset_index(drop=True)
    if len(df) != 3:
        raise RuntimeError(f"Expected exactly 3 historical candidates for {holdout}, found {len(df)}")
    top_value = float(df.iloc[0]["stage2_macro_f1_val"])
    top = df[(top_value - df["stage2_macro_f1_val"].astype(float)).abs() <= TOL].copy()
    if top.empty:
        raise RuntimeError("Historical top-equivalence set is empty.")
    if not top["apply_loao_stage1"].map(as_bool).all():
        raise RuntimeError("Top-equivalence set contains non-strict Stage-1 LOAO candidate.")
    return df, top


def find_manifest(aggregate_path: Path, candidate_rows: pd.DataFrame, holdout: str) -> tuple[Path, dict[str, Any]]:
    direct: list[Path] = []
    if "manifest_path" in candidate_rows.columns:
        for raw in candidate_rows["manifest_path"].dropna().astype(str):
            p = resolve_local_path(raw, anchors=[aggregate_path.parent])
            if p.exists():
                direct.append(p)

    search_roots = [aggregate_path.parent, *list(aggregate_path.parents)[:3]]
    seen: set[str] = set()
    candidates: list[Path] = []
    for p in direct:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            candidates.append(p.resolve())
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("scenario_manifest.json"):
                key = str(p.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(p.resolve())
        except Exception:
            pass

    for path in candidates:
        try:
            payload = load_json(path)
        except Exception:
            continue
        if str(payload.get("dataset", "")) == DATASET and str(payload.get("holdout_family", "")) == str(holdout):
            return path, payload
    raise RuntimeError(f"Could not resolve scenario manifest for {DATASET}/{holdout} near {aggregate_path}")


def resolve_dataset_dir(manifest: dict[str, Any], manifest_path: Path, aggregate_path: Path) -> Path:
    raw = str(manifest.get("processed_dir", "")).strip()
    if raw:
        p = resolve_local_path(raw, anchors=[manifest_path.parent, aggregate_path.parent])
        if p.exists():
            return p
    fallback = [
        REPO_ROOT / "outputs" / "02_prepared_data" / "processed_V5" / "B_day_file" / DATASET,
        REPO_ROOT / "processed_V5" / "B_day_file" / DATASET,
        REPO_ROOT.parent / "Codes" / "outputs" / "02_prepared_data" / "processed_V5" / "B_day_file" / DATASET,
        REPO_ROOT.parent / "Codes" / "processed_V5" / "B_day_file" / DATASET,
        REPO_ROOT.parent / "outputs" / "02_prepared_data" / "processed_V5" / "B_day_file" / DATASET,
    ]
    for p in fallback:
        if p.exists():
            return p.resolve()
    raise RuntimeError(f"Could not resolve prepared Protocol-B dataset from manifest: {manifest_path}")


def configure_native(cfg: dict[str, Any], seed: int) -> None:
    exp = cfg["experiment"]
    stage1 = cfg["stage1_threshold_search"]
    n_jobs = int(exp.get("n_jobs", 24))
    grid_mod.CFG["random_seed"] = int(seed)
    grid_mod.CFG["n_jobs"] = n_jobs
    grid_mod.CFG["max_train_rows"] = {DATASET: int(exp["max_train_rows"])}
    grid_mod.CFG["max_val_rows"] = {DATASET: int(exp["max_val_rows"])}
    grid_mod.CFG["max_test_rows"] = {DATASET: None}
    grid_mod.CFG["inv_family_clip_max"] = float(exp.get("inv_family_clip_max", 20.0))
    grid_mod.CFG["stage1_threshold_search"] = {
        "target_fpr_list": [float(x) for x in stage1["target_fpr_list"]],
        "min_family_support_list": [int(x) for x in stage1["min_family_support_list"]],
        "objective_mode_list": [str(x) for x in stage1["objective_mode_list"]],
        "p10_quantile_list": [float(x) for x in stage1["p10_quantile_list"]],
        "sweep_points": int(stage1["sweep_points"]),
        "selection_tiebreak": str(stage1["selection_tiebreak"]),
    }
    for key in ("xgb_binary_defaults", "xgb_multi_defaults"):
        grid_mod.CFG[key] = dict(grid_mod.CFG[key])
        grid_mod.CFG[key]["n_jobs"] = n_jobs
        grid_mod.CFG[key]["tree_method"] = "hist"
        grid_mod.CFG[key]["device"] = str(exp.get("xgb_device", "cuda"))
        grid_mod.CFG[key].pop("predictor", None)


def load_or_fit_preprocessor(dataset_dir: Path, case_dir: Path):
    prep_path = case_dir / "preprocessor.joblib"
    if prep_path.exists():
        return helper.safe_joblib_load(str(prep_path))
    case_dir.mkdir(parents=True, exist_ok=True)
    return helper.fit_preprocessor(str(dataset_dir), str(case_dir))


def load_splits(prep, dataset_dir: Path, manifest: dict[str, Any], cfg: dict[str, Any], seed: int):
    y1_col = grid_mod.canonical_col(str(manifest.get("y_stage1_col", "y_stage1_attack")))
    y2_col = grid_mod.canonical_col(str(manifest.get("y_stage2_col", "y_stage2_family")))
    usecols = prep.num_cols + prep.cat_cols + [y1_col, y2_col]
    exp = cfg["experiment"]
    train_df = grid_mod.collect_split_frame(str(dataset_dir), "train", usecols, int(exp["max_train_rows"]), seed=seed)
    val_df = grid_mod.collect_split_frame(str(dataset_dir), "val", usecols, int(exp["max_val_rows"]), seed=seed + 1)
    test_df = grid_mod.collect_split_frame(str(dataset_dir), "test", usecols, None, seed=seed + 2)
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("At least one split is empty.")
    return y1_col, y2_col, train_df, val_df, test_df


def blind_validation_mask(val_df: pd.DataFrame, y1_col: str, y2_col: str, holdout: str) -> np.ndarray:
    heldout = (val_df[y1_col].astype(int).to_numpy() == 1) & (val_df[y2_col].astype(str).to_numpy(dtype=object) == str(holdout))
    return ~heldout


def train_stage2_shared(prep, train_df: pd.DataFrame, blind_val_df: pd.DataFrame, y1_col: str, y2_col: str, valid_known_families: list[str], candidate: pd.Series, seed: int):
    attack_train = train_df.loc[(train_df[y1_col].astype(int) == 1) & (train_df[y2_col].astype(str).isin(valid_known_families))].copy()
    if attack_train.empty:
        raise RuntimeError("Stage-2 training set empty after known-family filtering.")
    labels = attack_train[y2_col].astype(str).to_numpy(dtype=object)
    train_set = {str(x) for x in labels}
    families = [f for f in valid_known_families if f in train_set]
    missing = [f for f in valid_known_families if f not in train_set]
    if len(families) < 2:
        raise RuntimeError(f"Fewer than two represented Stage-2 families after cap: {families}")
    if missing:
        attack_train = attack_train[attack_train[y2_col].astype(str).isin(families)].copy()
        labels = attack_train[y2_col].astype(str).to_numpy(dtype=object)

    fam_to_idx = {fam: i for i, fam in enumerate(families)}
    y_idx = np.array([fam_to_idx[str(f)] for f in labels], dtype=int)
    X2 = prep.transform(attack_train.drop(columns=[y1_col, y2_col], errors="ignore"))
    weight_mode = str(candidate.get("stage2_weight_mode", "balanced"))
    weights = grid_mod.stage2_sample_weights(y_idx, len(families), weight_mode)
    model = grid_mod.build_stage2_model(str(candidate["model_family"]), parse_dict(candidate["stage2_params"]), n_classes=len(families), seed=seed + 10)
    if weights is None:
        model.fit(X2, y_idx)
    else:
        model.fit(X2, y_idx, sample_weight=weights)

    known_val = blind_val_df.loc[(blind_val_df[y1_col].astype(int) == 1) & (blind_val_df[y2_col].astype(str).isin(families))].copy()
    if known_val.empty:
        stage2_f1 = float("nan")
    else:
        Xv = prep.transform(known_val.drop(columns=[y1_col, y2_col], errors="ignore"))
        yv = np.array([fam_to_idx[str(f)] for f in known_val[y2_col].astype(str)], dtype=int)
        pv = grid_mod.predict_multi_proba(model, Xv)
        pred = np.argmax(pv, axis=1)
        stage2_f1 = float(f1_score(yv, pred, average="macro", labels=list(range(len(families))), zero_division=0))
    return model, families, stage2_f1


def train_stage1_candidate(prep, train_df: pd.DataFrame, blind_val_df: pd.DataFrame, y1_col: str, y2_col: str, holdout: str, candidate: pd.Series, seed: int):
    s1_train = train_df.copy()
    drop = (s1_train[y1_col].astype(int) == 1) & (s1_train[y2_col].astype(str) == str(holdout))
    s1_train = s1_train.loc[~drop].reset_index(drop=True)
    X = prep.transform(s1_train.drop(columns=[y1_col, y2_col], errors="ignore"))
    y1 = s1_train[y1_col].astype(int).to_numpy()
    y2 = s1_train[y2_col].astype(str).fillna("").to_numpy(dtype=object)
    model = grid_mod.build_stage1_model(str(candidate["model_family"]), parse_dict(candidate["stage1_params"]), str(candidate["stage1_weight_mode"]), seed=seed)
    weights = grid_mod.make_stage1_sample_weights(y1, y2, str(candidate["stage1_weight_mode"]), float(grid_mod.CFG["inv_family_clip_max"]))
    if weights is None:
        model.fit(X, y1)
    else:
        model.fit(X, y1, sample_weight=weights)

    Xv = prep.transform(blind_val_df.drop(columns=[y1_col, y2_col], errors="ignore"))
    yv = blind_val_df[y1_col].astype(int).to_numpy()
    p = grid_mod.predict_binary_proba(model, Xv)
    auc = float(roc_auc_score(yv, p)) if np.unique(yv).size >= 2 else float("nan")
    return model, p, auc


def choose_profile(candidate_results: list[dict[str, Any]]) -> dict[str, Any]:
    def key(rec: dict[str, Any]):
        stage2 = float(rec["stage2_macro_f1_blind_val"])
        stage1 = float(rec["stage1_auc_blind_val"])
        return (-stage2 if np.isfinite(stage2) else float("inf"), -stage1 if np.isfinite(stage1) else float("inf"), str(rec["candidate_id"]))
    return sorted(candidate_results, key=key)[0]


def write_score_frame(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def case_complete(case_dir: Path) -> bool:
    needed = [
        "run_complete.json",
        "selected_profile.json",
        "stage1_threshold_best.json",
        "candidate_profile_selection.csv",
        "val_known_scores.csv.gz",
        "test_scores.csv.gz",
    ]
    return all((case_dir / name).exists() for name in needed)


def run_case(cfg: dict[str, Any], seed: int, holdout: str, *, dry_run: bool, force: bool) -> dict[str, Any]:
    if not DEFAULT_PREFLIGHT.exists() or not DEFAULT_TIE_AUDIT.exists():
        raise RuntimeError("Run jisa_phase3_preflight.py and jisa_phase3_tie_audit.py first.")
    preflight = load_json(DEFAULT_PREFLIGHT)
    tie = load_json(DEFAULT_TIE_AUDIT)
    if not bool(tie.get("execution_ready", False)):
        raise RuntimeError("Phase-3 tie audit is not execution-ready.")

    aggregate_path = selected_aggregate(seed, preflight)
    all_candidates, top_candidates = load_holdout_candidates(aggregate_path, holdout)
    manifest_path, manifest = find_manifest(aggregate_path, top_candidates, holdout)
    dataset_dir = resolve_dataset_dir(manifest, manifest_path, aggregate_path)
    output_root = resolve_local_path(str(cfg["experiment"]["output_root"]))
    case_dir = output_root / "score_generation" / f"seed_{seed}" / sanitize(holdout)

    plan = {
        "dataset": DATASET,
        "seed": seed,
        "holdout_family": holdout,
        "aggregate_path": str(aggregate_path),
        "manifest_path": str(manifest_path),
        "dataset_dir": str(dataset_dir),
        "case_dir": str(case_dir),
        "historical_candidate_count": int(len(all_candidates)),
        "top_equivalence_count": int(len(top_candidates)),
        "top_equivalence_labels": [short_label(r) for _, r in top_candidates.iterrows()],
        "preprocessor_exposure": "historical_full_train_unsupervised_feature_exposure",
        "heldout_validation_used_for_selection": False,
        "test_used_for_selection": False,
    }
    if dry_run:
        print(json.dumps(plan, indent=2))
        return {"status": "dry_run", **plan}
    if case_complete(case_dir) and not force:
        print(f"SKIP complete seed={seed} holdout={holdout}")
        return {"status": "skipped_complete", **plan}

    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    configure_native(cfg, seed)
    prep = load_or_fit_preprocessor(dataset_dir, case_dir)
    y1_col, y2_col, train_df, val_df, test_df = load_splits(prep, dataset_dir, manifest, cfg, seed)
    mask = blind_validation_mask(val_df, y1_col, y2_col, holdout)
    blind_val_df = val_df.loc[mask].reset_index(drop=True)
    hidden_val_rows = int((~mask).sum())
    if hidden_val_rows <= 0:
        raise RuntimeError(f"No heldout-family validation rows found for {holdout}")

    valid_known_families = [str(x) for x in manifest["valid_known_families"]]
    benign_label = str(manifest.get("benign_label", "Benign"))
    unknown_label = str(manifest.get("unknown_label", "Unknown"))

    stage2_cache: dict[str, tuple[Any, list[str], float]] = {}
    stage1_models: dict[str, Any] = {}
    stage1_probs: dict[str, np.ndarray] = {}
    candidate_results: list[dict[str, Any]] = []

    for _, candidate in top_candidates.iterrows():
        cid = str(candidate["_candidate_id"])
        sig = stage2_signature(candidate)
        if sig not in stage2_cache:
            stage2_cache[sig] = train_stage2_shared(prep, train_df, blind_val_df, y1_col, y2_col, valid_known_families, candidate, seed)
        _, _, stage2_f1 = stage2_cache[sig]
        stage1_model, p_blind_val, stage1_auc = train_stage1_candidate(prep, train_df, blind_val_df, y1_col, y2_col, holdout, candidate, seed)
        stage1_models[cid] = stage1_model
        stage1_probs[cid] = p_blind_val
        candidate_results.append(
            {
                "candidate_id": cid,
                "candidate_label": short_label(candidate),
                "model_family": str(candidate["model_family"]),
                "stage1_weight_mode": str(candidate["stage1_weight_mode"]),
                "stage2_weight_mode": str(candidate.get("stage2_weight_mode", "balanced")),
                "stage2_signature": sig,
                "stage2_macro_f1_blind_val": float(stage2_f1),
                "stage1_auc_blind_val": float(stage1_auc),
            }
        )

    selected = choose_profile(candidate_results)
    selected_cid = str(selected["candidate_id"])
    selected_row = top_candidates.loc[top_candidates["_candidate_id"].astype(str) == selected_cid].iloc[0]
    selected_stage1 = stage1_models[selected_cid]
    selected_p_blind_val = stage1_probs[selected_cid]
    selected_stage2, families, selected_stage2_f1 = stage2_cache[stage2_signature(selected_row)]

    y1_blind = blind_val_df[y1_col].astype(int).to_numpy()
    y2_blind = blind_val_df[y2_col].astype(str).fillna("").to_numpy(dtype=object)
    thr, thr_best, thr_grid = grid_mod.select_stage1_threshold(helper, y1_blind, y2_blind, selected_p_blind_val)

    selection_df = pd.DataFrame(candidate_results).sort_values(["stage2_macro_f1_blind_val", "stage1_auc_blind_val", "candidate_id"], ascending=[False, False, True])
    selection_df["selected"] = selection_df["candidate_id"].astype(str) == selected_cid
    selection_df.to_csv(case_dir / "candidate_profile_selection.csv", index=False)
    thr_grid.to_csv(case_dir / "stage1_threshold_grid.csv", index=False)
    (case_dir / "stage1_threshold_best.json").write_text(json.dumps(thr_best, indent=2, sort_keys=True), encoding="utf-8")

    selected_payload = {
        **plan,
        "blind_validation_rows": int(len(blind_val_df)),
        "hidden_holdout_validation_rows": hidden_val_rows,
        "train_rows_loaded": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "selected_candidate_id": selected_cid,
        "selected_candidate_label": str(selected["candidate_label"]),
        "selected_model_family": str(selected["model_family"]),
        "selected_stage1_weight_mode": str(selected["stage1_weight_mode"]),
        "selected_stage2_weight_mode": str(selected["stage2_weight_mode"]),
        "stage2_macro_f1_blind_val": float(selected_stage2_f1),
        "stage1_auc_blind_val": float(selected["stage1_auc_blind_val"]),
        "stage1_threshold": float(thr),
        "known_families": list(families),
        "selection_rule": "stage2_macro_f1_known_val_then_stage1_auc_known_val_then_lexical",
    }
    (case_dir / "selected_profile.json").write_text(json.dumps(selected_payload, indent=2, sort_keys=True), encoding="utf-8")

    val_scores = open_mod.build_score_frame(grid_mod, prep, selected_stage1, selected_stage2, "val_known", blind_val_df, y1_col, y2_col, holdout, families, benign_label, unknown_label)
    test_scores = open_mod.build_score_frame(grid_mod, prep, selected_stage1, selected_stage2, "test", test_df, y1_col, y2_col, holdout, families, benign_label, unknown_label)
    write_score_frame(case_dir / "val_known_scores.csv.gz", val_scores)
    write_score_frame(case_dir / "test_scores.csv.gz", test_scores)

    elapsed = time.perf_counter() - started
    complete = {
        "status": "complete",
        "dataset": DATASET,
        "seed": seed,
        "holdout_family": holdout,
        "elapsed_seconds": elapsed,
        "selected_candidate_label": selected["candidate_label"],
        "stage1_threshold": float(thr),
        "blind_validation_rows": int(len(blind_val_df)),
        "hidden_holdout_validation_rows": hidden_val_rows,
        "test_rows": int(len(test_df)),
        "row_level_scores_local_only": True,
        "models_saved": False,
    }
    (case_dir / "run_complete.json").write_text(json.dumps(complete, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"COMPLETE seed={seed} holdout={holdout} profile={selected['candidate_label']} "
        f"stage1_auc_blind={selected['stage1_auc_blind_val']:.6f} thr={thr:.12g} "
        f"hidden_val={hidden_val_rows} elapsed={elapsed:.1f}s"
    )
    return complete


def summarize(cfg: dict[str, Any]) -> int:
    root = resolve_local_path(str(cfg["experiment"]["output_root"])) / "score_generation"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("seed_*/*/run_complete.json")):
        try:
            rec = load_json(path)
        except Exception:
            continue
        selected_path = path.parent / "selected_profile.json"
        if selected_path.exists():
            selected = load_json(selected_path)
            rec.update(
                {
                    "selected_stage2_macro_f1_blind_val": selected.get("stage2_macro_f1_blind_val"),
                    "selected_stage1_auc_blind_val": selected.get("stage1_auc_blind_val"),
                    "selected_candidate_label": selected.get("selected_candidate_label"),
                    "selected_model_family": selected.get("selected_model_family"),
                    "selected_stage1_weight_mode": selected.get("selected_stage1_weight_mode"),
                    "selected_stage1_threshold": selected.get("stage1_threshold"),
                }
            )
        rows.append(rec)
    if not rows:
        print("No completed Phase-3 score-generation cases found.")
        return 1
    df = pd.DataFrame(rows).sort_values(["holdout_family", "seed"])
    out_dir = root.parent / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "score_generation_cases.csv", index=False)
    cols = [
        "seed",
        "holdout_family",
        "selected_candidate_label",
        "selected_stage1_auc_blind_val",
        "selected_stage1_threshold",
        "hidden_holdout_validation_rows",
        "test_rows",
    ]
    print(df[cols].to_string(index=False))
    print(f"completed cases: {len(df)}/30")
    return 0 if len(df) == 30 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--seed", type=int)
    p.add_argument("--holdout")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.summarize_only:
        return summarize(cfg)
    if args.seed is None or not args.holdout:
        raise SystemExit("--seed and --holdout are required unless --summarize-only is used.")
    seeds = [int(x) for x in cfg["experiment"]["seeds"]]
    holdouts = [str(x) for x in cfg["experiment"]["holdouts"]]
    if args.seed not in seeds:
        raise SystemExit(f"Seed {args.seed} not in configured seeds: {seeds}")
    if args.holdout not in holdouts:
        raise SystemExit(f"Holdout {args.holdout!r} not in configured holdouts: {holdouts}")
    run_case(cfg, args.seed, args.holdout, dry_run=args.dry_run, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
