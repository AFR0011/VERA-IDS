"""Resumable repeated-seed orchestration for manuscript reliability runs."""

from __future__ import annotations

import json
import math
import os
import socket
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ids_eval_framework.src.paths import (
    REPO_ROOT,
    FRAMEWORK_ROOT,
    deep_update,
    load_config,
    resolve_repo_path,
)


DEFAULT_LANES: dict[str, dict[str, Any]] = {
    "protocol_a_two_stage": {
        "enabled": True,
        "runner": "protocol_a_two_stage",
        "base_config": "config/protocol_a.yml",
        "output_subdir": "protocol_a_two_stage",
    },
    "protocol_a_flat": {
        "enabled": True,
        "runner": "protocol_a_flat",
        "base_config": "config/protocol_a.yml",
        "output_subdir": "protocol_a_flat",
    },
    "protocol_b_loao": {
        "enabled": True,
        "runner": "protocol_b_loao",
        "base_config": "config/protocol_b_loao.yml",
        "output_subdir": "protocol_b_loao",
    },
}

PRIMARY_METRICS = [
    "system_macro_f1_supported_labels",
    "system_macro_f1_declared_output_labels_historical",
    "system_accuracy",
    "stage2_macro_f1_fixedK",
    "stage2_macro_f1_present",
    "stage1_roc_auc",
    "macro_f1",
    "accuracy",
    "weighted_f1",
    "binary_attack_f1",
    "benign_family_fp_rate",
    "overall_reject_rate",
    "unknown_detection_rate",
    "false_unknown_rate_all_known",
    "false_unknown_rate_known_attacks",
    "stage1_auc_val",
    "stage2_macro_f1_val",
]

STATE_FILES = {
    "manifest": "run_manifest.json",
    "started": "run_started.json",
    "complete": "run_complete.json",
    "error": "error.json",
    "lock": "run.lock",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def progress_print(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        pass


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reliability_cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    section = cfg.get("seed_reliability", cfg)
    if not isinstance(section, Mapping):
        raise ValueError("seed_reliability config must be a mapping")
    return dict(section)


def configured_seeds(config: Mapping[str, Any] | None, seeds: Sequence[int] | None = None) -> list[int]:
    if seeds:
        return [int(x) for x in seeds]
    cfg = reliability_cfg(config)
    return [int(x) for x in cfg.get("seeds", [123, 124, 125, 126, 127])]


def output_root(config: Mapping[str, Any] | None) -> Path:
    cfg = reliability_cfg(config)
    root = cfg.get("output_root", "ids_eval_framework/outputs/10_seed_reliability")
    return Path(resolve_repo_path(str(root))).resolve()


def configured_lanes(
    config: Mapping[str, Any] | None,
    lane_names: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    cfg = reliability_cfg(config)
    parallelism = cfg.get("parallelism", {}) or {}
    if parallelism and not isinstance(parallelism, Mapping):
        raise ValueError("seed_reliability.parallelism must be a mapping")
    default_n_jobs = parallelism.get("n_jobs") if isinstance(parallelism, Mapping) else None
    raw_lanes = cfg.get("lanes")
    if raw_lanes is None:
        raw_lanes = DEFAULT_LANES
    if not isinstance(raw_lanes, Mapping):
        raise ValueError("seed_reliability.lanes must be a mapping")

    lanes: dict[str, dict[str, Any]] = {}
    for name, raw in raw_lanes.items():
        defaults = DEFAULT_LANES.get(str(name), {})
        if raw is False:
            lane = dict(defaults)
            lane["enabled"] = False
        elif isinstance(raw, Mapping):
            lane = dict(defaults)
            lane.update(raw)
        else:
            raise ValueError(f"seed_reliability.lanes.{name} must be a mapping or false")
        if default_n_jobs is not None and "n_jobs" not in lane:
            lane["n_jobs"] = int(default_n_jobs)
        lanes[str(name)] = lane

    if lane_names:
        requested = {str(x) for x in lane_names}
        missing = requested.difference(lanes)
        if missing:
            raise ValueError(f"Unknown seed reliability lanes: {sorted(missing)}")
        lanes = {name: lane for name, lane in lanes.items() if name in requested}

    return {name: lane for name, lane in lanes.items() if bool(lane.get("enabled", True))}


def lane_parallelism_overrides(lane_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return legacy override fragments for model-level parallelism."""
    if "n_jobs" not in lane_cfg or lane_cfg["n_jobs"] is None:
        return {}
    n_jobs = int(lane_cfg["n_jobs"])
    return {"n_jobs": n_jobs}


def item_dir(root: Path, lane_name: str, lane_cfg: Mapping[str, Any], seed: int) -> Path:
    subdir = str(lane_cfg.get("output_subdir") or lane_name)
    return root / subdir / f"seed_{int(seed)}"


def runs_root_for_item(root: Path, lane_name: str, lane_cfg: Mapping[str, Any], seed: int) -> Path:
    return item_dir(root, lane_name, lane_cfg, seed) / "runs"


def as_legacy_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _set_nested(cfg: dict[str, Any], path: Sequence[str], updates: Mapping[str, Any]) -> None:
    target: dict[str, Any] = cfg
    for key in path:
        value = target.setdefault(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"Cannot apply overrides below non-mapping config key: {'.'.join(path)}")
        target = value
    deep_update(target, updates)


def prepare_lane_config(
    lane_name: str,
    lane_cfg: Mapping[str, Any],
    seed: int,
    runs_root: Path,
) -> dict[str, Any]:
    base_config = lane_cfg.get("base_config")
    if not base_config:
        raise ValueError(f"Lane {lane_name} is missing base_config")
    cfg = load_config(str(base_config))
    legacy_runs_root = as_legacy_path(runs_root)
    runner = str(lane_cfg.get("runner", lane_name))
    parallelism = lane_parallelism_overrides(lane_cfg)

    if runner == "protocol_a_two_stage":
        updates = {"random_seed": int(seed), "runs_root": legacy_runs_root}
        updates.update(parallelism)
        _set_nested(
            cfg,
            ["two_stage_engine", "legacy_overrides"],
            updates,
        )
    elif runner == "protocol_a_flat":
        updates = {"random_seed": int(seed), "runs_root": legacy_runs_root}
        updates.update(parallelism)
        _set_nested(
            cfg,
            ["flat_multiclass_baseline", "legacy_overrides"],
            updates,
        )
    elif runner == "protocol_b_loao":
        updates = {"random_seed": int(seed), "runs_root": legacy_runs_root}
        updates.update(parallelism)
        if "n_jobs" in parallelism:
            updates["xgb_binary_defaults"] = {"n_jobs": int(parallelism["n_jobs"])}
            updates["xgb_multi_defaults"] = {"n_jobs": int(parallelism["n_jobs"])}
        _set_nested(
            cfg,
            ["protocol_b_grid", "legacy_overrides"],
            updates,
        )
        _set_nested(
            cfg,
            ["protocol_b_summary"],
            {"run_after_grid": True},
        )
        _set_nested(
            cfg,
            ["protocol_b_summary", "legacy_overrides"],
            {
                "aggregate_csv": as_legacy_path(runs_root / "aggregate_results.csv"),
                "out_dir": as_legacy_path(runs_root / "summary"),
            },
        )
    else:
        raise ValueError(f"Unsupported seed reliability runner: {runner}")

    return cfg


def run_lane(lane_name: str, lane_cfg: Mapping[str, Any], cfg: Mapping[str, Any], *, dry_run: bool) -> None:
    runner = str(lane_cfg.get("runner", lane_name))
    if runner == "protocol_a_two_stage":
        from ids_eval_framework.src import two_stage_engine

        two_stage_engine.run_protocol_a_core(dict(cfg), dry_run=dry_run)
    elif runner == "protocol_a_flat":
        from ids_eval_framework.src.flat_multiclass_baseline import run_protocol_a_flat_baseline

        run_protocol_a_flat_baseline(dict(cfg), dry_run=dry_run)
    elif runner == "protocol_b_loao":
        from ids_eval_framework.src import two_stage_engine

        two_stage_engine.run_protocol_b_loao_grid(dict(cfg), dry_run=dry_run)
    else:
        raise ValueError(f"Unsupported seed reliability runner: {runner}")


def infer_lane_complete(lane_cfg: Mapping[str, Any], runs_root: Path) -> bool:
    runner = str(lane_cfg.get("runner"))
    candidates: list[Path]
    if runner == "protocol_a_two_stage":
        candidates = [runs_root / "summary" / "protocol_a_core_summary.csv"]
    elif runner == "protocol_a_flat":
        candidates = [
            runs_root / "summary" / "run_complete.json",
            runs_root / "summary" / "winner_test_results.csv",
        ]
    elif runner == "protocol_b_loao":
        candidates = [
            runs_root / "aggregate_results.csv",
            runs_root / "summary" / "best_per_holdout.csv",
        ]
    else:
        return False

    for path in candidates:
        if not path.exists():
            return False
        if path.suffix.lower() == ".csv":
            try:
                if pd.read_csv(path).empty:
                    return False
            except Exception:
                return False
    return True


def item_state(
    lane_cfg: Mapping[str, Any],
    item_path: Path,
    runs_root: Path,
    *,
    retry_failed: bool = False,
) -> str:
    if (item_path / STATE_FILES["complete"]).exists():
        return "complete"
    if infer_lane_complete(lane_cfg, runs_root):
        return "recoverable_complete"
    if (item_path / STATE_FILES["lock"]).exists():
        return "locked"
    if (item_path / STATE_FILES["error"]).exists() and not retry_failed:
        return "failed"
    return "pending"


def acquire_lock(item_path: Path, *, force_unlock: bool = False) -> Path | None:
    item_path.mkdir(parents=True, exist_ok=True)
    lock_path = item_path / STATE_FILES["lock"]
    if lock_path.exists():
        if not force_unlock:
            return None
        lock_path.unlink()
    payload = {
        "created_at": now_iso(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "force_unlock_hint": "Use --force-unlock only after confirming no run is active.",
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(str(lock_path), flags)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return lock_path


def release_lock(lock_path: Path | None) -> None:
    if lock_path is not None and lock_path.exists():
        lock_path.unlink()


def build_run_manifest(
    lane_name: str,
    lane_cfg: Mapping[str, Any],
    seed: int,
    item_path: Path,
    runs_root: Path,
    prepared_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "lane": lane_name,
        "runner": str(lane_cfg.get("runner", lane_name)),
        "seed": int(seed),
        "base_config": str(lane_cfg.get("base_config", "")),
        "item_dir": str(item_path),
        "runs_root": str(runs_root),
        "runs_root_legacy": as_legacy_path(runs_root),
        "manifest_written_at": now_iso(),
        "prepared_config": prepared_cfg,
    }


def recover_complete_marker(
    lane_name: str,
    lane_cfg: Mapping[str, Any],
    seed: int,
    item_path: Path,
    runs_root: Path,
) -> None:
    write_json(
        item_path / STATE_FILES["complete"],
        {
            "lane": lane_name,
            "runner": str(lane_cfg.get("runner", lane_name)),
            "seed": int(seed),
            "completed_at": now_iso(),
            "status": "complete_recovered_from_artifacts",
            "runs_root": str(runs_root),
        },
    )


def run_seed_item(
    lane_name: str,
    lane_cfg: Mapping[str, Any],
    seed: int,
    root: Path,
    *,
    dry_run: bool = False,
    retry_failed: bool = False,
    force_unlock: bool = False,
    skip_existing: bool = True,
) -> dict[str, Any]:
    item_path = item_dir(root, lane_name, lane_cfg, seed)
    runs_root = runs_root_for_item(root, lane_name, lane_cfg, seed)
    prepared_cfg = prepare_lane_config(lane_name, lane_cfg, seed, runs_root)
    state = item_state(lane_cfg, item_path, runs_root, retry_failed=retry_failed)

    if dry_run:
        progress_print(
            f"[dry-run] lane={lane_name} seed={seed} state={state} "
            f"runs_root={as_legacy_path(runs_root)}"
        )
        run_lane(lane_name, lane_cfg, prepared_cfg, dry_run=True)
        return {"lane": lane_name, "seed": int(seed), "status": "dry_run", "state": state}

    if state == "recoverable_complete":
        recover_complete_marker(lane_name, lane_cfg, seed, item_path, runs_root)
        state = "complete"

    if state == "complete" and skip_existing:
        return {"lane": lane_name, "seed": int(seed), "status": "skipped_complete"}
    if state == "failed" and not retry_failed:
        return {"lane": lane_name, "seed": int(seed), "status": "skipped_failed"}
    if state == "locked" and not force_unlock:
        return {"lane": lane_name, "seed": int(seed), "status": "skipped_locked"}

    lock_path = acquire_lock(item_path, force_unlock=force_unlock)
    if lock_path is None:
        return {"lane": lane_name, "seed": int(seed), "status": "skipped_locked"}

    started = time.perf_counter()
    write_json(
        item_path / STATE_FILES["manifest"],
        build_run_manifest(lane_name, lane_cfg, seed, item_path, runs_root, prepared_cfg),
    )
    write_json(
        item_path / STATE_FILES["started"],
        {
            "lane": lane_name,
            "seed": int(seed),
            "started_at": now_iso(),
            "runs_root": str(runs_root),
        },
    )

    try:
        error_path = item_path / STATE_FILES["error"]
        if retry_failed and error_path.exists():
            error_path.unlink()
        run_lane(lane_name, lane_cfg, prepared_cfg, dry_run=False)
        if not infer_lane_complete(lane_cfg, runs_root):
            raise RuntimeError(
                f"Lane {lane_name} seed {seed} finished without expected completion artifacts."
            )
        write_json(
            item_path / STATE_FILES["complete"],
            {
                "lane": lane_name,
                "runner": str(lane_cfg.get("runner", lane_name)),
                "seed": int(seed),
                "completed_at": now_iso(),
                "elapsed_seconds": float(time.perf_counter() - started),
                "status": "complete",
                "runs_root": str(runs_root),
            },
        )
        if error_path.exists():
            error_path.unlink()
        return {"lane": lane_name, "seed": int(seed), "status": "complete"}
    except Exception as exc:
        write_json(
            item_path / STATE_FILES["error"],
            {
                "lane": lane_name,
                "seed": int(seed),
                "failed_at": now_iso(),
                "elapsed_seconds": float(time.perf_counter() - started),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return {"lane": lane_name, "seed": int(seed), "status": "failed", "error": str(exc)}
    finally:
        release_lock(lock_path)


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _resolve_run_dir(value: Any, runs_root: Path) -> Path | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [
        REPO_ROOT / path,
        FRAMEWORK_ROOT / path,
        runs_root / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _add_common_columns(
    df: pd.DataFrame,
    *,
    lane_name: str,
    lane_cfg: Mapping[str, Any],
    seed: int,
    item_path: Path,
    runs_root: Path,
    source_csv: Path,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, "lane", lane_name)
    out.insert(1, "seed", int(seed))
    out["runner"] = str(lane_cfg.get("runner", lane_name))
    out["item_dir"] = str(item_path)
    out["seed_runs_root"] = str(runs_root)
    out["source_csv"] = str(source_csv)
    out["item_status"] = "complete" if (item_path / STATE_FILES["complete"]).exists() else "artifact_complete"
    return add_comparison_key(out, lane_name, runs_root)


def add_comparison_key(df: pd.DataFrame, lane_name: str, runs_root: Path) -> pd.DataFrame:
    out = df.copy()
    if lane_name == "protocol_a_two_stage":
        parts = [
            out.get("dataset", pd.Series([""] * len(out))).astype(str),
            out.get("model_family", pd.Series([""] * len(out))).astype(str),
            out.get("policy_variant", pd.Series([""] * len(out))).astype(str),
        ]
        out["comparison_key"] = parts[0] + "::" + parts[1] + "::" + parts[2]
        run_dirs = out.get("run_dir", pd.Series([None] * len(out)))
        out["calibration_stage1"] = [
            "platt" if (resolved and (resolved / "stage1_platt.json").exists()) else ""
            for resolved in (_resolve_run_dir(value, runs_root) for value in run_dirs)
        ]
        out["calibration_stage2"] = [
            "temperature" if (resolved and (resolved / "stage2_temperature.json").exists()) else ""
            for resolved in (_resolve_run_dir(value, runs_root) for value in run_dirs)
        ]
    elif lane_name == "protocol_a_flat":
        parts = [
            out.get("dataset", pd.Series([""] * len(out))).astype(str),
            out.get("surface", pd.Series([""] * len(out))).astype(str),
            out.get("candidate", pd.Series([""] * len(out))).astype(str),
        ]
        out["comparison_key"] = parts[0] + "::" + parts[1] + "::" + parts[2]
        out["calibration_stage1"] = ""
        out["calibration_stage2"] = ""
    elif lane_name == "protocol_b_loao":
        parts = [
            out.get("dataset", pd.Series([""] * len(out))).astype(str),
            out.get("holdout_family", pd.Series([""] * len(out))).astype(str),
        ]
        out["comparison_key"] = parts[0] + "::" + parts[1]
        out["calibration_stage1"] = "none"
        out["calibration_stage2"] = "none"
    else:
        out["comparison_key"] = lane_name
    return out


def lane_result_rows(
    lane_name: str,
    lane_cfg: Mapping[str, Any],
    seed: int,
    root: Path,
) -> pd.DataFrame:
    item_path = item_dir(root, lane_name, lane_cfg, seed)
    runs_root = runs_root_for_item(root, lane_name, lane_cfg, seed)
    runner = str(lane_cfg.get("runner", lane_name))

    if runner == "protocol_a_two_stage":
        source = runs_root / "summary" / "protocol_a_core_summary.csv"
    elif runner == "protocol_a_flat":
        source = runs_root / "summary" / "winner_test_results.csv"
    elif runner == "protocol_b_loao":
        source = runs_root / "summary" / "best_per_holdout.csv"
    else:
        return pd.DataFrame()

    return _add_common_columns(
        _read_csv_if_present(source),
        lane_name=lane_name,
        lane_cfg=lane_cfg,
        seed=seed,
        item_path=item_path,
        runs_root=runs_root,
        source_csv=source,
    )


def failed_or_locked_rows(
    lane_name: str,
    lane_cfg: Mapping[str, Any],
    seed: int,
    root: Path,
) -> list[dict[str, Any]]:
    item_path = item_dir(root, lane_name, lane_cfg, seed)
    rows: list[dict[str, Any]] = []
    if (item_path / STATE_FILES["error"]).exists() and not (item_path / STATE_FILES["complete"]).exists():
        err = read_json(item_path / STATE_FILES["error"])
        rows.append(
            {
                "lane": lane_name,
                "seed": int(seed),
                "runner": str(lane_cfg.get("runner", lane_name)),
                "item_dir": str(item_path),
                "seed_runs_root": str(runs_root_for_item(root, lane_name, lane_cfg, seed)),
                "item_status": "failed",
                "error": err.get("error", ""),
                "comparison_key": lane_name,
            }
        )
    elif (item_path / STATE_FILES["lock"]).exists() and not (item_path / STATE_FILES["complete"]).exists():
        rows.append(
            {
                "lane": lane_name,
                "seed": int(seed),
                "runner": str(lane_cfg.get("runner", lane_name)),
                "item_dir": str(item_path),
                "seed_runs_root": str(runs_root_for_item(root, lane_name, lane_cfg, seed)),
                "item_status": "locked",
                "comparison_key": lane_name,
            }
        )
    return rows


def collect_seed_reliability_runs(
    config: Mapping[str, Any] | None,
    *,
    lanes: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    root = output_root(config)
    lane_map = configured_lanes(config, lanes)
    seed_list = configured_seeds(config, seeds)
    frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    for lane_name, lane_cfg in lane_map.items():
        for seed in seed_list:
            frame = lane_result_rows(lane_name, lane_cfg, seed, root)
            if not frame.empty:
                frames.append(frame)
            status_rows.extend(failed_or_locked_rows(lane_name, lane_cfg, seed, root))
    if frames:
        runs = pd.concat(frames, ignore_index=True, sort=False)
        if status_rows:
            runs = pd.concat([runs, pd.DataFrame(status_rows)], ignore_index=True, sort=False)
        return runs
    return pd.DataFrame(status_rows)


def t_critical_95(n: int) -> float:
    if n <= 1:
        return float("nan")
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, n - 1))
    except Exception:
        return 1.96


def summarize_metric_values(values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(len(clean))
    if n == 0:
        return {}
    mean = float(clean.mean())
    sd = float(clean.std(ddof=1)) if n > 1 else float("nan")
    half_width = t_critical_95(n) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "min": float(clean.min()),
        "max": float(clean.max()),
        "ci95_low": mean - half_width if n > 1 else float("nan"),
        "ci95_high": mean + half_width if n > 1 else float("nan"),
    }


def build_seed_reliability_summary(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    metric_cols = [col for col in PRIMARY_METRICS if col in runs.columns]
    if not metric_cols:
        return pd.DataFrame()
    if "item_status" in runs.columns:
        usable = runs[runs["item_status"] != "failed"].copy()
    else:
        usable = runs.copy()
    group_cols = ["lane", "comparison_key"]
    if "dataset" in usable.columns:
        group_cols.append("dataset")
    rows: list[dict[str, Any]] = []
    for group_values, group in usable.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        base = dict(zip(group_cols, group_values))
        for metric in metric_cols:
            stats = summarize_metric_values(group[metric])
            if stats:
                rows.append({**base, "metric": metric, **stats})
    return pd.DataFrame(rows)


def build_selection_audit(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    preferred = [
        "lane",
        "seed",
        "dataset",
        "comparison_key",
        "holdout_family",
        "model_family",
        "policy_variant",
        "surface",
        "candidate",
        "claim_status",
        "apply_loao_stage1",
        "stage1_weight_mode",
        "stage1_thr_high",
        "threshold",
        "tau",
        "unknown_detection_rate",
        "false_unknown_rate_all_known",
        "false_unknown_rate_known_attacks",
        "macro_f1",
        "system_macro_f1_supported_labels",
        "system_macro_f1_declared_output_labels_historical",
        "accuracy",
        "system_accuracy",
        "n_train_rows",
        "n_eval_rows",
        "calibration_stage1",
        "calibration_stage2",
        "run_name",
        "run_dir",
        "seed_runs_root",
        "item_status",
        "error",
    ]
    columns = [col for col in preferred if col in runs.columns]
    return runs.loc[:, columns].copy()


def write_seed_reliability_summaries(
    config: Mapping[str, Any] | None,
    *,
    lanes: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> dict[str, Path]:
    root = output_root(config)
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_seed_reliability_runs(config, lanes=lanes, seeds=seeds)
    summary = build_seed_reliability_summary(runs)
    selection = build_selection_audit(runs)
    paths = {
        "runs": summary_dir / "seed_reliability_runs.csv",
        "summary": summary_dir / "seed_reliability_summary.csv",
        "selection_audit": summary_dir / "seed_reliability_selection_audit.csv",
    }
    runs.to_csv(paths["runs"], index=False)
    summary.to_csv(paths["summary"], index=False)
    selection.to_csv(paths["selection_audit"], index=False)
    return paths


def run_seed_reliability(
    config: Mapping[str, Any] | None,
    *,
    lanes: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    dry_run: bool = False,
    retry_failed: bool = False,
    force_unlock: bool = False,
    skip_existing: bool = True,
    summary_only: bool = False,
) -> list[dict[str, Any]]:
    root = output_root(config)
    lane_map = configured_lanes(config, lanes)
    seed_list = configured_seeds(config, seeds)
    if dry_run:
        progress_print(f"[dry-run] seed reliability output_root={root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_seed_reliability_summaries(config)

    if summary_only:
        if not dry_run:
            paths = write_seed_reliability_summaries(config)
            progress_print(f"Wrote seed reliability summaries to: {paths['summary'].parent}")
        return []

    results: list[dict[str, Any]] = []
    for lane_name, lane_cfg in lane_map.items():
        for seed in seed_list:
            progress_print(f"[seed-reliability] starting {lane_name} seed={seed}")
            result = run_seed_item(
                lane_name,
                lane_cfg,
                seed,
                root,
                dry_run=dry_run,
                retry_failed=retry_failed,
                force_unlock=force_unlock,
                skip_existing=skip_existing,
            )
            results.append(result)
            progress_print(f"[seed-reliability] {lane_name} seed={seed}: {result['status']}")

    if not dry_run:
        paths = write_seed_reliability_summaries(config)
        progress_print(f"Wrote seed reliability summaries to: {paths['summary'].parent}")
    return results


def parse_csv_arg(values: Iterable[str] | None) -> list[str] | None:
    if not values:
        return None
    parsed: list[str] = []
    for value in values:
        parsed.extend([part.strip() for part in str(value).split(",") if part.strip()])
    return parsed or None


def parse_seed_args(values: Iterable[str] | None) -> list[int] | None:
    parsed = parse_csv_arg(values)
    if not parsed:
        return None
    return [int(value) for value in parsed]
