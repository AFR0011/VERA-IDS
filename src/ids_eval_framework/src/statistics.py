"""Statistical validation helpers and legacy statistics entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ids_eval_framework.src.native_runtime import run_native_main
from ids_eval_framework.src.paths import REPO_ROOT, deep_update, resolve_repo_path


def bh_adjust(p_values: Iterable[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(list(p_values), dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        running = min(running, p[idx] * n / original_rank)
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_mean_ci(values: Iterable[float], *, resamples: int = 2000, seed: int = 123, alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(resamples, arr.size), replace=True).mean(axis=1)
    return (
        float(arr.mean()),
        float(np.quantile(samples, alpha / 2.0)),
        float(np.quantile(samples, 1.0 - alpha / 2.0)),
    )


def run_protocol_b_statistics(config: Mapping[str, Any] | None = None, *, dry_run: bool = False) -> None:
    from ids_eval_framework._native import protocol_b_validation_selected

    cfg = config or {}
    stats_cfg = cfg.get("statistics", {}) or {}
    run_native_main(
        protocol_b_validation_selected,
        cfg_overrides=_validation_selected_overrides(stats_cfg.get("validation_selected_legacy_overrides")),
        dry_run=dry_run,
    )
    _run_q2_statistics(stats_cfg.get("q2_statistics_legacy_overrides"), dry_run=dry_run)


def _validation_selected_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not overrides:
        return None
    result = dict(overrides)
    if "aggregate_csvs" in result:
        result["aggregate_csvs"] = [Path(resolve_repo_path(str(path))) for path in result["aggregate_csvs"]]
    if "out_root" in result:
        result["out_root"] = Path(resolve_repo_path(str(result["out_root"])))
    if "run_roots" in result:
        result["run_roots"] = {key: Path(resolve_repo_path(str(value))) for key, value in result["run_roots"].items()}
    return result


def _q2_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not overrides:
        return None
    result = dict(overrides)
    for key in ("source_root", "out_root"):
        if key in result:
            result[key] = Path(resolve_repo_path(str(result[key])))
    return result


def _run_q2_statistics(overrides: Mapping[str, Any] | None, *, dry_run: bool = False) -> None:
    from ids_eval_framework._native import protocol_b_q2_statistics

    old_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        module = protocol_b_q2_statistics
        patched = _q2_overrides(overrides)
        if patched:
            deep_update(module.CFG, patched)

        if dry_run:
            print(f"[dry-run] native module: {module.__name__}")
            print(f"[dry-run] cwd={REPO_ROOT}")
            print(f"[dry-run] CFG keys: {sorted(module.CFG.keys())}")
            return
        module.main()
    finally:
        os.chdir(old_cwd)
