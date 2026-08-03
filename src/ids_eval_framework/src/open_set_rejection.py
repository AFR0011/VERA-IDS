"""Open-set, abstention, and rejector entrypoints."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ids_eval_framework.src.native_runtime import run_native_main


def max_softmax_confidence(probs: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(probs, dtype=float), axis=1)


def top2_margin(probs: np.ndarray) -> np.ndarray:
    arr = np.sort(np.asarray(probs, dtype=float), axis=1)
    if arr.shape[1] < 2:
        return np.ones(arr.shape[0], dtype=float)
    return arr[:, -1] - arr[:, -2]


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    arr = np.clip(arr, 1e-12, 1.0)
    entropy = -np.sum(arr * np.log(arr), axis=1)
    denom = np.log(arr.shape[1]) if arr.shape[1] > 1 else 1.0
    return entropy / denom


def run_open_set_baselines(config: Mapping[str, Any] | None = None, *, dry_run: bool = False) -> None:
    from ids_eval_framework._native import protocol_b_open_set

    cfg = config or {}
    overrides = (cfg.get("open_set_rejection", {}) or {}).get("legacy_overrides", {}).get("open_set_baselines")
    run_native_main(
        protocol_b_open_set,
        cfg_overrides=overrides,
        dry_run=dry_run,
    )


def run_sink_aware_rejector(config: Mapping[str, Any] | None = None, *, dry_run: bool = False) -> None:
    from ids_eval_framework._native import protocol_b_sink_aware

    cfg = config or {}
    overrides = (cfg.get("open_set_rejection", {}) or {}).get("legacy_overrides", {}).get("sink_aware_rejector")
    run_native_main(
        protocol_b_sink_aware,
        cfg_overrides=overrides,
        dry_run=dry_run,
    )
