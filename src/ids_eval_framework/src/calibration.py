"""Calibration utilities used by the evaluation/reporting layers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def brier_binary(y_true: Iterable[int], p_positive: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(p_positive), dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true: Iterable[int], p_positive: Iterable[float], *, bins: int = 15) -> float:
    """Compute binary ECE with fixed-width confidence bins."""
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(p_positive), dtype=float)
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high if high < 1.0 else p <= high)
        if not np.any(mask):
            continue
        acc = np.mean(y[mask])
        conf = np.mean(p[mask])
        ece += float(mask.mean() * abs(acc - conf))
    return float(ece)


def reliability_curve(y_true: Iterable[int], p_positive: Iterable[float], *, bins: int = 15) -> list[dict[str, float]]:
    """Return bin-level reliability data for plotting/reporting."""
    y = np.asarray(list(y_true), dtype=int)
    p = np.asarray(list(p_positive), dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (p >= low) & (p < high if high < 1.0 else p <= high)
        n = int(mask.sum())
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "n": float(n),
                "accuracy": float(np.mean(y[mask])) if n else float("nan"),
                "confidence": float(np.mean(p[mask])) if n else float("nan"),
            }
        )
    return rows
