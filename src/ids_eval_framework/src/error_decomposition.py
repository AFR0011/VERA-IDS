"""Stage-wise and end-to-end error decomposition helpers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd


def label_error_counts(y_true: Iterable[str], y_pred: Iterable[str]) -> pd.DataFrame:
    """Return sorted counts for true/predicted label pairs."""
    counts = Counter(zip(map(str, y_true), map(str, y_pred)))
    rows = [
        {"true_label": true, "pred_label": pred, "count": count}
        for (true, pred), count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(["count", "true_label", "pred_label"], ascending=[False, True, True])


def stage_error_summary(
    y_stage1_true: Iterable[int],
    y_stage1_pred: Iterable[int],
    y_system_true: Iterable[str],
    y_system_pred: Iterable[str],
    *,
    benign_label: str = "Benign",
    unknown_label: str = "Unknown",
) -> dict[str, int]:
    """Summarize common IDS two-stage failure modes."""
    s1_true = list(map(int, y_stage1_true))
    s1_pred = list(map(int, y_stage1_pred))
    sys_true = list(map(str, y_system_true))
    sys_pred = list(map(str, y_system_pred))
    return {
        "stage1_false_negatives": sum(1 for t, p in zip(s1_true, s1_pred) if t == 1 and p == 0),
        "stage1_false_positives": sum(1 for t, p in zip(s1_true, s1_pred) if t == 0 and p == 1),
        "stage2_family_errors": sum(
            1
            for t, p in zip(sys_true, sys_pred)
            if t not in {benign_label, unknown_label} and p not in {benign_label, unknown_label} and t != p
        ),
        "unknown_predictions": sum(1 for p in sys_pred if p == unknown_label),
        "benign_false_alarms": sum(1 for t, p in zip(sys_true, sys_pred) if t == benign_label and p != benign_label),
    }
