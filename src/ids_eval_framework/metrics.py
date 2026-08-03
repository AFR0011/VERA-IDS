"""Explicit metric surfaces used by the public release tests and documentation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence


BENIGN = "Benign"
UNKNOWN = "Unknown"


def _deduplicated(labels: Iterable[str]) -> list[str]:
    result: list[str] = []
    for label in labels:
        value = str(label)
        if value not in result:
            result.append(value)
    return result


def stage2_fixed_k_labels(families: Sequence[str]) -> list[str]:
    """Return every family learned by Stage 2, including absent test families."""
    return _deduplicated(families)


def stage2_present_family_labels(y_true: Iterable[str]) -> list[str]:
    """Return only attack-family labels with ground-truth support."""
    return sorted(set(str(value) for value in y_true))


def protocol_a_supported_labels(families: Sequence[str]) -> list[str]:
    """Closed-set Protocol A labels with genuine ground-truth support."""
    return [BENIGN, *[label for label in _deduplicated(families) if label != BENIGN]]


def protocol_a_declared_output_labels(families: Sequence[str]) -> list[str]:
    """Protocol A output vocabulary when abstention can emit unsupported Unknown."""
    return [*protocol_a_supported_labels(families), UNKNOWN]


def protocol_b_labels(known_families: Sequence[str], unknown_support: int) -> list[str]:
    """Return Protocol B labels only when Unknown has genuine ground-truth support."""
    if unknown_support <= 0:
        raise ValueError("Protocol B Unknown metrics require genuine ground-truth support")
    return [BENIGN, *[label for label in _deduplicated(known_families) if label != BENIGN], UNKNOWN]


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Compute zero-division-safe macro-F1 over an explicit declared label set."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not labels:
        raise ValueError("labels must not be empty")
    scores: list[float] = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return sum(scores) / len(scores)


def unknown_detection_rate(
    y_true: Sequence[str], y_pred: Sequence[str], unknown_label: str = UNKNOWN
) -> float:
    """Measure Unknown recall and reject unsupported interpretations."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    support = Counter(y_true)[unknown_label]
    if support == 0:
        raise ValueError("Unknown detection rate is undefined without true Unknown rows")
    detected = sum(t == unknown_label and p == unknown_label for t, p in zip(y_true, y_pred))
    return detected / support
