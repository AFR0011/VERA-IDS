from __future__ import annotations

import pytest

from ids_eval_framework.metrics import (
    macro_f1,
    protocol_a_declared_output_labels,
    protocol_a_supported_labels,
    protocol_b_labels,
    stage2_fixed_k_labels,
    stage2_present_family_labels,
    unknown_detection_rate,
)


def test_label_sets_are_not_interchangeable() -> None:
    families = ["Botnet", "DDoS"]
    assert stage2_fixed_k_labels(families) == ["Botnet", "DDoS"]
    assert stage2_present_family_labels(["Botnet", "Botnet"]) == ["Botnet"]
    assert protocol_a_supported_labels(families) == ["Benign", "Botnet", "DDoS"]
    assert protocol_a_declared_output_labels(families) == ["Benign", "Botnet", "DDoS", "Unknown"]
    assert protocol_b_labels(families, unknown_support=1) == ["Benign", "Botnet", "DDoS", "Unknown"]


def test_unknown_metrics_require_ground_truth_support() -> None:
    with pytest.raises(ValueError, match="genuine ground-truth support"):
        protocol_b_labels(["Botnet"], unknown_support=0)
    with pytest.raises(ValueError, match="undefined"):
        unknown_detection_rate(["Benign", "Botnet"], ["Benign", "Unknown"])


def test_explicit_label_set_changes_macro_f1_denominator() -> None:
    y_true = ["Benign", "Botnet"]
    y_pred = ["Benign", "Botnet"]
    supported = macro_f1(y_true, y_pred, ["Benign", "Botnet"])
    declared = macro_f1(y_true, y_pred, ["Benign", "Botnet", "Unknown"])
    assert supported == 1.0
    assert declared == pytest.approx(2 / 3)
