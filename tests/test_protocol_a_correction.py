from __future__ import annotations

from pathlib import Path

import pytest

from ids_eval_framework.src.protocol_a_correction import corrected_metrics, load_evidence


ROOT = Path(__file__).resolve().parents[1]


def records():
    return load_evidence(ROOT / "outputs" / "evidence" / "protocol_a_confusion_matrices.jsonl")


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("CICIDS2017", 0.818902771736530),
        ("CICIoT2023", 0.897154513910465),
    ],
)
def test_required_rf_strict_values(dataset: str, expected: float) -> None:
    record = next(
        item
        for item in records()
        if item["surface"] == "core"
        and item["dataset"] == dataset
        and item["model_family"] == "rf"
        and item["policy_variant"] == "strict"
    )
    metrics = corrected_metrics(record)
    assert metrics["system_macro_f1_supported_labels"] == pytest.approx(expected, abs=5e-16)
    assert metrics["system_macro_f1_supported_labels"] > metrics[
        "system_macro_f1_declared_output_labels_historical"
    ]


def test_evidence_cardinality_and_supported_unknown_policy() -> None:
    evidence = records()
    assert len(evidence) == 136
    assert len({item["run_id"] for item in evidence}) == 34
    assert {item["policy_variant"] for item in evidence} == {
        "strict",
        "cascade",
        "strict_tau",
        "cascade_tau",
    }
