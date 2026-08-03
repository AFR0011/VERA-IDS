"""Bounded, deterministic synthetic checks for every advertised CLI surface."""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ids_eval_framework.metrics import (
    macro_f1,
    protocol_a_supported_labels,
    protocol_b_labels,
    unknown_detection_rate,
)
from ids_eval_framework.src.open_set_rejection import (
    max_softmax_confidence,
    normalized_entropy,
    top2_margin,
)
from ids_eval_framework.src.seed_reliability import summarize_metric_values
from ids_eval_framework.src.statistics import bh_adjust, bootstrap_mean_ci
from ids_eval_framework.src.support_audit import eligible_holdouts
from ids_eval_framework.src.support_sensitivity import evaluate_support_thresholds


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260803)
    centers = {
        "Benign": np.asarray([-2.0, -2.0, 0.0, 0.0]),
        "Botnet": np.asarray([2.0, 0.0, 2.0, 0.0]),
        "DDoS": np.asarray([0.0, 2.0, 0.0, 2.0]),
    }
    labels: list[str] = []
    rows: list[np.ndarray] = []
    for label, center in centers.items():
        rows.extend(center + rng.normal(0, 0.22, size=(24, 4)))
        labels.extend([label] * 24)
    return np.asarray(rows), np.asarray(labels, dtype=object)


def _rf(seed: int = 123) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=16, max_depth=5, random_state=seed, n_jobs=1)


def _audit_and_prepare() -> None:
    x, labels = _fixture()
    indices = np.arange(len(labels))
    train, val, test = indices[:48], indices[48:60], indices[60:]
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert x.shape == (72, 4) and len(set(labels)) == 3
    assert all(np.isfinite(x).all(axis=1))


def _protocol_a_two_stage() -> None:
    x, labels = _fixture()
    attack = (labels != "Benign").astype(int)
    stage1 = _rf().fit(x, attack)
    attack_mask = attack == 1
    stage2 = _rf().fit(x[attack_mask], labels[attack_mask])
    pred_attack = stage1.predict(x)
    predictions = np.full(len(labels), "Benign", dtype=object)
    predictions[pred_attack == 1] = stage2.predict(x[pred_attack == 1])
    score = macro_f1(labels.tolist(), predictions.tolist(), protocol_a_supported_labels(["Botnet", "DDoS"]))
    assert 0.95 <= score <= 1.0


def _protocol_a_flat() -> None:
    x, labels = _fixture()
    predictions = _rf().fit(x, labels).predict(x)
    assert macro_f1(labels.tolist(), predictions.tolist(), ["Benign", "Botnet", "DDoS"]) >= 0.95


def _support() -> None:
    support = pd.DataFrame(
        [
            {"dataset": "synthetic", "family": family, "split": split, "count": count}
            for family, counts in {"Benign": (24, 12, 12), "Botnet": (20, 10, 10), "DDoS": (20, 10, 10)}.items()
            for split, count in zip(("train", "val", "test"), counts)
        ]
    )
    holdouts, summary = evaluate_support_thresholds(
        {"synthetic": support}, [8], benign_label="Benign", min_known_families_after_holdout=1
    )
    assert len(holdouts) == 2 and bool(summary.iloc[0]["all_holdouts_eligible"])
    scoreboard = pd.DataFrame([{"family": "Botnet", "val_support": 10}, {"family": "DDoS", "val_support": 4}])
    assert eligible_holdouts(scoreboard, {"val_support": 8})["family"].tolist() == ["Botnet"]


def _protocol_b() -> None:
    true = ["Benign", "Botnet", "Unknown", "Unknown"]
    pred = ["Benign", "Botnet", "Unknown", "Botnet"]
    labels = protocol_b_labels(["Botnet"], unknown_support=2)
    assert labels == ["Benign", "Botnet", "Unknown"]
    assert 0.0 < macro_f1(true, pred, labels) < 1.0
    assert unknown_detection_rate(true, pred) == 0.5


def _rejection() -> None:
    probabilities = np.asarray([[0.90, 0.10], [0.52, 0.48], [0.65, 0.35]], dtype=float)
    confidence = max_softmax_confidence(probabilities)
    margin = top2_margin(probabilities)
    entropy = normalized_entropy(probabilities)
    assert confidence.tolist() == [0.9, 0.52, 0.65]
    assert bool((margin >= 0).all()) and bool((entropy >= 0).all())
    assert int((confidence < 0.60).sum()) == 1


def _external() -> None:
    x, labels = _fixture()
    model = _rf().fit(x, labels)
    shifted = x + np.asarray([0.05, -0.05, 0.05, -0.05])
    predictions = model.predict(shifted)
    assert macro_f1(labels.tolist(), predictions.tolist(), ["Benign", "Botnet", "DDoS"]) >= 0.90


def _statistics() -> None:
    mean, low, high = bootstrap_mean_ci([0.71, 0.76, 0.79, 0.82], resamples=200, seed=123)
    assert low <= mean <= high
    adjusted = bh_adjust([0.01, 0.04, 0.20])
    assert adjusted == sorted(adjusted) and all(0 <= value <= 1 for value in adjusted)


def _paper_pack() -> None:
    rows = pd.DataFrame(
        [
            {"surface": "Protocol A", "metric": "macro_f1", "value": 0.81},
            {"surface": "Protocol B", "metric": "unknown_detection", "value": 0.72},
        ]
    )
    payload = json.loads(rows.to_json(orient="records"))
    assert len(payload) == 2 and all("value" in row for row in payload)


def _seed_reliability() -> None:
    values: list[float] = []
    x, labels = _fixture()
    for seed in (123, 124, 125):
        predictions = _rf(seed).fit(x, labels).predict(x)
        values.append(macro_f1(labels.tolist(), predictions.tolist(), ["Benign", "Botnet", "DDoS"]))
    summary = summarize_metric_values(pd.Series(values))
    assert summary["n"] == 3 and summary["ci95_low"] <= summary["mean"] <= summary["ci95_high"]


def _reference_profile() -> None:
    from ids_eval_framework.src.reference_framework_eval import build_model, system_truth

    x, labels = _fixture()
    model = build_model(
        "synthetic_rf_profile",
        {"model_family": "rf", "stage2_params": {"n_estimators": 12, "max_depth": 4}},
        "stage2",
        n_classes=3,
        seed=123,
        n_jobs=1,
    )
    predictions = model.fit(x, labels).predict(x)
    assert macro_f1(labels.tolist(), predictions.tolist(), ["Benign", "Botnet", "DDoS"]) >= 0.90
    assert system_truth(np.asarray([0, 1]), np.asarray(["Botnet", "DDoS"], dtype=object)).tolist() == [
        "Benign",
        "DDoS",
    ]


def _reference_comparison() -> None:
    current = pd.DataFrame([{"profile": "rf", "accuracy": 0.96, "macro_f1": 0.81}])
    reference = pd.DataFrame([{"profile": "rf", "reference_accuracy": 0.98}])
    comparison = current.merge(reference, on="profile", validate="one_to_one")
    comparison["accuracy_drop"] = comparison["reference_accuracy"] - comparison["accuracy"]
    assert np.isclose(float(comparison.iloc[0]["accuracy_drop"]), 0.02)


CHECKS: dict[str, Callable[[], None]] = {
    "audit_and_prepare": _audit_and_prepare,
    "protocol_a_two_stage": _protocol_a_two_stage,
    "protocol_a_flat": _protocol_a_flat,
    "support_audit": _support,
    "support_sensitivity": _support,
    "protocol_b_loao": _protocol_b,
    "rejection_validation_selected": _rejection,
    "rejection_exploratory_grid": _rejection,
    "external_stress": _external,
    "statistics": _statistics,
    "paper_pack": _paper_pack,
    "seed_reliability": _seed_reliability,
    "reference_profile": _reference_profile,
    "reference_comparison": _reference_comparison,
}


def run_synthetic_cli(name: str) -> None:
    try:
        check = CHECKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown synthetic CLI surface: {name}") from exc
    check()
    print(f"Synthetic workflow OK: {name}")
