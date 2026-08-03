"""Deterministic Protocol A macro-F1 correction from aggregate matrices."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


POLICIES = ("strict", "cascade", "strict_tau", "cascade_tau")
UNKNOWN = "Unknown"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_confusion_matrix(path: Path) -> tuple[list[str], list[list[int]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"Invalid confusion matrix: {path}")
    labels = [str(value) for value in rows[0][1:]]
    if [row[0] for row in rows[1:]] != labels:
        raise ValueError(f"Row/column labels differ: {path}")
    matrix = [[int(value) for value in row[1:]] for row in rows[1:]]
    if len(matrix) != len(labels) or any(len(row) != len(labels) for row in matrix):
        raise ValueError(f"Confusion matrix is not square: {path}")
    if any(value < 0 for row in matrix for value in row):
        raise ValueError(f"Negative confusion count: {path}")
    return labels, matrix


def macro_f1_from_matrix(
    labels: Sequence[str], matrix: Sequence[Sequence[int]], averaged_labels: Sequence[str]
) -> float:
    """Average per-class F1 while retaining every prediction column in FN/FP."""
    index = {label: idx for idx, label in enumerate(labels)}
    missing = [label for label in averaged_labels if label not in index]
    if missing:
        raise ValueError(f"Averaged labels missing from matrix: {missing}")
    scores: list[float] = []
    for label in averaged_labels:
        idx = index[label]
        tp = int(matrix[idx][idx])
        fp = sum(int(matrix[row][idx]) for row in range(len(labels))) - tp
        fn = sum(int(value) for value in matrix[idx]) - tp
        denominator = (2 * tp) + fp + fn
        scores.append(0.0 if denominator == 0 else (2.0 * tp) / denominator)
    return sum(scores) / len(scores)


def corrected_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    labels = list(record["labels"])
    matrix = list(record["matrix"])
    unknown_idx = labels.index(UNKNOWN) if UNKNOWN in labels else None
    if unknown_idx is not None and sum(int(value) for value in matrix[unknown_idx]) != 0:
        raise ValueError("Protocol A evidence unexpectedly has true Unknown support")
    supported = [label for label in labels if label != UNKNOWN]
    return {
        "system_macro_f1_supported_labels": macro_f1_from_matrix(labels, matrix, supported),
        "system_macro_f1_declared_output_labels_historical": macro_f1_from_matrix(labels, matrix, labels),
    }


def _surface(relative: str) -> str:
    normalized = relative.replace("\\", "/")
    if normalized.startswith("outputs/03_protocol_a_two_stage/"):
        return "core"
    if normalized.startswith("outputs/07_external_stress/"):
        return "external"
    if normalized.startswith("outputs/10_seed_reliability/"):
        return "seed_reliability"
    if normalized.startswith("outputs/11_reference_framework_eval/"):
        return "reference_profile"
    raise ValueError(f"Unexpected Protocol A source surface: {relative}")


def _metadata(source_root: Path, path: Path, policy: str) -> dict[str, Any]:
    relative = path.relative_to(source_root).as_posix()
    run_name = path.parent.name
    surface = _surface(relative)
    seed_match = re.search(r"/seed_(\d+)/", f"/{relative}")
    if surface == "reference_profile":
        parts = run_name.split("__")
        if len(parts) < 3:
            raise ValueError(f"Unparseable reference run name: {run_name}")
        profile, dataset = parts[0], parts[1]
        model_family = "xgb" if "xgb" in profile.lower() else "rf" if "rf" in profile.lower() else profile
    else:
        match = re.match(r"A_stratified__(.+?)__(rf|xgb)__", run_name)
        if not match:
            raise ValueError(f"Unparseable Protocol A run name: {run_name}")
        dataset, model_family = match.groups()
        profile = None
    return {
        "surface": surface,
        "run_id": run_name,
        "dataset": dataset,
        "model_family": model_family,
        "profile": profile,
        "seed": int(seed_match.group(1)) if seed_match else None,
        "policy_variant": policy,
    }


def build_evidence(source_root: Path) -> list[dict[str, Any]]:
    strict_files = sorted(source_root.rglob("confusion_matrix_system_strict_test.csv"))
    records: list[dict[str, Any]] = []
    for strict in strict_files:
        for policy in POLICIES:
            path = strict.with_name(f"confusion_matrix_system_{policy}_test.csv")
            if not path.exists():
                raise FileNotFoundError(f"Missing policy matrix beside {strict.name}: {policy}")
            labels, matrix = read_confusion_matrix(path)
            record = {
                **_metadata(source_root, path, policy),
                "labels": labels,
                "matrix": matrix,
                "source_sha256": _sha256(path),
            }
            record["evidence_id"] = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()[:24]
            records.append(record)
    records.sort(key=lambda item: (item["surface"], item["run_id"], item["policy_variant"]))
    validate_evidence(records)
    return records


def validate_evidence(records: Sequence[Mapping[str, Any]]) -> None:
    if len(records) != 136:
        raise ValueError(f"Expected 136 Protocol A matrices, found {len(records)}")
    run_policies: dict[str, set[str]] = defaultdict(set)
    ids: set[str] = set()
    for record in records:
        if any(key.lower().endswith("path") for key in record):
            raise ValueError("Public evidence must not contain filesystem path fields")
        evidence_id = str(record["evidence_id"])
        if evidence_id in ids:
            raise ValueError(f"Duplicate evidence ID: {evidence_id}")
        ids.add(evidence_id)
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["source_sha256"])):
            raise ValueError(f"Invalid source hash: {record['source_sha256']}")
        run_policies[str(record["run_id"])].add(str(record["policy_variant"]))
        corrected_metrics(record)
    if len(run_policies) != 34:
        raise ValueError(f"Expected 34 Protocol A runs, found {len(run_policies)}")
    bad = {run_id: sorted(values) for run_id, values in run_policies.items() if values != set(POLICIES)}
    if bad:
        raise ValueError(f"Runs do not contain exactly four policies: {bad}")


def write_evidence(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    validate_evidence(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_canonical_json(record) + "\n" for record in records)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_evidence(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_evidence(records)
    return records


def metric_index(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(record["run_id"]), str(record["policy_variant"])): {**record, **corrected_metrics(record)}
        for record in records
    }
