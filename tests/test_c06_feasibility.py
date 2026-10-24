"""Data-free and pinned-historical tests for C06 categorical reconstruction."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from ids_eval_framework.src.c06_feasibility import (
    BUDGETS, HOLDOUTS, METHODS, SOURCE_COLUMNS, SOURCE_GIT_BLOB_SHA,
    SOURCE_RELATIVE, derive_c06, write_c06_outputs,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic(*, override: dict[tuple[str, str, str], dict[str, str]] | None = None) -> bytes:
    override = override or {}
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=SOURCE_COLUMNS, lineterminator="\n")
    w.writeheader()
    for dataset, holdouts in HOLDOUTS.items():
        for budget in BUDGETS:
            for method in METHODS:
                row = dict(dataset=dataset, budget=str(budget), method=method,
                           n_holdouts_total=str(len(holdouts)), n_holdouts_feasible=str(len(holdouts)),
                           feasibility_fraction="1.0", feasible_holdouts="|".join(holdouts))
                row.update(override.get((dataset, method, str(budget)), {}))
                w.writerow(row)
    return s.getvalue().encode()


def _edit(raw: bytes, line_idx: int, column: str, value: str) -> bytes:
    r = csv.DictReader(io.StringIO(raw.decode()))
    rows = list(r)
    rows[line_idx][column] = value
    s = io.StringIO(newline="")
    w = csv.DictWriter(s, fieldnames=SOURCE_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode()


def test_synthetic_cardinality_and_provisional_status() -> None:
    result = derive_c06(_synthetic(), verify_frozen_blob=False)
    assert len(result.observations) == 192
    assert len(result.transitions) == 144
    assert len(result.summary) == 8
    assert all(t["descriptive_transition"] == "remained_feasible" for t in result.transitions)
    assert all(t["eligibility_status"] == "not_assessed" for t in result.transitions)
    assert all(t["formal_relation_status"] == "not_assessed" for t in result.transitions)
    assert len({x["condition_id"] for x in result.observations}) == 192


def test_detects_single_membership_transition_without_statistics() -> None:
    # Make Botnet infeasible only under the smallest budget for one method.
    members = "|".join(HOLDOUTS["CICIoT2023"][1:])
    b = _synthetic(override={("CICIoT2023", "entropy", "0.01"):
                             {"n_holdouts_feasible": "5", "feasibility_fraction": "0.8333333333333334",
                              "feasible_holdouts": members}})
    result = derive_c06(b, verify_frozen_blob=False)
    selected = [t for t in result.transitions if t["dataset"] == "CICIoT2023"
                and t["method"] == "entropy" and t["holdout_family"] == "Botnet"
                and t["budget_before"] == "0.01"]
    assert len(selected) == 1
    assert selected[0]["descriptive_transition"] == "became_feasible"
    assert selected[0]["formal_relation_status"] == "not_assessed"


@pytest.mark.parametrize("field,value,message", [
    ("n_holdouts_feasible", "7", "total/count"),
    ("feasibility_fraction", "0.1", "fraction"),
    ("feasible_holdouts", "Botnet|Botnet", "membership"),
    ("feasible_holdouts", "Unknown|BruteForce|DDoS|DoS|Scan/Recon|Web/App", "membership"),
    ("budget", "NaN", "budget"),
])
def test_rejects_corrupt_aggregate(field: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        derive_c06(_edit(_synthetic(), 0, field, value), verify_frozen_blob=False)


def test_rejects_missing_and_duplicate_grid_rows() -> None:
    raw = _synthetic()
    data = raw.splitlines(keepends=True)
    with pytest.raises(ValueError, match="Incomplete coverage"):
        derive_c06(b"".join(data[:-1]), verify_frozen_blob=False)
    with pytest.raises(ValueError, match="Duplicate historical"):
        derive_c06(b"".join(data + [data[-1]]), verify_frozen_blob=False)


def test_nonmonotone_fails_closed() -> None:
    # Feasible at 1% but absent at 3% contradicts nested fixed-grid feasibility.
    names = "|".join(HOLDOUTS["CICIDS2017"][1:])
    raw = _synthetic(override={("CICIDS2017", "entropy", "0.03"):
                               {"n_holdouts_feasible": "5", "feasibility_fraction": "0.8333333333333334",
                                "feasible_holdouts": names}})
    with pytest.raises(ValueError, match="Non-monotone"):
        derive_c06(raw, verify_frozen_blob=False)


def test_frozen_blob_mismatch_is_error() -> None:
    with pytest.raises(ValueError, match="Git blob mismatch"):
        derive_c06(_synthetic(), verify_frozen_blob=True)


def test_pinned_source_cardinality_and_transitions(tmp_path: Path) -> None:
    source = ROOT / SOURCE_RELATIVE
    if not source.is_file():
        pytest.skip("Pinned COMNET CSV is not included in this isolated test checkout")
    result = derive_c06(source.read_bytes())
    summary = {(r["dataset"], r["method"]): r for r in result.summary}
    assert [summary["CICIDS2017", "family_conditional"][f"feasible_{x}pct"] for x in (1, 3, 5, 10)] == ["0", "3", "3", "6"]
    assert [summary["CICIoT2023", "family_conditional"][f"feasible_{x}pct"] for x in (1, 3, 5, 10)] == ["1", "4", "4", "4"]
    assert all(summary[d, "max_confidence"][f"feasible_{x}pct"] == "6"
               for d in HOLDOUTS for x in (1, 3, 5, 10))
    assert result.evidence_manifest["observed_git_blob_sha"] == SOURCE_GIT_BLOB_SHA
    assert sum(x["descriptive_transition"] == "became_feasible" for x in result.transitions) == 21
    write_c06_outputs(result, tmp_path)
    assert len(list(csv.DictReader((tmp_path / "c06_validation_feasibility_observations.csv").open(encoding="utf8")))) == 192
    assert len(list(csv.DictReader((tmp_path / "c06_validation_feasibility_adjacent_transitions.csv").open(encoding="utf8")))) == 144
    manifest = json.loads((tmp_path / "c06_evidence_manifest.json").read_text(encoding="utf8"))
    assert manifest["eligibility_status"] == "not_assessed"
