"""C06 historical validation-feasibility derivation; no training or inference.

Expand the immutable COMNET budget summary into holdout-condition categories.
This module does NOT certify claim eligibility or statistical stability.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SOURCE_RELATIVE = "release/v2026.10/raw/visible_budget_feasibility_coverage.csv"
SOURCE_GIT_BLOB_SHA = "3a6c388a544f8d028b5c1223cae58d0b5ef1dc51"
BUDGETS = (Decimal("0.01"), Decimal("0.03"), Decimal("0.05"), Decimal("0.10"))
METHODS = ("entropy", "family_conditional", "margin", "max_confidence")
HOLDOUTS = {
    "CICIDS2017": ("Botnet", "BruteForce", "DDoS", "DoS", "Scan/Recon", "Web/App"),
    "CICIoT2023": ("Botnet", "BruteForce", "DDoS", "DoS", "Other", "Scan/Recon"),
}
SOURCE_COLUMNS = (
    "dataset", "budget", "method", "n_holdouts_total", "n_holdouts_feasible",
    "feasibility_fraction", "feasible_holdouts",
)
OBSERVATION_COLUMNS = (
    "condition_id", "claim_id", "dataset", "protocol", "selection_information",
    "holdout_family", "method", "validation_rejection_budget", "category",
    "metric_id", "metric_denominator", "source_aggregate_key", "evidence_path",
    "evidence_git_blob_sha", "eligibility_status", "formal_relation_status",
    "inference_status", "historical_restriction",
)
TRANSITION_COLUMNS = (
    "transition_id", "claim_id", "dataset", "protocol", "selection_information",
    "holdout_family", "method", "budget_before", "budget_after", "category_before",
    "category_after", "descriptive_relation", "descriptive_transition", "metric_id",
    "metric_denominator", "condition_id_before", "condition_id_after", "evidence_path",
    "evidence_git_blob_sha", "eligibility_status", "formal_relation_status",
    "inference_status", "historical_restriction",
)
SUMMARY_COLUMNS = (
    "dataset", "method", "holdouts_total", "feasible_1pct", "feasible_3pct",
    "feasible_5pct", "feasible_10pct", "gained_1to3", "gained_3to5",
    "gained_5to10", "descriptive_transition_count", "eligibility_status",
)


@dataclass(frozen=True)
class C06Result:
    observations: tuple[dict[str, str], ...]
    transitions: tuple[dict[str, str], ...]
    summary: tuple[dict[str, str], ...]
    evidence_manifest: dict[str, Any]


def _git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _canonical_bytes(raw: bytes) -> bytes:
    # Accommodate Git autocrlf checkout conversion; no content changes allowed.
    return raw.replace(b"\r\n", b"\n")


def _budget(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid budget: {text!r}") from exc
    if not value.is_finite() or value not in BUDGETS:
        raise ValueError(f"Unexpected budget: {text!r}")
    return value


def _integer(text: str, name: str) -> int:
    try:
        result = int(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}: {text!r}") from exc
    if str(result) != str(text).strip() or result < 0:
        raise ValueError(f"Invalid {name}: {text!r}")
    return result


def _context(dataset: str, method: str, budget: Decimal) -> str:
    return f"{dataset}|{method}|{budget:.2f}"


def _condition_id(dataset: str, method: str, holdout: str, budget: Decimal) -> str:
    # JSON-based explicit encoding prevents ambiguity from / in Scan/Recon and Web/App.
    key = json.dumps([dataset, method, holdout, f"{budget:.2f}"], ensure_ascii=True, separators=(",", ":"))
    return "C06:" + hashlib.sha256(key.encode("ascii")).hexdigest()[:20]


def derive_c06(raw: bytes, *, verify_frozen_blob: bool = True) -> C06Result:
    """Validate the full released coverage table and derive descriptive cells.

    The small summary is *not* a record of matched validation predictions, and
    unchanged budgets do not supply information about threshold trajectories.
    """
    blob = _canonical_bytes(raw)
    calculated_sha = _git_blob_sha(blob)
    if verify_frozen_blob and calculated_sha != SOURCE_GIT_BLOB_SHA:
        raise ValueError("Historical source Git blob mismatch; refusing to derive from modified evidence")
    try:
        text = blob.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Source is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
        raise ValueError("Historical feasibility CSV columns differ from frozen schema")
    parsed: dict[tuple[str, str, Decimal], set[str]] = {}
    for n, row in enumerate(reader, start=2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Malformed CSV row {n}")
        dataset, method = row["dataset"], row["method"]
        if dataset not in HOLDOUTS or method not in METHODS:
            raise ValueError(f"Unexpected dataset/method at row {n}")
        budget = _budget(row["budget"])
        key = (dataset, method, budget)
        if key in parsed:
            raise ValueError(f"Duplicate historical coverage row: {key}")
        total = _integer(row["n_holdouts_total"], "n_holdouts_total")
        feasible = _integer(row["n_holdouts_feasible"], "n_holdouts_feasible")
        if total != len(HOLDOUTS[dataset]) or feasible > total:
            raise ValueError(f"Unsupported holdout total/count at row {n}")
        names = row["feasible_holdouts"].split("|") if row["feasible_holdouts"] else []
        if len(names) != len(set(names)) or len(names) != feasible or not set(names).issubset(HOLDOUTS[dataset]):
            raise ValueError(f"Inconsistent feasible holdout membership at row {n}")
        try:
            fraction = Decimal(row["feasibility_fraction"])
        except InvalidOperation as exc:
            raise ValueError(f"Invalid feasibility fraction at row {n}") from exc
        if not fraction.is_finite() or abs(fraction - Decimal(feasible) / total) > Decimal("0.000000000001"):
            raise ValueError(f"Inconsistent feasibility fraction at row {n}")
        parsed[key] = set(names)

    required = {(dataset, method, budget) for dataset in HOLDOUTS for method in METHODS for budget in BUDGETS}
    if set(parsed) != required:
        missing = required - set(parsed)
        extra = set(parsed) - required
        raise ValueError(f"Incomplete coverage grid: missing={len(missing)}, extra={len(extra)}")

    observations: list[dict[str, str]] = []
    transitions: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    for dataset, holdouts in HOLDOUTS.items():
        for method in METHODS:
            previous: set[str] = set()
            counts: list[int] = []
            gains: list[int] = []
            for bi, budget in enumerate(BUDGETS):
                members = parsed[dataset, method, budget]
                if bi and not previous.issubset(members):
                    raise ValueError(f"Non-monotone budget feasibility in {_context(dataset, method, budget)}; requires audit")
                counts.append(len(members))
                if bi:
                    gains.append(len(members - previous))
                for holdout in holdouts:
                    category = "feasible" if holdout in members else "infeasible"
                    restriction = "recovered_partition_overlap_and_shared_source_days" if dataset == "CICIDS2017" else "unassessed_source_and_split_validity"
                    observations.append({
                        "condition_id": _condition_id(dataset, method, holdout, budget),
                        "claim_id": "C06", "dataset": dataset, "protocol": "B_LOAO",
                        "selection_information": "validation_visible",
                        "holdout_family": holdout, "method": method,
                        "validation_rejection_budget": f"{budget:.2f}", "category": category,
                        "metric_id": "validation_feasibility_category",
                        "metric_denominator": "one_eligible_holdout_candidate_at_one_method_budget",
                        "source_aggregate_key": _context(dataset, method, budget),
                        "evidence_path": SOURCE_RELATIVE, "evidence_git_blob_sha": SOURCE_GIT_BLOB_SHA,
                        "eligibility_status": "not_assessed", "formal_relation_status": "not_assessed",
                        "inference_status": "not_assessed", "historical_restriction": restriction,
                    })
                    if bi:
                        prev_budget = BUDGETS[bi - 1]
                        category_before = "feasible" if holdout in previous else "infeasible"
                        descriptive = "preserved" if category_before == category else "changed"
                        change = "became_feasible" if category_before != category else f"remained_{category}"
                        transitions.append({
                            "transition_id": _condition_id(dataset, method, holdout, prev_budget) + "->" + f"{budget:.2f}",
                            "claim_id": "C06", "dataset": dataset, "protocol": "B_LOAO",
                            "selection_information": "validation_visible", "holdout_family": holdout,
                            "method": method, "budget_before": f"{prev_budget:.2f}",
                            "budget_after": f"{budget:.2f}", "category_before": category_before,
                            "category_after": category, "descriptive_relation": descriptive,
                            "descriptive_transition": change,
                            "metric_id": "validation_feasibility_category",
                            "metric_denominator": "one_eligible_holdout_candidate_at_one_method_budget",
                            "condition_id_before": _condition_id(dataset, method, holdout, prev_budget),
                            "condition_id_after": _condition_id(dataset, method, holdout, budget),
                            "evidence_path": SOURCE_RELATIVE, "evidence_git_blob_sha": SOURCE_GIT_BLOB_SHA,
                            "eligibility_status": "not_assessed", "formal_relation_status": "not_assessed",
                            "inference_status": "not_assessed", "historical_restriction": restriction,
                        })
                previous = members
            summaries.append({
                "dataset": dataset, "method": method, "holdouts_total": str(len(holdouts)),
                "feasible_1pct": str(counts[0]), "feasible_3pct": str(counts[1]),
                "feasible_5pct": str(counts[2]), "feasible_10pct": str(counts[3]),
                "gained_1to3": str(gains[0]), "gained_3to5": str(gains[1]),
                "gained_5to10": str(gains[2]), "descriptive_transition_count": str(3 * len(holdouts)),
                "eligibility_status": "not_assessed",
            })
    if len(observations) != 192 or len(transitions) != 144 or len(summaries) != 8:
        raise AssertionError("Unexpected historic C06 cardinality")
    evidence_manifest = {
        "schema_version": 1, "claim_id": "C06", "derivation_status": "historical_descriptive_not_admissibility_certification",
        "source_path": SOURCE_RELATIVE, "source_git_blob_sha": SOURCE_GIT_BLOB_SHA,
        "observed_git_blob_sha": calculated_sha, "source_sha256_normalized_lf": hashlib.sha256(blob).hexdigest(),
        "historical_aggregate_rows": len(parsed), "derived_condition_rows": len(observations),
        "derived_adjacent_budget_transitions": len(transitions), "summary_rows": len(summaries),
        "candidate_budgets": [f"{v:.2f}" for v in BUDGETS],
        "candidate_methods": list(METHODS), "holdout_universe": {k: list(v) for k, v in HOLDOUTS.items()},
        "changed_factor": "validation_rejection_budget", "fixed_anchors": [
            "dataset", "protocol", "selection_information", "holdout_family", "method",
            "prepared_partition", "model_profile", "candidate_grid", "validation_population",
        ],
        "unverified_anchors": ["prepared_partition", "model_profile", "candidate_grid", "validation_population"],
        "eligibility_status": "not_assessed", "formal_relation_status": "not_assessed",
        "uncertainty": "none: exact categorization of released deterministic summary; no seed-level uncertainty or inference",
        "limitations": [
            "Historical validation-visible feasibility only; never a realized test rejection guarantee.",
            "Six eligible holdouts per dataset are historical count-support candidates, not fresh source/proxy eligibility decisions.",
            "Source is an aggregate; paired threshold trajectories and observation identity are not documented.",
            "CICIDS2017 recovered partition has documented shared source-days and cross-split feature-hash intersections.",
            "A descriptive state change is not a formally admitted claim until audited evidence gates and anchors pass.",
        ],
    }
    return C06Result(tuple(observations), tuple(transitions), tuple(summaries), evidence_manifest)


def _write_csv(path: Path, columns: tuple[str, ...], rows: tuple[dict[str, str], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_c06_outputs(result: C06Result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "c06_validation_feasibility_observations.csv", OBSERVATION_COLUMNS, result.observations)
    _write_csv(out_dir / "c06_validation_feasibility_adjacent_transitions.csv", TRANSITION_COLUMNS, result.transitions)
    _write_csv(out_dir / "c06_validation_feasibility_summary.csv", SUMMARY_COLUMNS, result.summary)
    (out_dir / "c06_evidence_manifest.json").write_text(
        json.dumps(result.evidence_manifest, indent=2) + "\n", encoding="utf-8"
    )
