"""Provisional, typed research-claim definitions for the Reckoning campaign.

Claim templates are NOT approved experiment specifications. Full-run readiness
is intentionally blocked until exact metric and evidence contracts are frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ClaimType(str, Enum):
    ORDINAL = "ordinal"
    CATEGORICAL = "categorical"
    MAGNITUDE = "magnitude"


@dataclass(frozen=True)
class ClaimTemplate:
    claim_id: str
    statement: str
    types: tuple[ClaimType, ...]
    candidate_changed_factors: tuple[str, ...]
    metric_ids: tuple[str, ...]
    denominator_id: str | None
    fixed_anchors: tuple[str, ...]
    required_eligibility_gates: tuple[str, ...]
    practical_margin: float | None
    uncertainty_spec: str | None
    multiplicity_family: str | None
    evidence_paths: tuple[str, ...]
    definition_status: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClaimTemplate":
        required = {
            "claim_id", "statement", "types", "candidate_changed_factors",
            "metric_ids", "denominator_id", "fixed_anchors",
            "required_eligibility_gates", "practical_margin", "uncertainty_spec",
            "multiplicity_family", "evidence_paths", "definition_status",
        }
        if set(value) != required:
            raise ValueError(f"Claim fields mismatch: missing={sorted(required - set(value))}, "
                             f"extra={sorted(set(value) - required)}")
        for key in ("claim_id", "statement"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"{key} must be nonempty text")
        for key in ("types", "candidate_changed_factors", "metric_ids", "fixed_anchors",
                    "required_eligibility_gates", "evidence_paths"):
            if not isinstance(value[key], list) or any(not isinstance(v, str) or not v for v in value[key]):
                raise ValueError(f"{key} must be a list of nonempty strings")
            if len(set(value[key])) != len(value[key]):
                raise ValueError(f"{key} must not have duplicates")
        if not value["types"] or not value["candidate_changed_factors"]:
            raise ValueError("Every claim needs type(s) and candidate change factor(s)")
        for key in ("denominator_id", "uncertainty_spec", "multiplicity_family"):
            if value[key] is not None and (not isinstance(value[key], str) or not value[key]):
                raise ValueError(f"{key} must be null or nonempty text")
        margin = value["practical_margin"]
        if margin is not None:
            import math
            if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
                raise ValueError("practical_margin must be null or finite and nonnegative")
        if value["definition_status"] not in ("provisional", "frozen"):
            raise ValueError("Unknown claim definition_status")
        return cls(
            claim_id=value["claim_id"], statement=value["statement"],
            types=tuple(ClaimType(t) for t in value["types"]),
            candidate_changed_factors=tuple(value["candidate_changed_factors"]),
            metric_ids=tuple(value["metric_ids"]),
            denominator_id=value["denominator_id"],
            fixed_anchors=tuple(value["fixed_anchors"]),
            required_eligibility_gates=tuple(value["required_eligibility_gates"]),
            practical_margin=margin, uncertainty_spec=value["uncertainty_spec"],
            multiplicity_family=value["multiplicity_family"],
            evidence_paths=tuple(value["evidence_paths"]),
            definition_status=value["definition_status"],
        )


def load_claim_registry(path: str | Path) -> tuple[ClaimTemplate, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"schema_version", "registry_status", "claims"}:
        raise ValueError("Unexpected claim registry structure")
    if data["schema_version"] != 1 or data["registry_status"] not in ("provisional", "frozen"):
        raise ValueError("Unsupported registry schema or status")
    if not isinstance(data["claims"], list) or not data["claims"]:
        raise ValueError("Claim registry must contain claims")
    claims = tuple(ClaimTemplate.from_mapping(item) for item in data["claims"])
    identifiers = [claim.claim_id for claim in claims]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate claim IDs")
    if data["registry_status"] == "frozen":
        require_frozen_claims(claims)
    return claims


def require_frozen_claims(claims: tuple[ClaimTemplate, ...]) -> None:
    """Fail closed if a provisional template is mistaken for a campaign-ready claim."""
    for claim in claims:
        if (claim.definition_status != "frozen" or not claim.metric_ids
                or not claim.denominator_id or not claim.fixed_anchors
                or not claim.required_eligibility_gates or not claim.uncertainty_spec
                or not claim.multiplicity_family or not claim.evidence_paths
                or len(claim.types) != 1 or len(claim.candidate_changed_factors) != 1):
            raise ValueError(f"{claim.claim_id}: claim is not atomically specified and frozen")
