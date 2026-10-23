"""Observed claim relations, deliberately separate from statistical inference.

Sign changes are descriptive. They are never labeled statistically established
without a separately specified, validated inferential analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ids_eval_framework.src.claim_comparability import ComparabilityResult, ComparabilityStatus


class OutcomeStatus(str, Enum):
    OBSERVED = "observed"
    NOT_COMPARABLE = "not_comparable"
    NOT_ASSESSED = "not_assessed"
    UNDEFINED = "undefined"


@dataclass(frozen=True)
class RelationResult:
    status: OutcomeStatus
    relation: str | None
    baseline: float | str | None
    changed: float | str | None
    difference: float | None
    inferential_status: str = "not_assessed"
    reason: str | None = None


@dataclass(frozen=True)
class OrdinalScores:
    first: float | None
    second: float | None


@dataclass(frozen=True)
class CategoricalOutcome:
    value: str | None
    defined: bool = True

    def __post_init__(self) -> None:
        if self.defined and (self.value is None or self.value == ""):
            raise ValueError("Defined outcome requires a category")
        if not self.defined and self.value is not None:
            raise ValueError("Undefined outcome must not invent a category")


def _blocked(gate: ComparabilityResult) -> RelationResult | None:
    if gate.status == ComparabilityStatus.COMPARABLE:
        return None
    status = OutcomeStatus.NOT_ASSESSED if gate.status == ComparabilityStatus.NOT_ASSESSED else OutcomeStatus.NOT_COMPARABLE
    return RelationResult(status, None, None, None, None, reason="; ".join(gate.reasons) or gate.status.value)


def _number(value: float | None) -> bool:
    return value is not None and not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def evaluate_ordinal(
    baseline: OrdinalScores, changed: OrdinalScores, gate: ComparabilityResult,
    *, practical_margin: float | None = None,
) -> RelationResult:
    """Report observed ordering preservation/reversal, not statistical certainty.

    An optional margin is a caller-supplied *prospectively justified* value;
    this module never selects one or consults test outcomes to choose it.
    """
    blocked = _blocked(gate)
    if blocked:
        return blocked
    if practical_margin is not None and (not _number(practical_margin) or practical_margin < 0):
        raise ValueError("Practical margin must be finite and nonnegative")
    if not all(_number(x) for x in (baseline.first, baseline.second, changed.first, changed.second)):
        return RelationResult(OutcomeStatus.NOT_ASSESSED, None, None, None, None, reason="Missing or nonfinite ordinal scores")
    d0 = float(baseline.first) - float(baseline.second)
    d1 = float(changed.first) - float(changed.second)

    def sign(d: float) -> int:
        if practical_margin is not None and abs(d) <= practical_margin:
            return 0
        return (d > 0) - (d < 0)

    a, b = sign(d0), sign(d1)
    if a == b:
        relation = "preserved" if a != 0 else "tie_or_equivalence_preserved"
    elif a == 0 or b == 0:
        relation = "tie_or_equivalence_transition"
    else:
        relation = "observed_reversal"
    return RelationResult(OutcomeStatus.OBSERVED, relation, d0, d1, d1 - d0)


def evaluate_magnitude(
    baseline: float | None, changed: float | None, gate: ComparabilityResult,
) -> RelationResult:
    blocked = _blocked(gate)
    if blocked:
        return blocked
    if not _number(baseline) or not _number(changed):
        return RelationResult(OutcomeStatus.NOT_ASSESSED, None, None, None, None, reason="Missing or nonfinite metric")
    delta = float(changed) - float(baseline)
    return RelationResult(OutcomeStatus.OBSERVED, "magnitude_delta", float(baseline), float(changed), delta)


def evaluate_categorical(
    baseline: CategoricalOutcome | None, changed: CategoricalOutcome | None,
    gate: ComparabilityResult,
) -> RelationResult:
    blocked = _blocked(gate)
    if blocked:
        return blocked
    if baseline is None or changed is None:
        return RelationResult(OutcomeStatus.NOT_ASSESSED, None, None, None, None, reason="Categorical observation missing")
    if not baseline.defined or not changed.defined:
        if not baseline.defined and not changed.defined:
            transition = "undefined_in_both"
        elif not baseline.defined:
            transition = "undefined_to_defined"
        else:
            transition = "defined_to_undefined"
        return RelationResult(OutcomeStatus.UNDEFINED, transition, baseline.value, changed.value, None,
                              reason="At least one outcome has no defined category (e.g., no residual predictions)")
    relation = "preserved" if baseline.value == changed.value else "changed"
    return RelationResult(OutcomeStatus.OBSERVED, relation, baseline.value, changed.value, None)
