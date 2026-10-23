"""Compare evaluation conditions without assuming a shared test population.

Differences are evaluated over explicit condition-factor maps; metadata such as
record hashes is kept separate from scientifically varied experimental factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from ids_eval_framework.src.claim_eligibility import EligibilityResult, EligibilityStatus


class ComparabilityStatus(str, Enum):
    COMPARABLE = "comparable"
    RESTRICTED = "restricted"
    INCOMPARABLE = "incomparable"
    NOT_ASSESSED = "not_assessed"


class PairingPolicy(str, Enum):
    NOT_REQUESTED = "not_requested"
    REQUIRE_IDENTICAL_TEST = "require_identical_test"


class PopulationPolicy(str, Enum):
    """How a claim-matched comparison treats test population provenance.

    Test population equality is a COMPARABILITY condition, not proof of
    correctly aligned observation-level pairs for statistical inference.
    """

    REQUIRE_IDENTICAL = "require_identical"
    ALLOW_DIFFERENT = "allow_different"


@dataclass(frozen=True)
class EvaluationCondition:
    factors: Mapping[str, str | int | float | bool | None]
    metric_id: str
    denominator_id: str
    label_set_id: str
    test_population_id: str | None = None
    test_identity_hash: str | None = None


@dataclass(frozen=True)
class ComparabilityResult:
    status: ComparabilityStatus
    reasons: tuple[str, ...]
    changed_factors: tuple[str, ...]
    matched_test_identity: bool
    pairing_metadata_match: bool


def compare_conditions(
    left: EvaluationCondition,
    right: EvaluationCondition,
    left_eligibility: EligibilityResult,
    right_eligibility: EligibilityResult,
    *,
    changed_factor: str,
    fixed_anchors: Sequence[str],
    pairing_policy: PairingPolicy = PairingPolicy.NOT_REQUESTED,
    test_population_policy: PopulationPolicy = PopulationPolicy.REQUIRE_IDENTICAL,
) -> ComparabilityResult:
    """Permit one factor change, with an explicit test-population rule.

    Population shifts are not silently ignored. For partition, holdout, or
    cross-environment studies, the caller must explicitly opt into comparing
    different test populations. This permits descriptive sensitivity only;
    inferential pairing needs subsequent verified observation-level alignment.
    """
    if not isinstance(left_eligibility, EligibilityResult) or not isinstance(right_eligibility, EligibilityResult):
        raise TypeError("Explicit eligibility results required for both conditions")
    if not isinstance(pairing_policy, PairingPolicy):
        raise TypeError("Explicit PairingPolicy required")
    if not isinstance(test_population_policy, PopulationPolicy):
        raise TypeError("Explicit PopulationPolicy required")
    if not isinstance(changed_factor, str) or not changed_factor or changed_factor in fixed_anchors:
        raise ValueError("Changed factor must be nonempty and distinct from fixed anchors")
    if not fixed_anchors:
        raise ValueError("At least one explicitly fixed comparison anchor is required")
    if len(set(fixed_anchors)) != len(fixed_anchors):
        raise ValueError("Duplicate fixed anchors")
    factors_l, factors_r = left.factors, right.factors
    keys = set(factors_l) | set(factors_r)
    different = tuple(sorted(k for k in keys if k not in factors_l or k not in factors_r or factors_l[k] != factors_r[k]))
    reasons: list[str] = []
    not_assessed = False
    incomparable = False
    restricted = False

    for side, result in (("left", left_eligibility), ("right", right_eligibility)):
        if result.status == EligibilityStatus.INADMISSIBLE:
            incomparable = True
            reasons.append(f"{side}: inadmissible evidence")
        elif result.status == EligibilityStatus.NOT_ASSESSED:
            not_assessed = True
            reasons.append(f"{side}: eligibility not assessed")
        elif result.status == EligibilityStatus.RESTRICTED:
            restricted = True
            reasons.append(f"{side}: restricted evidence")

    for name in ("metric_id", "denominator_id", "label_set_id"):
        a, b = getattr(left, name), getattr(right, name)
        if not a or not b:
            not_assessed = True
            reasons.append(f"{name}: target semantics missing")
        elif a != b:
            incomparable = True
            reasons.append(f"{name}: target semantics differ")

    for anchor in fixed_anchors:
        if anchor not in factors_l or anchor not in factors_r or factors_l[anchor] is None or factors_r[anchor] is None:
            not_assessed = True
            reasons.append(f"{anchor}: fixed anchor missing")
        elif factors_l[anchor] != factors_r[anchor]:
            incomparable = True
            reasons.append(f"{anchor}: fixed anchor differs")

    if changed_factor not in factors_l or changed_factor not in factors_r:
        not_assessed = True
        reasons.append("Changed factor unrecorded")
    elif changed_factor not in different:
        incomparable = True
        reasons.append("Declared factor did not change")
    other = [k for k in different if k != changed_factor]
    if other:
        incomparable = True
        reasons.append("Undeclared changes: " + ", ".join(other))

    complete_test_identity = all((
        left.test_population_id, right.test_population_id,
        left.test_identity_hash, right.test_identity_hash,
    ))
    paired = bool(
        complete_test_identity
        and left.test_population_id == right.test_population_id
        and left.test_identity_hash == right.test_identity_hash
    )
    if not complete_test_identity:
        not_assessed = True
        reasons.append("Test population provenance incomplete")
    elif test_population_policy == PopulationPolicy.REQUIRE_IDENTICAL and not paired:
        incomparable = True
        reasons.append("Test population/identity changed without an authorized population shift")
    if pairing_policy == PairingPolicy.REQUIRE_IDENTICAL_TEST and not paired:
        if complete_test_identity:
            incomparable = True
            reasons.append("Test population or identity differs; pairing metadata mismatch")
        else:
            not_assessed = True
            reasons.append("Test identity unavailable; pairing unverified")

    status = (ComparabilityStatus.INCOMPARABLE if incomparable
              else ComparabilityStatus.NOT_ASSESSED if not_assessed
              else ComparabilityStatus.RESTRICTED if restricted
              else ComparabilityStatus.COMPARABLE)
    return ComparabilityResult(
        status=status, reasons=tuple(reasons), changed_factors=different,
        matched_test_identity=paired, pairing_metadata_match=bool(
            paired and pairing_policy == PairingPolicy.REQUIRE_IDENTICAL_TEST
            and status == ComparabilityStatus.COMPARABLE
        ),
    )
