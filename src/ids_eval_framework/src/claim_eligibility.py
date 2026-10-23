"""Aggregate explicitly audited, claim-specific evidence gates.

This module NEVER infers that row support proves independent data sources.
A scientific auditor must assess each relevant gate against actual evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class GateVerdict(str, Enum):
    PASS = "pass"
    RESTRICTED = "restricted"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


class EligibilityStatus(str, Enum):
    ADMISSIBLE = "admissible"
    RESTRICTED = "restricted"
    INADMISSIBLE = "inadmissible"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class GateAssessment:
    verdict: GateVerdict
    reason: str
    evidence_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, GateVerdict):
            raise TypeError("Gate verdict must use GateVerdict enum")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Each gate requires an explicit reason")
        if self.verdict != GateVerdict.NOT_ASSESSED and not self.evidence_path:
            raise ValueError("Assessed gates require an evidence path or URI")


@dataclass(frozen=True)
class EligibilityResult:
    status: EligibilityStatus
    reasons: tuple[str, ...]
    missing_gates: tuple[str, ...]


def assess_eligibility(
    required_gates: Sequence[str], observed: Mapping[str, GateAssessment]
) -> EligibilityResult:
    """Evaluate only gates expressly required for this claim and protocol.

    A known failed gate makes the claim inadmissible even if another gate was
    not assessed. Otherwise, unknown gates cannot be treated as passes.
    """
    if not required_gates or any(not isinstance(g, str) or not g for g in required_gates):
        raise ValueError("Nonempty, explicitly named required_gates are mandatory")
    if len(set(required_gates)) != len(required_gates):
        raise ValueError("Duplicate required gates")
    if any(not isinstance(v, GateAssessment) for v in observed.values()):
        raise TypeError("Observed gates must contain GateAssessment values")
    missing = tuple(g for g in required_gates if g not in observed or observed[g].verdict == GateVerdict.NOT_ASSESSED)
    reasons = tuple(f"{g}: {observed[g].reason}" for g in required_gates if g in observed
                    and observed[g].verdict != GateVerdict.PASS)
    if any(observed[g].verdict == GateVerdict.FAIL for g in required_gates if g in observed):
        status = EligibilityStatus.INADMISSIBLE
    elif missing:
        status = EligibilityStatus.NOT_ASSESSED
    elif any(observed[g].verdict == GateVerdict.RESTRICTED for g in required_gates):
        status = EligibilityStatus.RESTRICTED
    else:
        status = EligibilityStatus.ADMISSIBLE
    return EligibilityResult(status=status, reasons=reasons, missing_gates=missing)
