"""Data-free contract tests for the provisional Reckoning claim framework."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ids_eval_framework.src.claim_registry import ClaimType, load_claim_registry, require_frozen_claims
from ids_eval_framework.src.claim_eligibility import (
    EligibilityResult, EligibilityStatus, GateAssessment, GateVerdict, assess_eligibility,
)
from ids_eval_framework.src.claim_comparability import (
    ComparabilityStatus, EvaluationCondition, PairingPolicy, PopulationPolicy, compare_conditions,
)
from ids_eval_framework.src.claim_relations import (
    CategoricalOutcome, OrdinalScores, OutcomeStatus, evaluate_categorical,
    evaluate_magnitude, evaluate_ordinal,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_eight_provisional_families() -> None:
    claims = load_claim_registry(ROOT / "config" / "claim_registry.json")
    assert [c.claim_id for c in claims] == [f"C{i:02}" for i in range(1, 9)]
    assert {kind for c in claims for kind in c.types} == set(ClaimType)
    with pytest.raises(ValueError, match="not atomically specified and frozen"):
        require_frozen_claims(claims)


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    original = json.loads((ROOT / "config" / "claim_registry.json").read_text(encoding="utf-8"))
    original["claims"][1]["claim_id"] = original["claims"][0]["claim_id"]
    target = tmp_path / "registry.json"
    target.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate claim IDs"):
        load_claim_registry(target)


def test_no_evidence_no_admissible_claim() -> None:
    result = assess_eligibility(["source_provenance", "label_validity"], {})
    assert result.status == EligibilityStatus.NOT_ASSESSED
    assert result.missing_gates == ("source_provenance", "label_validity")


def test_gate_requires_traceable_assessment() -> None:
    with pytest.raises(ValueError, match="evidence"):
        GateAssessment(GateVerdict.PASS, "looks fine")
    with pytest.raises(ValueError, match="reason"):
        GateAssessment(GateVerdict.NOT_ASSESSED, "")


def test_eligibility_precedence_and_scope() -> None:
    observed = {
        "a": GateAssessment(GateVerdict.PASS, "audited", "proof/a.json"),
        "b": GateAssessment(GateVerdict.RESTRICTED, "limited independence", "proof/b.json"),
        "c": GateAssessment(GateVerdict.FAIL, "overlap", "proof/c.json"),
    }
    assert assess_eligibility(["a"], observed).status == EligibilityStatus.ADMISSIBLE
    assert assess_eligibility(["a", "b"], observed).status == EligibilityStatus.RESTRICTED
    assert assess_eligibility(["a", "c", "absent"], observed).status == EligibilityStatus.INADMISSIBLE
    assert assess_eligibility(["a", "absent"], observed).status == EligibilityStatus.NOT_ASSESSED


def test_reject_unspecified_eligibility_rules() -> None:
    with pytest.raises(ValueError, match="required_gates"):
        assess_eligibility([], {})


ELIGIBLE = EligibilityResult(EligibilityStatus.ADMISSIBLE, (), ())
INELIGIBLE = EligibilityResult(EligibilityStatus.INADMISSIBLE, ("leakage",), ())
RESTRICTED = EligibilityResult(EligibilityStatus.RESTRICTED, ("limited",), ())
UNKNOWN = EligibilityResult(EligibilityStatus.NOT_ASSESSED, (), ("capture",))


def condition(*, policy: str = "broad", family: str = "Botnet", population: str | None = "test-1",
              row_hash: str | None = "sha256-1", denominator: str = "supported-labels",
              method: str = "RF", partition: str = "p0") -> EvaluationCondition:
    return EvaluationCondition(
        factors={"dataset": "CICIoT2023", "family": family, "policy": policy, "method": method,
                 "partition": partition},
        metric_id="system_macro_f1", denominator_id=denominator,
        label_set_id="Protocol-B-supported", test_population_id=population,
        test_identity_hash=row_hash,
    )


def comparable(*, paired: bool = False):
    return compare_conditions(
        condition(), condition(policy="fpr_matched"), ELIGIBLE, ELIGIBLE,
        changed_factor="policy", fixed_anchors=["dataset", "family", "method"],
        pairing_policy=PairingPolicy.REQUIRE_IDENTICAL_TEST if paired else PairingPolicy.NOT_REQUESTED,
    )


def test_policy_change_is_comparable_without_automatic_pairing() -> None:
    result = comparable()
    assert result.status == ComparabilityStatus.COMPARABLE
    assert result.matched_test_identity
    assert not result.pairing_metadata_match


def test_pairing_requires_matched_test_identity() -> None:
    result = comparable(paired=True)
    assert result.status == ComparabilityStatus.COMPARABLE
    assert result.pairing_metadata_match  # Metadata match, NOT inferential authorization
    mismatch = compare_conditions(
        condition(), condition(policy="fpr_matched", row_hash="different"), ELIGIBLE, ELIGIBLE,
        changed_factor="policy", fixed_anchors=["dataset", "family", "method"],
        pairing_policy=PairingPolicy.REQUIRE_IDENTICAL_TEST,
    )
    assert mismatch.status == ComparabilityStatus.INCOMPARABLE
    assert not mismatch.pairing_metadata_match
    missing = compare_conditions(
        condition(), condition(policy="fpr_matched", row_hash=None), ELIGIBLE, ELIGIBLE,
        changed_factor="policy", fixed_anchors=["dataset", "family", "method"],
        pairing_policy=PairingPolicy.REQUIRE_IDENTICAL_TEST,
    )
    assert missing.status == ComparabilityStatus.NOT_ASSESSED


def test_policy_comparison_blocks_uncontrolled_test_population_change() -> None:
    result = compare_conditions(
        condition(), condition(policy="fpr_matched", population="test-2", row_hash="hash-2"),
        ELIGIBLE, ELIGIBLE, changed_factor="policy", fixed_anchors=["dataset", "family", "method"],
    )
    assert result.status == ComparabilityStatus.INCOMPARABLE
    assert "population" in " ".join(result.reasons).lower()


def test_partition_sensitivity_explicitly_allows_different_test_population() -> None:
    left = condition()
    right = condition(partition="p1", population="test-2", row_hash="hash-2")
    result = compare_conditions(
        left, right, ELIGIBLE, ELIGIBLE,
        changed_factor="partition", fixed_anchors=["dataset", "family", "method", "policy"],
        test_population_policy=PopulationPolicy.ALLOW_DIFFERENT,
    )
    assert result.status == ComparabilityStatus.COMPARABLE
    assert not result.matched_test_identity
    assert not result.pairing_metadata_match


def test_population_shift_requires_documented_identity_and_no_pairing() -> None:
    result = compare_conditions(
        condition(), condition(partition="p1", population="test-2", row_hash=None),
        ELIGIBLE, ELIGIBLE,
        changed_factor="partition", fixed_anchors=["dataset", "family", "method", "policy"],
        test_population_policy=PopulationPolicy.ALLOW_DIFFERENT,
    )
    assert result.status == ComparabilityStatus.NOT_ASSESSED


def test_denominator_and_other_changes_block_comparison() -> None:
    mismatch = compare_conditions(
        condition(), condition(policy="fpr_matched", denominator="declared-labels"), ELIGIBLE, ELIGIBLE,
        changed_factor="policy", fixed_anchors=["dataset", "family", "method"],
    )
    assert mismatch.status == ComparabilityStatus.INCOMPARABLE
    undeclared = compare_conditions(
        condition(), condition(policy="fpr_matched", method="XGB"), ELIGIBLE, ELIGIBLE,
        changed_factor="policy", fixed_anchors=["dataset", "family", "method"],
    )
    assert undeclared.status == ComparabilityStatus.INCOMPARABLE
    assert "method" in " ".join(undeclared.reasons)


def test_missing_anchor_and_no_factor_change() -> None:
    result = compare_conditions(condition(), condition(policy="fpr_matched"), ELIGIBLE, ELIGIBLE,
                                changed_factor="policy", fixed_anchors=["dataset", "unrecorded"])
    assert result.status == ComparabilityStatus.NOT_ASSESSED
    same = compare_conditions(condition(), condition(), ELIGIBLE, ELIGIBLE,
                              changed_factor="policy", fixed_anchors=["dataset"])
    assert same.status == ComparabilityStatus.INCOMPARABLE


def test_scientific_eligibility_blocks_claim_relations() -> None:
    blocked = compare_conditions(condition(), condition(policy="fpr_matched"), ELIGIBLE, INELIGIBLE,
                                 changed_factor="policy", fixed_anchors=["dataset", "family", "method"])
    assert blocked.status == ComparabilityStatus.INCOMPARABLE
    assert evaluate_magnitude(0.8, 0.9, blocked).difference is None
    restricted = compare_conditions(condition(), condition(policy="fpr_matched"), ELIGIBLE, RESTRICTED,
                                    changed_factor="policy", fixed_anchors=["dataset", "family", "method"])
    assert restricted.status == ComparabilityStatus.RESTRICTED
    assert evaluate_ordinal(OrdinalScores(0.8, 0.7), OrdinalScores(0.6, 0.7), restricted).status == OutcomeStatus.NOT_COMPARABLE
    unassessed = compare_conditions(condition(), condition(policy="fpr_matched"), ELIGIBLE, UNKNOWN,
                                    changed_factor="policy", fixed_anchors=["dataset", "family", "method"])
    assert unassessed.status == ComparabilityStatus.NOT_ASSESSED


def test_ordinal_reversal_preservation_and_ties() -> None:
    gate = comparable()
    reversed_result = evaluate_ordinal(OrdinalScores(0.9, 0.8), OrdinalScores(0.7, 0.8), gate)
    assert reversed_result.relation == "observed_reversal"
    assert reversed_result.inferential_status == "not_assessed"
    assert reversed_result.baseline == pytest.approx(0.1)
    assert reversed_result.changed == pytest.approx(-0.1)
    assert evaluate_ordinal(OrdinalScores(0.9, 0.8), OrdinalScores(0.85, 0.8), gate).relation == "preserved"
    assert evaluate_ordinal(OrdinalScores(0.8, 0.8), OrdinalScores(0.9, 0.8), gate).relation == "tie_or_equivalence_transition"
    assert evaluate_ordinal(OrdinalScores(0.8, 0.8), OrdinalScores(0.7, 0.7), gate).relation == "tie_or_equivalence_preserved"


def test_prospective_equivalence_margin_is_optional() -> None:
    gate = comparable()
    assert evaluate_ordinal(OrdinalScores(0.8, 0.79), OrdinalScores(0.9, 0.91), gate,
                            practical_margin=0.02).relation == "tie_or_equivalence_preserved"
    with pytest.raises(ValueError, match="Practical margin"):
        evaluate_ordinal(OrdinalScores(0.8, 0.79), OrdinalScores(0.9, 0.91), gate,
                         practical_margin=-0.1)


def test_magnitude_keeps_signed_delta_not_stability_binary() -> None:
    result = evaluate_magnitude(0.8, 0.55, comparable())
    assert result.status == OutcomeStatus.OBSERVED
    assert result.relation == "magnitude_delta"
    assert result.difference == pytest.approx(-0.25)
    assert evaluate_magnitude(None, 0.5, comparable()).status == OutcomeStatus.NOT_ASSESSED
    assert evaluate_magnitude(0.5, float("nan"), comparable()).status == OutcomeStatus.NOT_ASSESSED


def test_categories_preserved_switched_undefined_and_missing() -> None:
    gate = comparable()
    d = CategoricalOutcome("DDoS")
    assert evaluate_categorical(d, CategoricalOutcome("DDoS"), gate).relation == "preserved"
    assert evaluate_categorical(d, CategoricalOutcome("Other"), gate).relation == "changed"
    undefined = evaluate_categorical(d, CategoricalOutcome(None, defined=False), gate)
    assert undefined.status == OutcomeStatus.UNDEFINED
    assert undefined.relation == "defined_to_undefined"
    assert evaluate_categorical(CategoricalOutcome(None, defined=False), CategoricalOutcome(None, defined=False), gate).relation == "undefined_in_both"
    assert evaluate_categorical(d, None, gate).status == OutcomeStatus.NOT_ASSESSED
    with pytest.raises(ValueError, match="Undefined outcome"):
        CategoricalOutcome("DDoS", defined=False)
