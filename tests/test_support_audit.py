from __future__ import annotations

import pandas as pd
import pytest

from ids_eval_framework.src.support_audit import (
    build_support_audit_overrides,
    canonical_support_rules_to_legacy,
    eligible_holdouts,
)


RULES = {
    "require_unknown_in_val_when_tuning_tau": True,
    "min_benign_rows": {"val": 200, "test": 200},
    "min_family_support": {"train": 200, "val": 200, "test": 200},
    "unknown_support_for_holdout": {"val": 200, "test": 200},
    "min_known_families_after_holdout": 2,
    "require_all_remaining_train_families_to_be_valid": True,
}


def test_support_rules_map_without_drift() -> None:
    mapped = canonical_support_rules_to_legacy(RULES)
    assert mapped["min_unknown_test"] == 200
    assert mapped["min_train_per_known_family"] == 200


def test_conflicting_legacy_rule_fails_clearly() -> None:
    config = {
        "support_audit_rules": RULES,
        "support_audit": {"legacy_overrides": {"support_rules": {"min_unknown_test": 1}}},
    }
    with pytest.raises(ValueError, match="Conflicting support audit rules"):
        build_support_audit_overrides(config)


def test_tiny_support_audit_filters_ineligible_holdout() -> None:
    frame = pd.DataFrame({"unknown_test": [250, 150], "known_train": [300, 300]})
    result = eligible_holdouts(frame, {"unknown_test": 200, "known_train": 200})
    assert result.to_dict("records") == [{"unknown_test": 250, "known_train": 300}]
