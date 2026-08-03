"""Support-audit helpers for Protocol B / LOAO validity."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping

import pandas as pd

from ids_eval_framework.src.native_runtime import run_native_main


def _require_mapping(rules: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = rules.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"support_audit_rules.{key} must be a mapping")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, path: str) -> int:
    if key not in mapping:
        raise ValueError(f"Missing required support-audit rule: {path}.{key}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{key} must be an integer")
    return int(value)


def canonical_support_rules_to_legacy(rules: Mapping[str, Any] | None) -> dict[str, Any]:
    """Translate framework support-audit rules into the legacy audit CFG shape."""
    if not rules:
        return {}

    benign = _require_mapping(rules, "min_benign_rows")
    family = _require_mapping(rules, "min_family_support")
    unknown = _require_mapping(rules, "unknown_support_for_holdout")
    if "min_known_families_after_holdout" in rules:
        min_known = rules["min_known_families_after_holdout"]
    else:
        min_known = rules.get("min_known_families_val_test")
    if isinstance(min_known, bool) or not isinstance(min_known, int):
        raise ValueError(
            "support_audit_rules.min_known_families_after_holdout must be an integer"
        )

    return {
        "require_unknown_in_val_when_tuning_tau": bool(
            rules.get("require_unknown_in_val_when_tuning_tau", True)
        ),
        "min_benign_val": _require_int(benign, "val", "support_audit_rules.min_benign_rows"),
        "min_benign_test": _require_int(benign, "test", "support_audit_rules.min_benign_rows"),
        "min_unknown_val": _require_int(
            unknown,
            "val",
            "support_audit_rules.unknown_support_for_holdout",
        ),
        "min_unknown_test": _require_int(
            unknown,
            "test",
            "support_audit_rules.unknown_support_for_holdout",
        ),
        "min_train_per_known_family": _require_int(
            family,
            "train",
            "support_audit_rules.min_family_support",
        ),
        "min_val_per_known_family": _require_int(
            family,
            "val",
            "support_audit_rules.min_family_support",
        ),
        "min_test_per_known_family": _require_int(
            family,
            "test",
            "support_audit_rules.min_family_support",
        ),
        "min_known_families_after_holdout": int(min_known),
        "require_all_remaining_train_families_to_be_valid": bool(
            rules.get("require_all_remaining_train_families_to_be_valid", True)
        ),
    }


def build_support_audit_overrides(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Merge canonical support rules into legacy overrides without silent drift."""
    cfg = config or {}
    audit_cfg = cfg.get("support_audit", {}) or {}
    if not isinstance(audit_cfg, Mapping):
        raise ValueError("support_audit must be a mapping")
    overrides = deepcopy(audit_cfg.get("legacy_overrides") or {})
    if not isinstance(overrides, MutableMapping):
        raise ValueError("support_audit.legacy_overrides must be a mapping")

    mapped_rules = canonical_support_rules_to_legacy(cfg.get("support_audit_rules"))
    if not mapped_rules:
        return dict(overrides) if overrides else None

    legacy_rules = overrides.get("support_rules")
    if legacy_rules is not None:
        if not isinstance(legacy_rules, Mapping):
            raise ValueError("support_audit.legacy_overrides.support_rules must be a mapping")
        conflicts = {
            key: {"canonical": value, "legacy_override": legacy_rules.get(key)}
            for key, value in mapped_rules.items()
            if key in legacy_rules and legacy_rules.get(key) != value
        }
        if conflicts:
            raise ValueError(
                "Conflicting support audit rules between support_audit_rules and "
                f"support_audit.legacy_overrides.support_rules: {conflicts}"
            )
        merged_rules = dict(legacy_rules)
        merged_rules.update(mapped_rules)
    else:
        merged_rules = mapped_rules

    overrides["support_rules"] = merged_rules
    return dict(overrides)


def eligible_holdouts(scoreboard: pd.DataFrame, rules: Mapping[str, Any]) -> pd.DataFrame:
    """Filter a support-audit scoreboard with explicit minimum-support rules."""
    result = scoreboard.copy()
    for column, minimum in rules.items():
        if column in result.columns and isinstance(minimum, (int, float)):
            result = result.loc[pd.to_numeric(result[column], errors="coerce") >= float(minimum)]
    return result.reset_index(drop=True)


def run_support_audit(config: Mapping[str, Any] | None = None, *, dry_run: bool = False) -> None:
    """Run Protocol B/LOAO support eligibility and manifest generation."""
    from ids_eval_framework._native import protocol_b_support_audit

    overrides = build_support_audit_overrides(config)
    run_native_main(
        protocol_b_support_audit,
        cfg_overrides=overrides,
        dry_run=dry_run,
    )
