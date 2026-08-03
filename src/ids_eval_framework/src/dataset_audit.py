"""Dataset/schema/leakage audit entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ids_eval_framework.src.native_runtime import run_native_main
from ids_eval_framework.src.paths import resolve_repo_path


def _legacy_overrides(config: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    return (config.get("dataset_audit", {}) or {}).get("legacy_overrides", {}).get(key)


def _path_overrides(overrides: Mapping[str, Any] | None, keys: set[str]) -> dict[str, Any] | None:
    if not overrides:
        return None
    result = dict(overrides)
    for key in keys:
        if key in result:
            result[key] = Path(resolve_repo_path(str(result[key])))
    return result


def run_dataset_audit(config: Mapping[str, Any] | None = None, *, dry_run: bool = False) -> None:
    """Run dataset summary/schema checks and split leakage checks."""
    from ids_eval_framework._native import analyze_datasets, audit_split_leakage

    cfg = config or {}
    audit_cfg = cfg.get("dataset_audit", {}) or {}
    enabled = set(audit_cfg.get("enabled_checks", ["schema_summary", "split_leakage"]))

    if "schema_summary" in enabled:
        run_native_main(
            analyze_datasets,
            cfg_overrides=_legacy_overrides(cfg, "analyze_datasets"),
            dry_run=dry_run,
        )

    if "split_leakage" in enabled:
        run_native_main(
            audit_split_leakage,
            cfg_overrides=_path_overrides(_legacy_overrides(cfg, "split_leakage"), {"dataset_root", "out_root"}),
            dry_run=dry_run,
        )
