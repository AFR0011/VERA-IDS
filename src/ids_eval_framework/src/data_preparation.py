"""Dataset preparation and external harmonization entrypoints."""

from __future__ import annotations

from typing import Any, Mapping

from ids_eval_framework.src.native_runtime import run_native_main


def run_preparation(config: Mapping[str, Any] | None = None, *, dry_run: bool = False) -> None:
    """Run reusable preprocessing, label mapping, and split construction."""
    from ids_eval_framework._native import prepare_datasets

    cfg = config or {}
    overrides = (cfg.get("preparation", {}) or {}).get("legacy_overrides")
    run_native_main(
        prepare_datasets,
        cfg_overrides=overrides,
        dry_run=dry_run,
    )
