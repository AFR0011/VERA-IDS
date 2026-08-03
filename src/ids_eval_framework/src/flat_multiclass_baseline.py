"""Protocol A flat/direct multiclass baseline entrypoints."""

from __future__ import annotations

from typing import Any, Mapping

from ids_eval_framework.src.native_runtime import run_native_main


def run_protocol_a_flat_baseline(config: Mapping[str, Any] | None = None, *, dry_run: bool = False) -> None:
    """Run the direct multiclass baseline lane without changing two-stage logic."""
    from ids_eval_framework._native import protocol_a_competitive

    cfg = config or {}
    overrides = (cfg.get("flat_multiclass_baseline", {}) or {}).get("legacy_overrides")
    run_native_main(
        protocol_a_competitive,
        cfg_overrides=overrides,
        dry_run=dry_run,
    )
