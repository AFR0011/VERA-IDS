"""Execution helpers for package-local workflow modules."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, MutableMapping

from ids_eval_framework.src.paths import REPO_ROOT, deep_update


def run_native_main(
    module: ModuleType,
    *,
    cfg_overrides: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> Any:
    """Run a statically imported workflow module with temporary CFG overrides.

    Dry runs validate the native import and configuration surface without
    executing training or writing output. Module configuration is restored after
    every invocation so repeated CLI calls in one interpreter are deterministic.
    """
    if not hasattr(module, "main") or not callable(module.main):
        raise RuntimeError(f"Native workflow module has no main(): {module.__name__}")
    config_attr = "CFG" if hasattr(module, "CFG") else "CONFIG" if hasattr(module, "CONFIG") else None
    cfg = getattr(module, config_attr, None) if config_attr else None
    original = deepcopy(cfg) if isinstance(cfg, MutableMapping) else None
    if cfg_overrides:
        if not isinstance(cfg, MutableMapping):
            raise RuntimeError(f"Native workflow module has no mutable CFG/CONFIG: {module.__name__}")
        deep_update(cfg, cfg_overrides)
    try:
        if dry_run:
            print(f"[dry-run] native module: {module.__name__}")
            print(f"[dry-run] repository root: {REPO_ROOT}")
            if isinstance(cfg, MutableMapping):
                print(f"[dry-run] {config_attr} keys: {sorted(cfg.keys())}")
            return None
        old_cwd = Path.cwd()
        os.chdir(REPO_ROOT)
        try:
            return module.main()
        finally:
            os.chdir(old_cwd)
    finally:
        if original is not None:
            cfg.clear()
            cfg.update(original)
