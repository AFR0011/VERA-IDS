"""Repository-relative path and configuration helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping

try:
    import yaml
except Exception:  # pragma: no cover - handled when config loading is requested
    yaml = None


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
FRAMEWORK_ROOT = REPO_ROOT
CONFIG_DIR = FRAMEWORK_ROOT / "config"
SRC_DIR = PACKAGE_ROOT / "src"
SCRIPTS_DIR = FRAMEWORK_ROOT / "scripts"
OUTPUTS_ROOT = FRAMEWORK_ROOT / "outputs"


def ensure_runtime_paths() -> None:
    """Make the public package importable from numbered scripts."""
    for path in (REPO_ROOT / "src", REPO_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def repo_path(*parts: str | os.PathLike[str]) -> str:
    """Join path parts under the public repository root."""
    return str(REPO_ROOT.joinpath(*(str(part) for part in parts)))


def framework_path(*parts: str | os.PathLike[str]) -> str:
    """Join path parts under the public repository root."""
    return str(FRAMEWORK_ROOT.joinpath(*(str(part) for part in parts)))


def output_path(*parts: str | os.PathLike[str]) -> str:
    """Join path parts under `ids_eval_framework/outputs/`."""
    return str(OUTPUTS_ROOT.joinpath(*(str(part) for part in parts)))


def resolve_repo_path(path: str | os.PathLike[str]) -> str:
    """Resolve an absolute path or a path relative to the repository root."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return repo_path(path_obj)


def resolve_framework_path(path: str | os.PathLike[str]) -> str:
    """Resolve an absolute path or a path relative to `ids_eval_framework/`."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return framework_path(path_obj)


def load_config(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Load a YAML config file. Missing config means an empty override set."""
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = FRAMEWORK_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load .yml config files.")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data or {}


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any] | None) -> MutableMapping[str, Any]:
    """Recursively update dictionaries without replacing entire nested config blocks."""
    if not updates:
        return base
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def ensure_dirs(*paths: str | os.PathLike[str] | None) -> None:
    """Create output directories, skipping empty values."""
    for path in paths:
        if path:
            Path(path).mkdir(parents=True, exist_ok=True)
