from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "ids_eval_framework",
        "ids_eval_framework.metrics",
        "ids_eval_framework.src.paths",
        "ids_eval_framework.src.calibration",
        "ids_eval_framework.src.error_decomposition",
        "ids_eval_framework.src.support_audit",
        "ids_eval_framework.src.support_sensitivity",
    ],
)
def test_lightweight_modules_import(module: str) -> None:
    assert importlib.import_module(module) is not None


def test_heavy_engine_import_when_runtime_dependencies_exist() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    assert importlib.import_module("ids_eval_framework.src.two_stage_engine") is not None
