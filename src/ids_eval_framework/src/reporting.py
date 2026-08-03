"""Reporting, external stress-test, and paper-pack entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ids_eval_framework.src.native_runtime import run_native_main
from ids_eval_framework.src.paths import (
    REPO_ROOT,
    OUTPUTS_ROOT,
    deep_update,
    resolve_repo_path,
)


OUTPUT_ROOT_ALIASES = {
    "processed_V5": OUTPUTS_ROOT / "02_prepared_data" / "processed_V5",
    "processed_V5_cicids17_recovery": OUTPUTS_ROOT / "02_prepared_data" / "processed_V5_cicids17_recovery",
    "processed_V5_external_validation": OUTPUTS_ROOT / "02_prepared_data" / "processed_V5_external_validation",
    "processed_V5_external_protocolb": OUTPUTS_ROOT / "02_prepared_data" / "processed_V5_external_protocolb",
    "runs_two_stage_V5_A_core": OUTPUTS_ROOT / "03_protocol_a_two_stage" / "runs_two_stage_V5_A_core",
    "runs_competitive_metrics": OUTPUTS_ROOT / "03b_protocol_a_flat_baseline" / "runs_competitive_metrics",
    "competitive_metrics_pack": OUTPUTS_ROOT / "03b_protocol_a_flat_baseline" / "competitive_metrics_pack",
    "protocolB_support_audit_out": OUTPUTS_ROOT / "04_protocol_b_support_audit" / "protocolB_support_audit_out",
    "protocolB_support_audit_out_cicids17_recovery": OUTPUTS_ROOT / "04_protocol_b_support_audit" / "protocolB_support_audit_out_cicids17_recovery",
    "protocolB_support_audit_out_external_validation": OUTPUTS_ROOT / "04_protocol_b_support_audit" / "protocolB_support_audit_out_external_validation",
    "protocolB_grid_runs step 2 stage-1 LOAO": OUTPUTS_ROOT / "05_protocol_b_loao" / "protocolB_grid_runs step 2 stage-1 LOAO",
    "protocolB_grid_runs step 3 - CICIDS2017 sweep": OUTPUTS_ROOT / "05_protocol_b_loao" / "protocolB_grid_runs step 3 - CICIDS2017 sweep",
    "protocolB_grid_runs step 5 - open-set baselines": OUTPUTS_ROOT / "06_open_set_rejection" / "protocolB_grid_runs step 5 - open-set baselines",
    "protocolB_grid_runs step 6 - sink-aware rejection": OUTPUTS_ROOT / "06b_sink_aware_rejection" / "protocolB_grid_runs step 6 - sink-aware rejection",
    "runs_two_stage_V5_A_external_validation": OUTPUTS_ROOT / "07_external_stress" / "runs_two_stage_V5_A_external_validation",
    "q2_validation_selected_protocol_b": OUTPUTS_ROOT / "08_statistics" / "q2_validation_selected_protocol_b",
    "q2_statistical_refresh": OUTPUTS_ROOT / "08_statistics" / "q2_statistical_refresh",
    "thesis_full_scope_pack": OUTPUTS_ROOT / "supplementary" / "exploratory" / "thesis_full_scope_pack",
    "drift_action_study": OUTPUTS_ROOT / "supplementary" / "drift" / "drift_action_study",
    "thesis_core_pack": OUTPUTS_ROOT / "09_paper_pack" / "thesis_core_pack",
    "thesis_finalization_pack": OUTPUTS_ROOT / "09_paper_pack" / "thesis_finalization_pack",
}


def resolve_output_aware_path(path_text: str) -> str:
    """Resolve legacy artifact paths into the framework output tree when possible."""
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    normalized = str(path_text).replace("\\", "/")
    for alias, target in sorted(OUTPUT_ROOT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        alias_norm = alias.replace("\\", "/")
        if normalized == alias_norm:
            return str(target)
        prefix = alias_norm + "/"
        if normalized.startswith(prefix):
            return str(target / normalized[len(prefix):])
    root_path = REPO_ROOT / path_text
    if root_path.exists():
        return str(root_path)
    return resolve_repo_path(path_text)


def _absolute_string_overrides(overrides: Mapping[str, Any] | None, keys: set[str]) -> dict[str, Any] | None:
    if not overrides:
        return None
    result = dict(overrides)
    for key in keys:
        if key in result:
            result[key] = resolve_repo_path(str(result[key]))
    if "protocol_b_sources" in result:
        sources = []
        for source in result["protocol_b_sources"]:
            item = dict(source)
            if "best_csv" in item:
                item["best_csv"] = resolve_repo_path(str(item["best_csv"]))
            sources.append(item)
        result["protocol_b_sources"] = sources
    return result


def _path_overrides(overrides: Mapping[str, Any] | None, keys: set[str]) -> dict[str, Any] | None:
    if not overrides:
        return None
    result = dict(overrides)
    for key in keys:
        if key in result:
            result[key] = Path(resolve_repo_path(str(result[key])))
    return result


def run_external_stress_tests(config: Mapping[str, Any] | None = None, *, dry_run: bool = False) -> None:
    from ids_eval_framework._native import external_robustness

    cfg = config or {}
    overrides = (cfg.get("external_stress", {}) or {}).get("legacy_overrides")
    run_native_main(
        external_robustness,
        cfg_overrides=overrides,
        dry_run=dry_run,
    )


def _run_thesis_core_pack(overrides: Mapping[str, Any] | None, *, dry_run: bool = False) -> None:
    from ids_eval_framework._native import build_thesis_core_pack

    old_cwd = Path.cwd()
    import os

    os.chdir(REPO_ROOT)
    try:
        module = build_thesis_core_pack
        patched = _absolute_string_overrides(
            overrides,
            {
                "protocol_a_summary_csv",
                "step4_aggregate_csv",
                "step4_runs_root",
                "out_root",
                "validation_selected_protocol_b_csv",
                "validation_selected_sink_csv",
            },
        )
        if patched:
            deep_update(module.CFG, patched)
        module.resolve_repo_path = resolve_output_aware_path
        if dry_run:
            print(f"[dry-run] native module: {module.__name__}")
            print(f"[dry-run] cwd={REPO_ROOT}")
            print(f"[dry-run] CFG keys: {sorted(module.CFG.keys())}")
            return
        module.main()
    finally:
        os.chdir(old_cwd)


def build_paper_pack(config: Mapping[str, Any] | None = None, *, dry_run: bool = False) -> None:
    from ids_eval_framework._native import build_competitive_pack, build_thesis_finalization_pack

    cfg = config or {}
    reporting_cfg = cfg.get("reporting", {}) or {}
    legacy = reporting_cfg.get("legacy_overrides", {})

    if reporting_cfg.get("build_competitive_pack", True):
        run_native_main(
            build_competitive_pack,
            cfg_overrides=_path_overrides(legacy.get("competitive_pack"), {"runs_root", "baseline_pack", "out_root"}),
            dry_run=dry_run,
        )
    if reporting_cfg.get("build_core_pack", True):
        _run_thesis_core_pack(legacy.get("thesis_core_pack"), dry_run=dry_run)
    if reporting_cfg.get("build_finalization_pack", True):
        finalization_overrides = legacy.get("thesis_finalization_pack")
        source_manuscript = (finalization_overrides or {}).get("source_manuscript")
        if (
            source_manuscript
            and reporting_cfg.get("skip_finalization_if_source_missing", True)
            and not Path(resolve_repo_path(str(source_manuscript))).exists()
        ):
            print(f"[skip] thesis finalization pack source manuscript not found: {source_manuscript}")
            return
        run_native_main(
            build_thesis_finalization_pack,
            cfg_overrides=_path_overrides(
                finalization_overrides,
                {
                    "thesis_pack",
                    "step5_root",
                    "step6_root",
                    "drift_action_root",
                    "processed_v5",
                    "processed_cicids_recovery",
                    "support_audit_cicids_recovery",
                    "out_root",
                    "source_manuscript",
                    "manuscript_dir",
                },
            ),
            dry_run=dry_run,
        )
