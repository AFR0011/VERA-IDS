#!/usr/bin/env python3
"""
12.ExternalRobustnessValidation.py
==================================

Purpose
-------
Run limited external robustness validation on NSL-KDD and UNSW-NB15.

Scope
-----
- Fresh Protocol A preprocessing into a dedicated external-validation root
- Fresh Protocol A RF/XGB baselines with the same summary schema as the thesis-core table
- Protocol B support/status assessment only; external Protocol B runs are attempted
  only if the support audit says they are defensible
"""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from ids_eval_framework._native.full_scope_utils import (
    external_protocol_a_dataset_configs,
    external_protocol_b_dataset_configs,
    load_module,
    resolve_repo_path,
    safe_mkdir,
)


CFG: Dict[str, object] = {
    "prepare_script": "ids_eval_framework._native.prepare_datasets",
    "audit_script": "ids_eval_framework._native.protocol_b_support_audit",
    "core_runner_script": "ids_eval_framework._native.protocol_a_core",
    "processed_root_a": "processed_V5_external_validation",
    "processed_root_b": "processed_V5_external_protocolb",
    "runs_root_a": "runs_two_stage_V5_A_external_validation",
    "audit_root_b": "protocolB_support_audit_out_external_validation",
    "datasets_a": ["NSL-KDD", "UNSW-NB15"],
    "datasets_b": ["UNSW-NB15"],
    "external_summary_csv": "external_protocol_a_summary.csv",
}


def load_existing_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def write_external_snapshot(protocol_a_summary: str = "", protocol_b_support_status: str = "") -> None:
    snapshot = {
        "protocol_a_summary_exists": bool(protocol_a_summary and os.path.exists(protocol_a_summary)),
        "protocol_b_support_status_exists": bool(protocol_b_support_status and os.path.exists(protocol_b_support_status)),
        "processed_root_a_exists": os.path.exists(resolve_repo_path(str(CFG["processed_root_a"]))),
        "processed_root_b_exists": os.path.exists(resolve_repo_path(str(CFG["processed_root_b"]))),
        "runs_root_a_exists": os.path.exists(resolve_repo_path(str(CFG["runs_root_a"]))),
        "audit_root_b_exists": os.path.exists(resolve_repo_path(str(CFG["audit_root_b"]))),
    }
    snapshot_path = os.path.join(resolve_repo_path(str(CFG["runs_root_a"])), "external_validation_snapshot.json")
    safe_mkdir(resolve_repo_path(str(CFG["runs_root_a"])))
    pd.Series(snapshot).to_json(snapshot_path, indent=2)


def processed_dataset_ready(root: str, protocol: str, dataset: str) -> bool:
    base = os.path.join(root, protocol, dataset)
    required = [
        os.path.join(base, "train"),
        os.path.join(base, "val"),
        os.path.join(base, "test"),
        os.path.join(base, "SPLIT_REPORT.json"),
    ]
    return all(os.path.exists(p) for p in required)


def run_prepare(protocol: str, out_root: str, datasets: Dict[str, Dict[str, object]], active_names: List[str]) -> None:
    prep_mod = load_module(str(CFG["prepare_script"]), f"ids_prepare_external_{protocol}")
    prep_mod.CFG["protocol"] = protocol
    prep_mod.CFG["out_root"] = out_root
    prep_mod.CFG["datasets"] = datasets
    prep_mod.CFG["active_datasets"] = active_names
    safe_mkdir(out_root)
    for ds_name in active_names:
        if processed_dataset_ready(out_root, protocol, ds_name):
            print(f"[skip] processed {protocol} dataset already exists: {ds_name}")
            continue
        prep_mod.process_dataset(ds_name, datasets[ds_name])


def run_protocol_a_external() -> str:
    core = load_module(str(CFG["core_runner_script"]), "ids_protocol_a_external_runner")
    helper = core.load_helper_module(str(core.CFG["helper_script"]))

    core.CFG["processed_root"] = str(CFG["processed_root_a"])
    core.CFG["datasets"] = list(CFG["datasets_a"])
    core.CFG["runs_root"] = str(CFG["runs_root_a"])
    core.CFG["model_families"] = ["xgb", "rf"]
    core.CFG["max_train_rows"] = {"NSL-KDD": 200_000, "UNSW-NB15": 220_000}
    core.CFG["max_val_rows"] = {"NSL-KDD": 40_000, "UNSW-NB15": 50_000}
    safe_mkdir(str(CFG["runs_root_a"]))

    core.configure_helper_for_protocol_a(helper)
    helper.ensure_deps()

    # External validation is robustness-oriented, not a new broad search.
    helper.CFG["stage1_xgb_grid"]["grid"] = [
        {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        }
    ]
    helper.CFG["stage2_xgb_grid"]["grid"] = [
        {
            "n_estimators": 1400,
            "max_depth": 8,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        }
    ]
    helper.CFG["stage1_xgb_grid"]["max_train_rows"].update(core.CFG["max_train_rows"])
    helper.CFG["stage1_xgb_grid"]["max_val_rows"].update(core.CFG["max_val_rows"])
    helper.CFG["stage2_xgb_grid"]["max_train_rows"].update(core.CFG["max_train_rows"])
    helper.CFG["stage2_xgb_grid"]["max_val_rows"].update(core.CFG["max_val_rows"])
    out_copy = os.path.join(resolve_repo_path(str(CFG["runs_root_a"])), "summary", str(CFG["external_summary_csv"]))

    for ds_name in core.CFG["datasets"]:
        for model_family in core.CFG["model_families"]:
            run_dir, action = core.choose_run_dir(str(CFG["runs_root_a"]), ds_name, model_family)
            print(f"\n=== External Protocol A run: dataset={ds_name} model={model_family} ===")
            print(f"run_dir={run_dir}")
            if action == "skip_complete":
                print(f"[skip] found completed run for {ds_name} / {model_family}")
                continue
            if action == "resume_partial":
                print(f"[resume] found partial run for {ds_name} / {model_family}")
            core.run_protocol_a_eval(helper, ds_name, model_family, run_dir)
            print(f"[done] {ds_name} / {model_family}: {run_dir}")
            out_csv = core.write_protocol_a_summary(str(CFG["runs_root_a"]))
            pd.read_csv(out_csv).to_csv(out_copy, index=False)
            write_external_snapshot(protocol_a_summary=out_copy)

    out_csv = core.write_protocol_a_summary(str(CFG["runs_root_a"]))
    pd.read_csv(out_csv).to_csv(out_copy, index=False)
    print(f"Wrote external Protocol A summary: {out_copy}")
    write_external_snapshot(protocol_a_summary=out_copy)
    return out_copy


def run_protocol_b_support_status() -> str:
    out_csv = os.path.join(resolve_repo_path(str(CFG["audit_root_b"])), "protocol_b_support_status.csv")
    existing = load_existing_csv(out_csv)
    rows: List[Dict[str, object]] = existing.to_dict("records") if not existing.empty else []

    def has_dataset(dataset: str) -> bool:
        return any(str(r.get("dataset")) == dataset for r in rows)

    def upsert(row: Dict[str, object]) -> None:
        nonlocal rows
        rows = [r for r in rows if str(r.get("dataset")) != str(row["dataset"])]
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        write_external_snapshot(protocol_b_support_status=out_csv)

    # NSL-KDD has only the predefined train/test source files, so a train/val/test
    # Protocol B day/file audit is not defensible without inventing chronology.
    if not has_dataset("NSL-KDD"):
        upsert(
            {
                "dataset": "NSL-KDD",
                "protocol_b_status": "unsupported",
                "reason": "Only the predefined KDDTrain+/KDDTest+ source files are available locally, so there is no defensible train/val/test file chronology for Protocol B.",
                "eligible_holdouts": 0,
                "audit_root": "",
            }
        )

    b_datasets = external_protocol_b_dataset_configs()
    eligible_csv = os.path.join(str(CFG["audit_root_b"]), "eligible_holdouts_all.csv")
    if not has_dataset("UNSW-NB15") and not os.path.exists(eligible_csv):
        run_prepare("B_day_file", str(CFG["processed_root_b"]), b_datasets, list(CFG["datasets_b"]))

        audit_mod = load_module(str(CFG["audit_script"]), "ids_protocol_b_external_audit")
        audit_mod.CFG["processed_root"] = str(CFG["processed_root_b"])
        audit_mod.CFG["protocol"] = "B_day_file"
        audit_mod.CFG["datasets"] = list(CFG["datasets_b"])
        audit_mod.CFG["out_root"] = str(CFG["audit_root_b"])
        safe_mkdir(str(CFG["audit_root_b"]))
        audit_mod.main()

    n_eligible = 0
    if os.path.exists(eligible_csv):
        df = pd.read_csv(eligible_csv)
        n_eligible = int(len(df))
    if not has_dataset("UNSW-NB15"):
        upsert(
            {
                "dataset": "UNSW-NB15",
                "protocol_b_status": "valid_support" if n_eligible > 0 else "invalid_or_weak_support",
                "reason": "Support audit completed on the fresh external Protocol B preprocessing lane.",
                "eligible_holdouts": n_eligible,
                "audit_root": str(CFG["audit_root_b"]),
            }
        )
    return out_csv


def main() -> None:
    safe_mkdir(str(CFG["processed_root_a"]))
    a_datasets = external_protocol_a_dataset_configs()
    run_prepare("A_stratified", str(CFG["processed_root_a"]), a_datasets, list(CFG["datasets_a"]))
    run_protocol_a_external()
    support_csv = run_protocol_b_support_status()
    print(f"Wrote external Protocol B support status: {support_csv}")


if __name__ == "__main__":
    main()
