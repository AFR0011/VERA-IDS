#!/usr/bin/env python3
"""C03 historical candidate binding gate, READ ONLY and NO FITTING.

Run from VERA-IDS root with:
    python -B c03_candidate_fit_contract.py

Reads only one historical aggregate CSV, the approved policy JSON, the saved
prospective content-manifest JSON, and current Git HEAD. Does not import ML
libraries, open train/validation/test observations or score archives, fit models,
or write to any location. This is NOT a runner or a model-memory benchmark.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PINNED_HEAD = "ca13f1097f851ce6693658ccc0c6dd35480fabd6"
POLICY_SHA = "bf56f938d1fb193d2b383deeeec29f9d84759a4be829f6a0011f1bc20fe5e099"
BASELINE_SHA = "13e667a86a6872ff6192d1662085383e2b526752ceb783f8fc372b51487de9f4"
AGG = Path("outputs/12_jisa_finalization/09_ciciot_seed_reliability/protocol_b_loao_ciciot2023/seed_123/runs/aggregate_results.csv")
from c03_paths import POLICY, ORIGINAL_SNAPSHOT
EXPECTED = {
    "xgb_inv_family_clipped": ("xgb", "inv_family_clipped", "balanced"),
    "rf_inv_family_clipped": ("rf", "inv_family_clipped", "balanced"),
    "rf_class_weight_balanced": ("rf", "class_weight_balanced", "balanced"),
}
COLS = ("model_profile", "model_family", "stage1_weight_mode", "stage2_weight_mode", "apply_loao_stage1", "stage1_params", "stage2_params")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_params(value: str, key: str) -> dict:
    if not value.strip():
        raise ValueError(f"Missing {key} params")
    try:
        obj = json.loads(value)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Cannot parse {key} parameters: {value[:160]!r}") from exc
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"Non-dict or empty {key} parameters")
    if not all(isinstance(k, str) for k in obj):
        raise ValueError(f"Non-string parameter name in {key}")
    return obj


def truth(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def candidate_contract(rows: list[dict]) -> dict:
    selected = [r for r in rows if str(r.get("dataset", "")) == "CICIoT2023" and str(r.get("holdout_family", "")) == "Botnet"]
    if not selected:
        raise ValueError("No historical CICIoT2023/Botnet candidate records")
    unique = {}
    for row in selected:
        for field in COLS:
            if field not in row:
                raise ValueError(f"Required historical aggregate column missing: {field}")
        label = row["model_profile"].strip()
        if label not in EXPECTED:
            raise ValueError(f"Unexpected historical candidate label: {label!r}")
        fam, w1, w2 = EXPECTED[label]
        if (row["model_family"].strip(), row["stage1_weight_mode"].strip(), row["stage2_weight_mode"].strip()) != (fam, w1, w2):
            raise ValueError(f"Candidate identity/weighting drift: {label}")
        if not truth(row["apply_loao_stage1"]):
            raise ValueError(f"Strict supervised Stage1 LOAO is false: {label}")
        contract = {"family":fam, "stage1_weight":w1,"stage2_weight":w2,
                    "stage1_params":read_params(row["stage1_params"],f"{label} Stage1"),
                    "stage2_params":read_params(row["stage2_params"],f"{label} Stage2"),
                    "apply_loao_stage1":True}
        if label in unique and unique[label] != contract:
            raise ValueError(f"Conflicting repeated historical parameter records: {label}")
        unique[label] = contract
    if set(unique) != set(EXPECTED):
        raise ValueError(f"Incomplete historical candidate universe: {sorted(unique)}")
    for label, c in unique.items():
        if c["family"] == "rf" and (c["stage1_params"].get("n_estimators") != 300 or
                                         c["stage2_params"].get("n_estimators") != 300):
            raise ValueError(f"Unexpected RF tree count: {label}")
        if c["family"] == "rf" and c["stage2_params"].get("max_depth") is not None:
            raise ValueError(f"Unexpected RF Stage2 max_depth: {label}")
        if c["family"] == "xgb" and (c["stage1_params"].get("n_estimators") != 800 or
                                          c["stage2_params"].get("n_estimators") != 1400):
            raise ValueError("Unexpected historical XGB estimator counts")
    rf1, rf2 = unique["rf_inv_family_clipped"], unique["rf_class_weight_balanced"]
    share_rf_stage2 = rf1["stage2_params"] == rf2["stage2_params"] and rf1["stage2_weight"] == rf2["stage2_weight"]
    if not share_rf_stage2:
        raise ValueError("RF Stage2 configurations differ: cannot share fitted component")
    return {"profiles":unique,"rows_in_case":len(selected),"rf_stage2_config_equal":share_rf_stage2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=AGG)
    parser.add_argument("--manifest", type=Path, default=ORIGINAL_SNAPSHOT)
    parser.add_argument("--policy",type=Path,default=POLICY)
    parser.add_argument("--skip-git", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo = Path.cwd().resolve()
    print("=== C03 HISTORICAL CANDIDATE / NO-FIT PREFLIGHT ===")
    if not (repo/"src/ids_eval_framework/_native/protocol_b_grid.py").is_file():
        raise RuntimeError("Must run from VERA-IDS root")
    if not args.skip_git:
        p = subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True)
        if p.stdout.strip() != PINNED_HEAD:
            raise RuntimeError(f"Unreviewed Git HEAD: {p.stdout.strip()}; expected {PINNED_HEAD}")
    policy_path = args.policy.resolve()
    if sha(policy_path) != POLICY_SHA:
        raise RuntimeError("Approved policy content SHA256 mismatch")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("claim_id") != "C03" or policy.get("policy_status") != "SCIENTIFIC_POLICY_APPROVED__INTEGRATION_AND_EXECUTION_NOT_FROZEN":
        raise RuntimeError("Approved policy status/claim drift")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("snapshot_sha256") != BASELINE_SHA or len(manifest.get("data_files", [])) != 369:
        raise RuntimeError("Unexpected stored input baseline; DO NOT replace silently")
    # Important: this consumes the PRIOR verified inventory, not train/val/test data bytes.
    with args.aggregate.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    res = candidate_contract(rows)
    print("Git HEAD: pinned reference confirmed" if not args.skip_git else "Git HEAD: fixture only")
    print("Approved selector policy SHA256: verified")
    print(f"Prior input baseline: {BASELINE_SHA}")
    print(f"Historical aggregate: {args.aggregate}")
    print(f"Historical pilot candidate rows: {res['rows_in_case']}; distinct complete profiles: {len(res['profiles'])}")
    for label in EXPECTED:
        p=res["profiles"][label]
        print(f"  {label}:")
        print(f"    Stage1 {p['family']} / weight={p['stage1_weight']} / params={json.dumps(p['stage1_params'],sort_keys=True)}")
        print(f"    Stage2 {p['family']} / weight={p['stage2_weight']} / params={json.dumps(p['stage2_params'],sort_keys=True)}")
    print("RF Stage2 exact configuration/weighting match: True")
    print("DISTINCT MODEL COMPONENTS: 5 configurations (2 XGB + 2 RF Stage1 + 1 RF Stage2)")
    print("LIMIT: shared RF Stage2 DATA, SAMPLE-WEIGHT, RNG and FEATURE identity still require actual runner verification")
    print("RESOURCE: current 15.4 GiB machine and previously measured 4.3 GiB free are NOT cleared for fitting")
    print("NO FITTING OR TEST READ: this is a metadata-only preflight, not an executable model runner")
    print("RESULT: PASS historical candidate binding; pilot execution remains separately gated")
    return 0

if __name__=="__main__":
    try:
        sys.exit(main())
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
        print("BLOCKED:",exc,file=sys.stderr)
        sys.exit(2)
