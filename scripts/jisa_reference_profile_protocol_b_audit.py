#!/usr/bin/env python3
"""Read-only audit of literature-derived Protocol-B reference-profile runs.

The tracked compact summary proves matrix completeness and metric values, but does
not retain the validation-feasibility flag from tau_best.json. This audit closes
that last provenance gap using the owner's local outputs/11_reference_framework_eval
run tree. It performs no model fitting, threshold selection, or file mutation
outside .release-audit/.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "summaries" / "reference_profile_metric_drop.csv"
LOCAL_BEST = ROOT / "outputs" / "11_reference_framework_eval" / "protocol_b" / "summary" / "best_per_holdout.csv"
OUT = ROOT / ".release-audit" / "jisa_reference_profile_protocol_b"

EXPECTED = {
    ("adewole2025_xgb_profile", "CICIDS2017"): {"Botnet","BruteForce","DDoS","DoS","Scan/Recon","Web/App"},
    ("adewole2025_xgb_profile", "CICIoT2023"): {"Botnet","BruteForce","DDoS","DoS","Other","Scan/Recon"},
    ("neto2023_rf_profile", "CICIoT2023"): {"Botnet","BruteForce","DDoS","DoS","Other","Scan/Recon"},
}

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not SUMMARY.exists():
        raise RuntimeError(f"Missing tracked compact summary: {SUMMARY}")
    compact = pd.read_csv(SUMMARY)
    b = compact[compact["full_framework_surface"].astype(str) == "protocol_b_reference_profile"].copy()

    matrix_rows = []
    matrix_ok = True
    for (profile, dataset), expected in EXPECTED.items():
        g = b[(b["model_profile"].astype(str)==profile) & (b["dataset"].astype(str)==dataset)].copy()
        actual = set(g["task_or_holdout"].astype(str))
        vals = pd.to_numeric(g["full_framework_macro_f1_supported_labels"], errors="coerce")
        ok = len(g)==6 and actual==expected and vals.notna().all()
        matrix_ok &= ok
        matrix_rows.append({
            "model_profile": profile,
            "dataset": dataset,
            "n_rows": len(g),
            "holdouts": "|".join(sorted(actual)),
            "matrix_complete": ok,
            "protocol_b_macro_f1_mean": float(vals.mean()) if len(g) else None,
            "protocol_b_macro_f1_min": float(vals.min()) if len(g) else None,
            "protocol_b_macro_f1_max": float(vals.max()) if len(g) else None,
        })

    feasibility_rows = []
    local_found = LOCAL_BEST.exists()
    local_ok = False
    if local_found:
        best = pd.read_csv(LOCAL_BEST)
        needed = {"model_profile","dataset","holdout_family","run_dir"}
        missing = needed - set(best.columns)
        if missing:
            raise RuntimeError(f"Local best_per_holdout missing columns: {sorted(missing)}")
        local_ok = True
        for (profile, dataset), expected in EXPECTED.items():
            g = best[(best["model_profile"].astype(str)==profile) & (best["dataset"].astype(str)==dataset)].copy()
            if len(g) != 6 or set(g["holdout_family"].astype(str)) != expected:
                local_ok = False
            for _, row in g.iterrows():
                run_dir = Path(str(row["run_dir"]))
                if not run_dir.is_absolute():
                    run_dir = ROOT / run_dir
                tau_path = run_dir / "tau_best.json"
                tau_obj = {}
                if tau_path.exists():
                    try:
                        tau_obj = json.loads(tau_path.read_text(encoding="utf-8"))
                    except Exception:
                        tau_obj = {}
                ok_flag = tau_obj.get("ok")
                fallback = (ok_flag is False) or (not tau_path.exists())
                if fallback:
                    local_ok = False
                feasibility_rows.append({
                    "model_profile": profile,
                    "dataset": dataset,
                    "holdout_family": str(row["holdout_family"]),
                    "run_dir": str(run_dir),
                    "tau_best_exists": tau_path.exists(),
                    "tau_feasible_ok": ok_flag,
                    "fallback_or_unverified": fallback,
                    "tau": row.get("tau"),
                    "unknown_detection_rate": row.get("unknown_detection_rate"),
                    "false_unknown_rate_all_known": row.get("false_unknown_rate_all_known"),
                    "overall_reject_rate": row.get("overall_reject_rate"),
                    "macro_f1": row.get("macro_f1"),
                })

    pd.DataFrame(matrix_rows).to_csv(OUT / "tracked_matrix_audit.csv", index=False)
    pd.DataFrame(feasibility_rows).to_csv(OUT / "local_feasibility_audit.csv", index=False)

    decision = {
        "tracked_protocol_b_matrix_complete": bool(matrix_ok),
        "local_best_per_holdout_found": bool(local_found),
        "all_local_selected_reference_runs_validation_feasible_without_fallback": bool(local_ok) if local_found else None,
        "main_text_reference_protocol_b_authorized": bool(matrix_ok and local_found and local_ok),
        "claim_boundary": (
            "If authorized, use only as a single-seed, validation-visible literature-derived "
            "configuration sensitivity analysis. Do not call the profiles exact reproductions, "
            "do not compare their Protocol-B values causally with published source metrics, "
            "and do not merge them with the five-seed validation-blind primary analysis."
        ),
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    print("Tracked Protocol-B reference matrix:")
    print(pd.DataFrame(matrix_rows).to_string(index=False))
    print(f"tracked matrix complete: {matrix_ok}")
    print(f"local best_per_holdout found: {local_found}")
    if local_found:
        print(f"all selected reference runs feasible without fallback: {local_ok}")
        if feasibility_rows:
            print(pd.DataFrame(feasibility_rows)[
                ["model_profile","dataset","holdout_family","tau_feasible_ok","fallback_or_unverified","macro_f1"]
            ].to_string(index=False))
    print(f"MAIN_TEXT_REFERENCE_PROTOCOL_B_AUTHORIZED={decision['main_text_reference_protocol_b_authorized']}")
    print(f"audit: {OUT / 'decision.json'}")
    print("PASS=True")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
