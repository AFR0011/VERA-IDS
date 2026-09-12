#!/usr/bin/env python3
"""Read-only structural audit for frozen JISA Phase-4 partition-sensitivity outputs.

The audit verifies the six-holdout sensitivity table, recomputes all summary
statistics, checks that the group-safe finalist selection audit contains exactly
one validation-selected winner per holdout, verifies that no test metric entered
the selection rule, and validates all recorded input hashes in the build manifest.

Audit artifacts are written only under .release-audit/jisa_phase4_results/.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs" / "13_jisa_q1_revision" / "phase4_partition_validity"
AUDIT = REPO_ROOT / ".release-audit" / "jisa_phase4_results"

PAIRED = OUT / "cicids_partition_sensitivity_by_holdout.csv"
SUMMARY = OUT / "cicids_partition_sensitivity_summary.json"
SELECTION = OUT / "group_safe_validation_selection_audit.csv"
MANIFEST = OUT / "build_manifest.json"
EXPECTED_HOLDOUTS = ["Botnet", "BruteForce", "DDoS", "DoS", "Scan/Recon", "Web/App"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    required = [PAIRED, SUMMARY, SELECTION, MANIFEST]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing Phase-4 outputs:\n  " + "\n  ".join(missing))

    paired = pd.read_csv(PAIRED)
    summary = load_json(SUMMARY)
    selection = pd.read_csv(SELECTION)
    manifest = load_json(MANIFEST)

    checks: dict[str, Any] = {}

    # Paired surface structure.
    checks["paired_rows_6"] = len(paired) == 6
    checks["paired_holdouts_exact"] = sorted(paired["holdout_family"].astype(str).tolist()) == sorted(EXPECTED_HOLDOUTS)
    checks["paired_holdouts_unique"] = paired["holdout_family"].astype(str).nunique() == 6

    required_cols = {
        "recovered_unknown_detection_rate",
        "group_safe_unknown_detection_rate",
        "delta_group_safe_minus_recovered",
        "absolute_family_shift",
    }
    checks["paired_required_columns"] = required_cols.issubset(paired.columns)

    for c in ["recovered_unknown_detection_rate", "group_safe_unknown_detection_rate"]:
        vals = pd.to_numeric(paired[c], errors="coerce")
        checks[f"{c}_valid"] = bool(vals.notna().all() and vals.between(0, 1, inclusive="both").all())

    calc_delta = paired["group_safe_unknown_detection_rate"].astype(float) - paired["recovered_unknown_detection_rate"].astype(float)
    calc_abs = calc_delta.abs()
    checks["delta_recomputed"] = bool((calc_delta - paired["delta_group_safe_minus_recovered"].astype(float)).abs().max() <= 1e-12)
    checks["absolute_shift_recomputed"] = bool((calc_abs - paired["absolute_family_shift"].astype(float)).abs().max() <= 1e-12)

    # Summary recomputation.
    mean_r = float(paired["recovered_unknown_detection_rate"].mean())
    mean_g = float(paired["group_safe_unknown_detection_rate"].mean())
    mean_abs = float(paired["absolute_family_shift"].mean())
    max_abs = float(paired["absolute_family_shift"].max())
    max_holdout = str(paired.loc[paired["absolute_family_shift"].idxmax(), "holdout_family"])
    spearman = float(paired["recovered_unknown_detection_rate"].corr(paired["group_safe_unknown_detection_rate"], method="spearman"))

    summary_expect = {
        "n_common_holdouts": 6,
        "recovered_mean_unknown_detection_rate": mean_r,
        "group_safe_mean_unknown_detection_rate": mean_g,
        "mean_signed_shift_group_safe_minus_recovered": mean_g - mean_r,
        "mean_absolute_family_shift": mean_abs,
        "max_absolute_family_shift": max_abs,
        "max_shift_holdout_family": max_holdout,
        "spearman_family_rank_correlation": spearman,
    }
    summary_checks: dict[str, bool] = {}
    for key, expected in summary_expect.items():
        actual = summary.get(key)
        if isinstance(expected, str):
            ok = str(actual) == expected
        elif isinstance(expected, int):
            ok = int(actual) == expected
        else:
            ok = close(float(actual), float(expected))
        summary_checks[key] = bool(ok)
    checks["summary_recomputed"] = all(summary_checks.values())

    # Selection audit. Current frozen surface is expected to have 12 finalists and 6 winners.
    selected_col = "selected_by_phase4_validation_rule"
    checks["selection_audit_rows_12"] = len(selection) == 12
    checks["selection_selected_column"] = selected_col in selection.columns
    if selected_col in selection.columns:
        selected_mask = selection[selected_col].astype(str).str.lower().isin(["true", "1"])
        winners = selection[selected_mask].copy()
        checks["selection_winners_6"] = len(winners) == 6
        checks["selection_one_winner_per_holdout"] = (
            len(winners) == 6
            and winners["holdout_family"].astype(str).nunique() == 6
            and sorted(winners["holdout_family"].astype(str).tolist()) == sorted(EXPECTED_HOLDOUTS)
        )
        if "group_safe_selected_run_name" in paired.columns and "run_name" in winners.columns:
            left = paired.set_index("holdout_family")["group_safe_selected_run_name"].astype(str).sort_index()
            right = winners.set_index("holdout_family")["run_name"].astype(str).sort_index()
            checks["selected_run_names_match_paired_surface"] = left.equals(right)
        else:
            checks["selected_run_names_match_paired_surface"] = False
    else:
        checks["selection_winners_6"] = False
        checks["selection_one_winner_per_holdout"] = False
        checks["selected_run_names_match_paired_surface"] = False

    rule_text = " ".join(selection.get("selection_rule", pd.Series(dtype=str)).astype(str).unique()).lower()
    checks["selection_rule_validation_only"] = (
        "validation stage-2" in rule_text
        and "validation stage-1" in rule_text
        and "test" not in rule_text
    )
    checks["summary_declares_no_test_selection"] = summary.get("group_safe_test_metrics_used_for_selection") is False
    checks["manifest_declares_no_test_selection"] = manifest.get("test_labels_used_for_new_selection") is False
    checks["manifest_declares_no_training"] = manifest.get("new_model_training") is False
    checks["manifest_forbids_duplicate_causality"] = manifest.get("causal_duplicate_removal_claim_authorized") is False

    # Hash every manifest input and verify it still matches the build-time artifact.
    hash_rows = []
    hash_ok = True
    for label, rec in dict(manifest.get("inputs", {}) or {}).items():
        path = Path(str(rec.get("path", "")))
        if not path.is_absolute():
            path = REPO_ROOT / path
        exists = path.exists()
        actual = sha256_file(path) if exists else ""
        expected = str(rec.get("sha256", ""))
        ok = bool(exists and expected and actual == expected)
        hash_ok = hash_ok and ok
        hash_rows.append({"input": label, "path": str(path), "exists": exists, "expected_sha256": expected, "actual_sha256": actual, "match": ok})
    checks["all_manifest_input_hashes_match"] = hash_ok and bool(hash_rows)
    pd.DataFrame(hash_rows).to_csv(AUDIT / "input_hash_audit.csv", index=False)

    passed = all(bool(v) for v in checks.values())
    payload = {
        "checks": checks,
        "summary_checks": summary_checks,
        "recomputed": summary_expect,
        "PASS": passed,
    }
    (AUDIT / "audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"paired rows: {len(paired)}/6")
    print(f"paired holdouts exact: {checks['paired_holdouts_exact']}")
    print(f"summary recomputation matches: {checks['summary_recomputed']}")
    print(f"selection audit rows: {len(selection)}/12")
    print(f"validation-selected winners: {int(selection[selected_col].astype(str).str.lower().isin(['true','1']).sum()) if selected_col in selection.columns else 0}/6")
    print(f"one winner per holdout: {checks['selection_one_winner_per_holdout']}")
    print(f"selected run names match paired surface: {checks['selected_run_names_match_paired_surface']}")
    print(f"selection rule validation-only: {checks['selection_rule_validation_only']}")
    print(f"input hashes match: {checks['all_manifest_input_hashes_match']}")
    print("")
    print(f"recovered mean UDR: {mean_r:.6f}")
    print(f"group-safe mean UDR: {mean_g:.6f}")
    print(f"mean absolute family shift: {mean_abs:.6f}")
    print(f"max absolute family shift: {max_abs:.6f} ({max_holdout})")
    print(f"Spearman family-rank correlation: {spearman:.6f}")
    print(f"audit: {AUDIT / 'audit.json'}")
    print(f"PASS={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
