#!/usr/bin/env python3
"""Build the journal-final VERA-IDS claim ledger from frozen JISA analyses.

This script performs no training, model selection, threshold selection, or statistical
re-analysis beyond compact descriptive aggregation of already frozen outputs. Its purpose
is editorial: every manuscript-level claim is tied to the exact evidence surface that
supports it and to an explicit interpretation boundary.

The generated ledger is intentionally conservative. Historical thesis-era Protocol-B
winner tables are not used for journal-final CICIoT claims, and the recovered CICIDS2017
surface is used only in the explicit split-sensitivity comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/12_jisa_finalization/17_claim_ledger"

SURFACE_COMPARISON = ROOT / "outputs/12_jisa_finalization/07_group_safe_comparison/matched_surface_comparison.csv"
SURFACE_SUMMARY = ROOT / "outputs/12_jisa_finalization/07_group_safe_comparison/comparison_summary.json"
SEED_SUMMARY = ROOT / "outputs/12_jisa_finalization/10_ciciot_seed_analysis/analysis_summary.json"
SEED_METRICS = ROOT / "outputs/12_jisa_finalization/10_ciciot_seed_analysis/metric_summary_by_holdout.csv"
KNOWN_HELDOUT = ROOT / "outputs/12_jisa_finalization/11_known_vs_heldout/known_vs_heldout_primary_strict.csv"
KNOWN_HELDOUT_SUMMARY = ROOT / "outputs/12_jisa_finalization/11_known_vs_heldout/analysis_summary.json"
FAILURE_SUMMARY = ROOT / "outputs/12_jisa_finalization/12_failure_destinations/failure_structure_primary_fixed_rf.csv"
REJECTOR_SUMMARY = ROOT / "outputs/12_jisa_finalization/15_rejector_tradeoff_analysis/analysis_summary.json"
REJECTOR_PRIMARY = ROOT / "outputs/12_jisa_finalization/15_rejector_tradeoff_analysis/primary_3pct_case_metrics.csv"
BUDGET_AUDIT = ROOT / "outputs/12_jisa_finalization/16_rejector_budget_audit/analysis_summary.json"
BUDGET_METHODS = ROOT / "outputs/12_jisa_finalization/16_rejector_budget_audit/primary_3pct_method_generalization_summary.csv"
BUDGET_SIG = ROOT / "outputs/12_jisa_finalization/16_rejector_budget_audit/primary_3pct_udr_significance_summary.csv"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict:
    return json.loads(require(path).read_text(encoding="utf-8"))


def fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(float(x)):
        return "NA"
    return f"{float(x):.{digits}f}"


def claim_row(
    claim_id: str,
    section: str,
    claim: str,
    evidence: str,
    quantitative_support: str,
    boundary: str,
    status: str = "PRIMARY",
) -> dict[str, str]:
    return {
        "claim_id": claim_id,
        "status": status,
        "intended_section": section,
        "manuscript_claim": claim,
        "evidence_artifact": evidence,
        "quantitative_support": quantitative_support,
        "interpretation_boundary": boundary,
    }


def main() -> None:
    # Load frozen evidence.
    surface = pd.read_csv(require(SURFACE_COMPARISON))
    surface_json = load_json(SURFACE_SUMMARY)
    seed_json = load_json(SEED_SUMMARY)
    seed_metrics = pd.read_csv(require(SEED_METRICS))
    known = pd.read_csv(require(KNOWN_HELDOUT))
    known_json = load_json(KNOWN_HELDOUT_SUMMARY)
    failures = pd.read_csv(require(FAILURE_SUMMARY))
    rejector_json = load_json(REJECTOR_SUMMARY)
    rejector_primary = pd.read_csv(require(REJECTOR_PRIMARY))
    budget_json = load_json(BUDGET_AUDIT)
    budget_methods = pd.read_csv(require(BUDGET_METHODS))
    budget_sig = pd.read_csv(require(BUDGET_SIG))

    claims: list[dict[str, str]] = []

    # C1: CICIDS surface sensitivity.
    hist_udr_mean = float(pd.to_numeric(surface["historical_unknown_detection_rate"], errors="raise").mean())
    new_udr_mean = float(pd.to_numeric(surface["new_unknown_detection_rate"], errors="raise").mean())
    hist_f1_mean = float(pd.to_numeric(surface["historical_macro_f1"], errors="raise").mean())
    new_f1_mean = float(pd.to_numeric(surface["new_macro_f1"], errors="raise").mean())
    claims.append(claim_row(
        "C1",
        "Results: evaluation-surface sensitivity",
        "Family-level held-out conclusions changed materially between the recovered contiguous-within-day CICIDS2017 evaluation and the exact-representation-grouped sensitivity partition, even though aggregate mean UDR changed little and mean macro-F1 increased.",
        str(SURFACE_COMPARISON.relative_to(ROOT)),
        (
            f"mean UDR {fmt(hist_udr_mean)} -> {fmt(new_udr_mean)}; "
            f"mean absolute family UDR shift {fmt(surface_json['mean_absolute_udr_surface_shift'])}; "
            f"maximum absolute shift {fmt(surface_json['max_absolute_udr_surface_shift'])}; "
            f"mean macro-F1 {fmt(hist_f1_mean)} -> {fmt(new_f1_mean)}; "
            f"descriptive Spearman old-vs-new UDR {fmt(surface_json['descriptive_spearman_udr_old_vs_new'])}."
        ),
        "The two surfaces differ in partition composition as well as exact-representation overlap. Do not attribute the entire shift causally to duplicate removal. The grouped split is a sensitivity analysis, not a uniquely correct split.",
    ))

    # C2: seed stability.
    macro = seed_metrics[seed_metrics["metric"].astype(str) == "macro_f1"].copy()
    max_macro_sd = float(pd.to_numeric(macro["sd"], errors="raise").max())
    claims.append(claim_row(
        "C2",
        "Results: repeated-seed stability",
        "Validation-only candidate identity was stable across retraining seeds on CICIoT2023, while end-to-end macro-F1 varied only modestly.",
        str(SEED_SUMMARY.relative_to(ROOT)),
        (
            f"all 6 holdouts selected the same profile in all 5 seeds; "
            f"30 validation-selected cases from 90 candidate runs; maximum holdout macro-F1 SD {fmt(max_macro_sd, 4)}."
        ),
        "Repeated seeds quantify training/sampling variability under the fixed day/file evaluation surface. They do not establish robustness to a different dataset or partition policy.",
    ))

    # C3: known versus held-out behavior.
    ciciot_known = known[known["dataset"].astype(str) == "CICIoT2023"].copy()
    near_perfect = ciciot_known[pd.to_numeric(ciciot_known["known_recall"], errors="raise") >= 0.99]
    low_udr = near_perfect[pd.to_numeric(near_perfect["heldout_udr_mean"], errors="raise") <= 0.05]
    rho_ciciot = float(known_json["datasets"]["CICIoT2023"]["spearman_rho_descriptive"])
    rho_cicids = float(known_json["datasets"]["CICIDS2017"]["spearman_rho_descriptive"])
    claims.append(claim_row(
        "C3",
        "Results: known versus held-out recognition",
        "Strong recognition of an attack family when it is represented during development did not imply strong recognition of that family as Unknown after exclusion from training.",
        str(KNOWN_HELDOUT.relative_to(ROOT)),
        (
            f"CICIoT2023 descriptive Spearman known recall vs held-out UDR {fmt(rho_ciciot)}; "
            f"CICIDS2017 {fmt(rho_cicids)}; {len(low_udr)} CICIoT families had known recall >=0.99 but held-out UDR <=0.05."
        ),
        "The family-level correlations use n=6 families per dataset and are descriptive, not confirmatory inference. The comparison fixes RF class-balanced profiles to avoid model-selection confounding.",
    ))

    # C4: structured failure destinations.
    majority = failures[pd.to_numeric(failures["dominant_destination_share_of_failures_mean"], errors="raise") > 0.5]
    ciciot_fail = failures[failures["dataset"].astype(str) == "CICIoT2023"].copy()
    stable = ciciot_fail[pd.to_numeric(ciciot_fail["dominant_destination_stability_fraction"], errors="raise") >= 1.0 - 1e-12]
    max_share = float(pd.to_numeric(failures["dominant_destination_share_of_failures_mean"], errors="raise").max())
    claims.append(claim_row(
        "C4",
        "Results: failure destinations",
        "When held-out observations were not rejected as Unknown, their errors were typically concentrated into a small number of known-label destinations rather than diffusely distributed.",
        str(FAILURE_SUMMARY.relative_to(ROOT)),
        (
            f"dominant destination exceeded 50% of residual failures in {len(majority)}/{len(failures)} dataset-family cases; "
            f"all {len(stable)}/{len(ciciot_fail)} CICIoT dominant destinations were identical across 5 seeds; maximum dominant share {fmt(max_share)}."
        ),
        "Destination concentration is a model-behavior diagnostic. It does not imply semantic similarity or causal equivalence between held-out and sink families.",
    ))

    # C5: rejection heterogeneity at the primary validation budget.
    improvements = rejector_json["primary_improvement_counts_vs_max_confidence"]
    fc = improvements["family_conditional"]
    entropy = improvements["entropy"]
    margin = improvements["margin"]
    claims.append(claim_row(
        "C5",
        "Results: rejection trade-offs",
        "No rejection rule was uniformly superior under a common validation-time rejection budget; family-conditional rejection produced the largest gains in some cases but was less broadly feasible and could also reduce UDR.",
        str(REJECTOR_PRIMARY.relative_to(ROOT)),
        (
            f"at the 3% validation budget: family-conditional feasible in {fc['feasible_cases']}/12 cases, improving UDR in {fc['udr_improved_cases']} and worsening it in {fc['udr_worse_cases']} (mean delta {fmt(fc['mean_udr_delta'])}); "
            f"entropy feasible {entropy['feasible_cases']}/12 with mean delta {fmt(entropy['mean_udr_delta'])}; "
            f"margin feasible {margin['feasible_cases']}/12 with mean delta {fmt(margin['mean_udr_delta'])}."
        ),
        "The 3% budget is enforced on validation only. Family-conditional thresholding is conformal-inspired and carries no formal conformal coverage guarantee.",
    ))

    # C6: budget/constraint generalization.
    n_feasible = int(budget_json["validation_feasible_rows"])
    n_any = int(budget_json["test_any_selection_constraint_exceeded_rows"])
    n_budget = int(budget_json["test_budget_exceeded_rows"])
    claims.append(claim_row(
        "C6",
        "Results/Discussion: operating-point generalization",
        "Validation-feasible rejection operating points did not always preserve their development-time cost constraints on test data.",
        str(BUDGET_METHODS.relative_to(ROOT)),
        (
            f"{n_any}/{n_feasible} ({100*n_any/n_feasible:.1f}%) validation-feasible method/case operating points exceeded at least one selection constraint on test; "
            f"{n_budget}/{n_feasible} ({100*n_budget/n_feasible:.1f}%) exceeded the 3% rejection budget itself."
        ),
        "Test exceedance is retained as an out-of-sample result and is never used to reselect an operating point. Constraint generalization is descriptive.",
    ))

    # C7: adjusted paired evidence supports heterogeneity, not a universal winner.
    fc_sig = budget_sig[budget_sig["method"].astype(str) == "family_conditional"].iloc[0]
    margin_sig = budget_sig[budget_sig["method"].astype(str) == "margin"].iloc[0]
    entropy_sig = budget_sig[budget_sig["method"].astype(str) == "entropy"].iloc[0]
    claims.append(claim_row(
        "C7",
        "Results: rejection uncertainty",
        "The positive and negative UDR differences between rejectors and max-confidence were largely robust to multiplicity correction, supporting genuine family-specific heterogeneity rather than a single global ranking.",
        str(BUDGET_SIG.relative_to(ROOT)),
        (
            f"family-conditional: Bonferroni-significant positive {int(fc_sig['bonferroni_significant_positive_udr_cases'])}/{int(fc_sig['positive_udr_delta_cases'])}, "
            f"negative {int(fc_sig['bonferroni_significant_negative_udr_cases'])}/{int(fc_sig['negative_udr_delta_cases'])}; "
            f"margin: positive {int(margin_sig['bonferroni_significant_positive_udr_cases'])}/{int(margin_sig['positive_udr_delta_cases'])}, negative {int(margin_sig['bonferroni_significant_negative_udr_cases'])}/{int(margin_sig['negative_udr_delta_cases'])}; "
            f"entropy: positive {int(entropy_sig['bonferroni_significant_positive_udr_cases'])}/{int(entropy_sig['positive_udr_delta_cases'])}, negative {int(entropy_sig['bonferroni_significant_negative_udr_cases'])}/{int(entropy_sig['negative_udr_delta_cases'])}."
        ),
        "Paired bootstrap inference is conditional on the selected model and observed test sample. It does not quantify retraining instability; repeated-seed evidence is reported separately.",
    ))

    ledger = pd.DataFrame(claims)
    OUT.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUT / "jisa_claim_ledger.csv", index=False)

    md = [
        "# VERA-IDS JISA Claim Ledger",
        "",
        "This ledger is generated from the frozen journal-final analysis outputs. It is an editorial guardrail, not an additional experiment.",
        "",
    ]
    for row in claims:
        md.extend([
            f"## {row['claim_id']} — {row['intended_section']}",
            "",
            f"**Claim.** {row['manuscript_claim']}",
            "",
            f"**Evidence.** `{row['evidence_artifact']}`",
            "",
            f"**Quantitative support.** {row['quantitative_support']}",
            "",
            f"**Boundary.** {row['interpretation_boundary']}",
            "",
        ])
    (OUT / "JISA_CLAIM_LEDGER.md").write_text("\n".join(md), encoding="utf-8")

    payload = {
        "claim_count": len(claims),
        "status": "EXPERIMENTS_FROZEN_EDITORIAL_LEDGER_BUILT",
        "prohibited_primary_sources": [
            "historical CICIoT test-ranked protocol_b_best_per_holdout rows",
            "historical recovered CICIDS2017 surface except explicit split-sensitivity comparison",
        ],
        "next_stage": "figure-source tables and manuscript rewrite",
    }
    (OUT / "claim_ledger_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print("\nFrozen manuscript claims:")
    print(ledger[["claim_id", "intended_section", "manuscript_claim"]].to_string(index=False))
    print(f"\nWrote claim ledger to: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
