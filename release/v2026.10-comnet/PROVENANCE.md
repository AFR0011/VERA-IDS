# VERA-IDS v2026.10-comnet manuscript-to-evidence provenance

This snapshot freezes the compact aggregate evidence used by the Computer
Networks submission. It supersedes the Computer Communications-specific
mapping in `release/v2026.10/PROVENANCE.md` for the current manuscript while
retaining the earlier core evidence files unchanged.

| Manuscript / SI surface | Evidence in this tag |
|---|---|
| Table 2, prepared partitions | `outputs/summaries/data_audit_summary.csv` and preparation manifests/configuration |
| Table 3, Protocol-A component/system results | `outputs/summaries/protocol_a_core_summary.csv` |
| Table 3, broad direct benchmark | `outputs/summaries/protocol_a_flat_vs_two_stage.csv`; `release/v2026.10/raw/competitive_winner_test_results.csv` |
| Figure 2, five-seed direct-versus-staged comparison | `release/v2026.10/raw/controlled_direct_test_seed_summary.csv`; `outputs/summaries/seed_reliability_summary.csv` |
| Figure 3, validation-blind rejector results | `release/v2026.10/raw/blind_phase3b_five_seed_summary.csv` |
| Section 4.3 paired rejector deltas | `release/v2026.10/raw/blind_primary_3pct_paired_delta_summary.csv`; `outputs/summaries/protocol_b_paired_method_tests_1000.csv` |
| Table 4, validation-visible common-case comparison | `release/v2026.10/raw/visible_primary_3pct_common_case_method_summary.csv` |
| Figure 4, validation feasibility across budgets | `release/v2026.10/raw/visible_budget_feasibility_coverage.csv` |
| Figure 5, fixed-RF residual destinations | `release/v2026.10/raw/residual_failure_structure_primary_fixed_rf.csv` |
| Figure 6, CICIDS2017 partition sensitivity | `release/v2026.10/raw/cicids_partition_sensitivity_by_holdout.csv` |
| External NSL-KDD / UNSW-NB15 stress tests | `outputs/summaries/external_protocol_a_summary.csv` |
| Table 5 / SI Tables S7-S8, model-competence audit | `release/v2026.10-comnet/raw/model_competence_summary.csv`; `release/v2026.10-comnet/raw/primary_vera_five_seed_reference.csv`; `config/model_competence.yml` |
| SI Table S9, external-profile reconstruction provenance | `config/external_profile_robustness.yml`; `config/reference_framework_eval.yml`; source citations in the manuscript |
| SI Table S10, five-seed external Protocol-A transfer | `release/v2026.10-comnet/raw/external_protocol_a_summary.csv` |
| Table 6 / SI Table S11, external Protocol-B heterogeneity | `release/v2026.10-comnet/raw/external_protocol_b_family_heterogeneity_summary.csv` |
| SI Table S12, external Protocol-B holdout-level results | `release/v2026.10-comnet/raw/external_protocol_b_holdout_summary.csv` |
| Final comparison completion gate | `release/v2026.10-comnet/raw/final_campaign_gate.txt` |

## Evidence boundaries

- The direct-versus-staged experiment compares complete systems under an
  aligned operating requirement; it is not a pure causal architecture
  ablation.
- Validation-visible and validation-blind selection are different information
  conditions and are not represented as a seed-matched causal effect estimate.
- The CICIDS2017 partition comparison is a sensitivity analysis of two
  defensible constructions, not a repeated-seed partition experiment.
- Repeated seeds measure fitting, validation-selection, and capped-row
  subsampling variability on fixed prepared partitions.
- Row-level bootstrap intervals are conditional on the observed test rows and
  fitted systems.
- Protocol-B rejector analyses use the fitted model score distributions;
  calibration is assessed separately under Protocol A.
- The model-competence lane addresses the weak-primary-model alternative
  explanation; it is not a SOTA ranking.
- External Protocol A is a configuration-transfer diagnostic and uses its
  predeclared fixed transfer caps.
- The external Protocol-B campaign establishes configuration generality for
  held-out-family dependence specifically; it does not independently reproduce
  every conclusion-stability phenomenon in the manuscript.

No benchmark datasets, prepared flow rows, raw prediction tables, or trained
models are redistributed.
