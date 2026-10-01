# VERA-IDS v2026.10 manuscript-to-evidence provenance

This snapshot complements the public `v2026.08` release with the frozen aggregate evidence used by the Computer Communications manuscript.

| Manuscript surface | Evidence in this snapshot / tagged repository |
|---|---|
| Table 3, Protocol A component and system results | `outputs/summaries/protocol_a_core_summary.csv` |
| Table 3, direct benchmark macro-F1 and accuracy | `outputs/summaries/protocol_a_flat_vs_two_stage.csv` |
| Table 3, direct benchmark benign FPR | `release/v2026.10/raw/competitive_winner_test_results.csv` |
| Figure 2, five-seed direct-versus-staged comparison | `release/v2026.10/raw/controlled_direct_test_seed_summary.csv` plus `outputs/summaries/seed_reliability_summary.csv` |
| Figure 3, validation-blind 3% rejector results | `release/v2026.10/raw/blind_phase3b_five_seed_summary.csv` |
| Section 4.3 paired rejector deltas | `release/v2026.10/raw/blind_primary_3pct_paired_delta_summary.csv` |
| Table 4, validation-visible common-case comparison | `release/v2026.10/raw/visible_primary_3pct_common_case_method_summary.csv` |
| Figure 4, validation feasibility across budgets | `release/v2026.10/raw/visible_budget_feasibility_coverage.csv` |
| Web/App validation-to-test transfer example | `release/v2026.10/raw/webapp_transfer_summary.csv` |
| Figure 5, fixed-RF residual destinations | `release/v2026.10/raw/residual_failure_structure_primary_fixed_rf.csv` |
| Figure 6, partition sensitivity | `release/v2026.10/raw/cicids_partition_sensitivity_by_holdout.csv` |
| Table 5 / SI Table S7, literature-derived profiles | `release/v2026.10/raw/reference_manuscript_table_corrected.csv`, `reference_accuracy_vs_full_framework_summary.csv`, and `reference_published_source_anchor_catalog.csv` |
| External NSL-KDD and UNSW-NB15 stress tests | `outputs/summaries/external_protocol_a_summary.csv` |

## Evidence boundaries

- The direct-versus-staged experiment compares complete systems under an aligned operating requirement; it is not a pure causal architecture ablation.
- Validation-visible and validation-blind selection are different information conditions and are not represented as a seed-matched causal effect estimate.
- The CICIDS2017 partition comparison is a sensitivity analysis of two defensible constructions, not a repeated-seed partition experiment.
- Row-level bootstrap intervals are conditional on the observed test rows and fitted systems.
- The fixed-RF residual-destination diagnostic is a separate analysis lane from validation-selected rejector comparisons.

No benchmark datasets, prepared flow rows, raw prediction tables, or trained models are redistributed.
