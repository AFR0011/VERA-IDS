# JISA Phase 5E — Validation-Visible Common-Case Results Frozen

Status: FROZEN  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Decision

Phase 5E is complete and frozen. The historical validation-visible rejector comparison must no longer use method means computed over unequal feasible holdout subsets as if they were directly comparable.

The revised reporting authority is the common-case package built from:

`outputs/12_jisa_finalization/15_rejector_tradeoff_analysis/budget_selected_case_metrics.csv`

with outputs under:

`outputs/13_jisa_q1_revision/phase5_visible_common_case/`

No model fitting, threshold tuning, resampling, or hypothesis testing was performed in Phase 5E.

## Primary 3% feasibility coverage

### CICIDS2017

- entropy: 6/6 holdouts feasible;
- family-conditional: 3/6 holdouts feasible;
- margin: 6/6 holdouts feasible;
- max-confidence: 6/6 holdouts feasible.

The three holdouts jointly feasible for all four methods are:

- Botnet;
- BruteForce;
- Web/App.

### CICIoT2023

- entropy: 4/6 holdouts feasible;
- family-conditional: 4/6 holdouts feasible;
- margin: 5/6 holdouts feasible;
- max-confidence: 6/6 holdouts feasible.

The four holdouts jointly feasible for all four methods are:

- BruteForce;
- DoS;
- Other;
- Scan/Recon.

## Primary 3% common-case method means

These means are computed only on the same jointly feasible holdouts within each dataset.

### CICIDS2017, n = 3 common holdouts

| Method | Unknown detection | Macro-F1 | False-unknown rate, all known | Overall rejection |
| --- | ---: | ---: | ---: | ---: |
| Entropy | 0.311916 | 0.817099 | 0.006769 | 0.007941 |
| Family-conditional | 0.941332 | 0.770914 | 0.060754 | 0.062703 |
| Margin | 0.286736 | 0.812838 | 0.005978 | 0.007090 |
| Max-confidence | 0.443256 | 0.812084 | 0.011619 | 0.013013 |

### CICIoT2023, n = 4 common holdouts

| Method | Unknown detection | Macro-F1 | False-unknown rate, all known | Overall rejection |
| --- | ---: | ---: | ---: | ---: |
| Entropy | 0.166322 | 0.762094 | 0.008567 | 0.016743 |
| Family-conditional | 0.111987 | 0.765118 | 0.007399 | 0.016958 |
| Margin | 0.148387 | 0.761222 | 0.008605 | 0.015133 |
| Max-confidence | 0.187387 | 0.769305 | 0.009881 | 0.019027 |

## Interpretation boundary

The common-case comparison reinforces the central conclusion that no uncertainty rule is uniformly superior. On CICIDS2017, family-conditional rejection attains substantially higher unknown-detection rate on the three jointly feasible holdouts, but at materially higher false-unknown and rejection rates and lower macro-F1. On CICIoT2023, max-confidence gives the highest common-case unknown-detection rate and macro-F1 among the four methods, while differences in rejection cost remain modest.

These are descriptive benchmark-case comparisons, not population estimators or significance-tested superiority claims.

## Cross-budget aggregation decision

The configured budgets are 1%, 3%, 5%, and 10%.

For both CICIDS2017 and CICIoT2023, the global fixed common-case set that is feasible for all four methods at all four budgets is empty:

- CICIDS2017: `n_fixed_common_holdouts = 0`;
- CICIoT2023: `n_fixed_common_holdouts = 0`.

Therefore no aggregate cross-budget method-mean curve is authorized. Any revised budget-sensitivity figure must show case-level trajectories, feasibility coverage, or another representation that does not change the denominator across methods/budgets.

## Manuscript consequence

- Replace the old unequal-feasible-subset rejector aggregate table with the 3% common-case table plus explicit feasibility coverage.
- Do not retain an aggregate multi-budget curve whose denominator changes across method or budget.
- If a budget figure is retained, use case-level or feasibility-coverage reporting and state that no non-empty globally fixed common holdout set existed across all four budgets.
- Historical row-bootstrap intervals remain conditional fixed-test-row intervals only; Phase 5E does not alter the Phase-5 inferential boundary.

## Machine-readable authority

Frozen local outputs:

- `outputs/13_jisa_q1_revision/phase5_visible_common_case/primary_3pct_feasibility_coverage.csv`
- `outputs/13_jisa_q1_revision/phase5_visible_common_case/primary_3pct_joint_feasible_cases.csv`
- `outputs/13_jisa_q1_revision/phase5_visible_common_case/primary_3pct_common_case_method_summary.csv`
- `outputs/13_jisa_q1_revision/phase5_visible_common_case/primary_3pct_common_case_paired_deltas.csv`
- `outputs/13_jisa_q1_revision/phase5_visible_common_case/budget_global_common_cases.csv`
- `outputs/13_jisa_q1_revision/phase5_visible_common_case/budget_feasibility_coverage.csv`
- `outputs/13_jisa_q1_revision/phase5_visible_common_case/build_manifest.json`
