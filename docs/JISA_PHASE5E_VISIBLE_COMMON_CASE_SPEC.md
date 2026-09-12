# JISA Phase 5E — Validation-visible rejector common-case reporting cleanup

Status: FROZEN FOR EXECUTION  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Purpose

Phase 5E repairs a reporting comparability problem in the historical validation-visible rejector analysis. The existing aggregate method means were computed over method-specific feasible subsets, so different methods could be averaged over different held-out families. Those means are not valid head-to-head method comparisons.

This phase performs no model fitting, threshold tuning, resampling, or hypothesis testing. It only rebuilds descriptive summaries from retained case-level artifacts with explicit budget and feasibility columns.

## Frozen source surface

Primary source:

`outputs/12_jisa_finalization/15_rejector_tradeoff_analysis/budget_selected_case_metrics.csv`

The preflight established that this file contains explicit dataset, holdout, method, budget, feasibility, and test-metric fields and contains all 48 expected method-by-case rows at the primary 3% budget.

The following files are supporting diagnostics only and must not replace the primary source silently:

- `outputs/12_jisa_finalization/15_rejector_tradeoff_analysis/primary_3pct_case_metrics.csv`
- `outputs/12_jisa_finalization/16_rejector_budget_audit/primary_3pct_test_constraint_generalization.csv`

## Method identity

Historical method labels are normalized only for reporting:

- `control_tau` / `max_confidence` -> `max_confidence`
- `margin_reject` / `margin` -> `margin`
- `entropy_reject` / `entropy` -> `entropy`
- `conformal_reject` / `family_conditional` -> `family_conditional`

No method result is otherwise altered.

## Primary 3% comparison

For each dataset separately:

1. Restrict to budget = 0.03.
2. Require exactly one row per holdout-method cell.
3. Report method-specific feasibility coverage over all available holdouts.
4. Define the **joint-feasible common-case set** as holdouts for which all four methods are validation-feasible at 3%.
5. Compute method means only on this same joint-feasible holdout set.
6. Compute paired descriptive deltas versus `max_confidence` only on the same joint-feasible holdouts.

This common-case table replaces any main-text aggregate comparison whose methods were averaged over unequal feasible subsets.

## Budget-sensitivity comparison

The budget curve may be aggregated only over a denominator that is fixed across both methods and budgets.

For each dataset:

1. Determine the configured budgets present in the source surface.
2. Define the **global fixed common-case set** as holdouts that contain one row for every method at every budget and are feasible for every method at every budget.
3. If this set is non-empty, compute budget-by-method means using only that same holdout set at every budget.
4. If the set is empty, no aggregate cross-budget mean curve is authorized. Preserve case-level curves or a feasibility-coverage display instead.

The denominator must never change from one method or budget point to another in an aggregate curve.

## Metrics

Primary descriptive metrics retained when available:

- unknown detection rate;
- supported/system macro-F1 (`macro_f1` on the historical surface);
- false-unknown rate over all known traffic;
- overall rejection rate;
- benign rejection rate;
- benign-to-family false-positive rate;
- accuracy.

Existing row-bootstrap confidence intervals may be carried through only as conditional fixed-test-row intervals. They are not recomputed or promoted to population-level inference here.

## Validation versus test feasibility

The `feasible` flag in the selected-case source defines the operating-point selection feasibility used for the common-case denominator. Test-set constraint exceedances are a separate generalization diagnostic. A case is not retrospectively removed from the common-case denominator because of test-set constraint violation.

## Outputs

Write only under:

`outputs/13_jisa_q1_revision/phase5_visible_common_case/`

Required outputs:

- `primary_3pct_feasibility_coverage.csv`
- `primary_3pct_joint_feasible_cases.csv`
- `primary_3pct_common_case_method_summary.csv`
- `primary_3pct_common_case_paired_deltas.csv`
- `budget_global_common_cases.csv`
- `budget_fixed_denominator_summary.csv` (only rows for datasets with a non-empty fixed set)
- `budget_feasibility_coverage.csv`
- `build_manifest.json`

## Claim boundary

Phase 5E supports descriptive comparability only. It does not authorize a claim that one rejector is universally superior, nor does it create a new significance-testing family. If common-case results differ from the old unequal-subset means, the common-case results take precedence for direct method comparison.
