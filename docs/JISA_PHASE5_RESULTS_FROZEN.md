# JISA Phase 5 — Statistical Refresh Results Frozen

Status: FROZEN  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Decision

Phase 5 is closed without a Phase-5D cluster-resampling stage.

The raw-score cluster gate established:

- `PHASE5D_REQUIRED=False`;
- Phase-3 heldout-validation-blind raw test scores do not retain an admissible provenance cluster column on the exact score rows;
- validation-visible raw test scores likewise do not retain an admissible provenance cluster column on the exact score rows;
- therefore no cluster bootstrap or paired cluster-level randomization/permutation test is authorized for those score surfaces.

Synthetic positional `row_id` is not a provenance key and is not used as an inferential sampling unit.

## Retained statistical evidence

### Phase-3 heldout-validation-blind rejector surface

Primary evidence is the frozen five-seed (123–127) result surface at the 3% rejection budget. Each holdout × rejector cell contains five completed runs. Seed means, standard deviations, ranges, and paired seed-level deltas against max-confidence are retained as robustness evidence for training/selection randomness.

The seed dimension is not interpreted as population sampling uncertainty.

### Fixed-test-row uncertainty

Existing row-level bootstrap intervals, where retained, are described only as uncertainty conditional on the observed test rows and the fixed fitted model. Their bootstrap p-values are not promoted as population-level inferential evidence.

Existing Wilson intervals for unknown detection, false-unknown rate, and rejection rate may be retained as denominator-based descriptive intervals. They answer a different question from cluster-aware sampling uncertainty.

### CICIDS2017 partition sensitivity

Phase-4 partition sensitivity remains a separate validity result. It is not pooled with row-level or seed uncertainty and is not interpreted causally as a duplicate-removal effect.

## Formal-testing decision

No new formal null-hypothesis tests are added in Phase 5 because the retained raw score files do not support a defensible shared provenance-cluster unit.

This is preferable to reporting extremely precise tests whose effective sampling unit cannot be justified.

## Multiple testing

The historical bootstrap p-value families are not used as the revised manuscript's inferential centerpiece. Because no new formal Phase-5 tests are run, no new Bonferroni/BH testing family is created.

## Validation-visible rejector denominator cleanup

The remaining descriptive comparability issue was resolved in Phase 5E.

The historical validation-visible aggregate method means had been computed over unequal feasible holdout subsets. The revised reporting package now:

- reports feasibility coverage for every method;
- restricts the direct 3% method comparison to the same jointly feasible holdouts within each dataset;
- forbids an aggregate cross-budget method-mean curve when no holdout set remains feasible for every method at every reported budget.

The frozen authority is documented in `docs/JISA_PHASE5E_RESULTS_FROZEN.md` and machine-readable outputs under `outputs/13_jisa_q1_revision/phase5_visible_common_case/`.

Because the global fixed cross-budget common-case set is empty for both CICIDS2017 and CICIoT2023, no aggregate cross-budget rejector-mean curve is authorized in the revised manuscript.

## Manuscript wording boundary

Recommended wording:

> Five-seed summaries quantify sensitivity to training and selection randomness. Row-level bootstrap intervals, where shown, are conditional on the observed test rows and fixed fitted models. Cluster-aware resampling was not performed because admissible source/file/day provenance was not retained on the exact rejection-score rows; consequently, row-level bootstrap p-values are not treated as population-level inferential evidence.

Do not claim that the refreshed analysis estimates full retraining uncertainty or deployment-population uncertainty.

For validation-visible rejector comparisons, direct method means at the primary 3% budget must use the frozen joint-feasible common-case denominator and report feasibility coverage explicitly.

## Frozen machine-readable inputs

The local revision workspace contains the machine-readable Phase-5 artifacts under:

- `outputs/13_jisa_q1_revision/phase5_statistical_refresh/`
- `outputs/13_jisa_q1_revision/phase5_visible_common_case/`
- `.release-audit/jisa_phase5_statistics_preflight/`
- `.release-audit/jisa_phase5_joinability/`
- `.release-audit/jisa_phase5c_cluster_gate/`
- `.release-audit/jisa_phase5e_visible_common_case/`

The Phase-5B package includes the five-seed blind 3% summary, paired seed deltas, equal-holdout descriptive means, and inferential-scope manifest. The Phase-5C decision is the authority for the no-Phase-5D decision. The Phase-5E package is the authority for validation-visible common-case rejector reporting.

## Phase status

Phase 5 and its reporting cleanup are complete. No further statistical or rejector-denominator experiment is authorized before manuscript reconstruction unless a new provenance issue is discovered during the final claim-to-source audit.
