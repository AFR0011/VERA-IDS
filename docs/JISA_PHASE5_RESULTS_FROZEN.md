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

## Manuscript wording boundary

Recommended wording:

> Five-seed summaries quantify sensitivity to training and selection randomness. Row-level bootstrap intervals, where shown, are conditional on the observed test rows and fixed fitted models. Cluster-aware resampling was not performed because admissible source/file/day provenance was not retained on the exact rejection-score rows; consequently, row-level bootstrap p-values are not treated as population-level inferential evidence.

Do not claim that the refreshed analysis estimates full retraining uncertainty or deployment-population uncertainty.

## Frozen machine-readable inputs

The local revision workspace contains the machine-readable Phase-5 artifacts under:

- `outputs/13_jisa_q1_revision/phase5_statistical_refresh/`
- `.release-audit/jisa_phase5_statistics_preflight/`
- `.release-audit/jisa_phase5_joinability/`
- `.release-audit/jisa_phase5c_cluster_gate/`

The Phase-5B package includes the five-seed blind 3% summary, paired seed deltas, equal-holdout descriptive means, and inferential-scope manifest. The Phase-5C decision is the authority for the no-Phase-5D decision.

## Remaining reporting cleanup before manuscript rewrite

The statistical experiment sequence is frozen. One non-inferential reporting issue remains: the historical validation-visible rejector table/figure used unequal feasible case subsets across methods. Before rewriting the manuscript, rebuild that descriptive comparison on an explicit common/joint-feasible case denominator where the retained local tradeoff artifacts permit it, or remove the unequal-denominator method mean comparison if they do not.
