# JISA Phase 5C — Raw-score cluster-cardinality gate

Status: FROZEN BEFORE EXECUTION  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Why this gate exists

Phase 5B reported `PHASE5C_REQUIRED=True`, but that decision was intentionally broad: it was triggered by any claim-oriented row-level artifact classified as carrying direct provenance. That is not sufficient to authorize cluster-aware inference on the Phase-3 rejection claims.

The retained Phase-3 score-construction source creates `row_id` positionally with `np.arange(len(frame))`. Therefore `row_id` is not an admissible source/file/day/unit identifier. Before any cluster bootstrap or paired randomization is implemented, Phase 5C must verify provenance on the **exact raw score files from which the rejection metrics are replayed**, not on derived summaries, diagnostics, manifests, or tables that merely contain fields such as `source_score_table`.

## Scientific question

Do the exact Phase-3 and historical validation-visible raw test-score surfaces retain a genuine provenance column that can define shared resampling clusters for method comparisons?

## Admissible raw-score surface

A file is considered a raw rejection-score surface only if it contains, at minimum:

- `y_true_sys`
- `p_attack`
- `fam_pred_family`
- `fam_pmax`

The primary Phase-3 files are the per-seed/per-holdout `test_scores.csv.gz` artifacts under the frozen heldout-validation-blind score-generation namespace. Historical validation-visible `test_scores.csv.gz` files are audited separately.

Derived result tables such as `selected_points.csv`, `selected_methods.csv`, confidence-interval tables, statistical summaries, sink diagnostics, and files containing only a path back to a score table are not raw score surfaces and cannot establish cluster joinability.

## Admissible provenance columns

Only explicit source/group identifiers present on the raw score rows qualify, using exact normalized column names from this set:

- `source_file`
- `source_day`
- `file_day`
- `capture_day`
- `source_unit`
- `unit_id`
- `capture_id`
- `session_id`
- `pcap`
- `group_id`
- `identity_group`
- `exact_identity`
- `duplicate_group`
- `source_group`

`row_id`, `index`, `original_index`, row order, filenames of derived artifacts, and textual fields such as `source_score_table` are explicitly inadmissible.

## Cardinality gate

No inferential test is run in Phase 5C. For every admissible provenance column on a raw score surface, report:

- number of rows;
- number of non-missing clusters;
- missing-provenance fraction;
- minimum, median, mean, and maximum cluster size;
- largest-cluster share.

A candidate cluster variable is considered **usable for the next statistical implementation gate** only if all of the following hold for the relevant comparison surface:

1. it is directly present on the raw score rows;
2. it is not positional/synthetic;
3. at least 20 non-missing clusters are present;
4. missing provenance is at most 1% of rows;
5. no single cluster contains more than 50% of rows;
6. the compared methods are replayed from the same raw score surface, so cluster membership is shared exactly within each paired comparison.

The 20-cluster rule is a design threshold for authorizing the planned resampling implementation, not a claim that 20 clusters guarantees asymptotic validity. If fewer clusters exist, a later exact small-cluster randomization design may be considered separately, but it is not silently substituted here.

## Decision rules

- If the Phase-3 raw score files contain no admissible provenance column, Phase-3 cluster-aware row resampling is **not authorized**. Five-seed variability remains the primary robustness evidence, while row/Wilson intervals may only be described as conditional on the observed test rows and fixed fitted model.
- If an admissible column exists and passes the cardinality gate consistently across the Phase-3 primary cases, a separate Phase-5D implementation may perform cluster-aware intervals and predeclared paired tests.
- Historical validation-visible files are judged independently. Failure there does not invalidate Phase-3 seed summaries; it only limits population-style inference for that historical surface.
- No p-value, confidence interval, method ranking, or test result may influence the choice of provenance variable.

## Hard interpretation boundary

Cluster-aware intervals, if later authorized, quantify uncertainty relative to the retained source grouping and fixed fitted models. They do not include model-retraining instability. Five-seed reruns remain the evidence for training-randomness sensitivity.
