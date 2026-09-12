# JISA Phase 5B — Claim-Oriented Statistical Refresh

Status: EXECUTION SPEC FROZEN  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Purpose

Phase 5B translates the Phase-5 provenance and joinability audits into a manuscript-facing statistical evidence package. It does not assume that a successful provenance preflight implies cluster-aware inference is available for every result surface.

The immediate objective is to replace row-level bootstrap p-values as the inferential centerpiece with evidence whose interpretation matches the retained experimental structure.

## Frozen evidence hierarchy

### Heldout-validation-blind CICIoT2023 surface

The primary Phase-3B evidence is the complete five-seed, six-holdout, four-rejector result surface at the 3% operating budget.

For this surface Phase 5B reports:

- one five-seed mean and sample standard deviation for each holdout × method cell;
- paired seed-level deltas relative to `max_confidence` within the same seed and holdout;
- equal-holdout descriptive summaries, where each omitted attack family contributes one equally weighted scenario mean;
- feasibility coverage.

The five seeds quantify sensitivity to fitted-model randomness under the fixed experiment design. They are not independent draws from a population of datasets or attacks. Therefore Phase 5B does not convert five-seed variation into population-level p-values or claim that it captures test-sample uncertainty.

### Row-level uncertainty

The Phase-5A decision artifact is authoritative for whether a score surface has direct provenance or only conditional row-level evidence.

- Synthetic positional `row_id` values are not admissible cluster identifiers.
- Candidate mapping headers are not sufficient for cluster inference until exact join cardinality and match rate are verified.
- Where direct/verified cluster provenance is absent, existing row-level bootstrap intervals may be retained only as uncertainty conditional on the observed test rows and fixed fitted model.
- Row-level bootstrap p-values are not used as population-level inferential evidence in the revised manuscript.

### Validation-visible rejector surface

Phase 5B inventories the retained `selected_methods.csv` artifacts from the finalization rejector-tradeoff runs. A subsequent common-case comparison is authorized only if the local files expose an auditable 3% budget, feasibility flag, method identity, and test metrics for the same holdout scenarios.

Method means computed on different feasible holdout subsets are not interpreted as method effects. Coverage and joint/common-case comparisons must be reported together.

## Paired comparison rule

The descriptive control is `max_confidence` for the heldout-validation-blind surface.

For every non-control method and each holdout, paired differences are computed within the same seed:

`delta = method metric - max_confidence metric`.

Phase 5B reports the five paired deltas, their mean, and sample standard deviation. No seed-level null-hypothesis p-value is promoted to a main inferential claim because `n=5` represents algorithmic randomness, not independent benchmark sampling units.

## Equal-holdout aggregation

Across holdouts, the manuscript-facing aggregate is the arithmetic mean of the six holdout-specific five-seed means. This prevents large holdout row counts from dominating the summary.

This quantity is a descriptive benchmark-scenario average. It is not a population estimator for future unknown attack families.

## Formal cluster-aware tests

Phase 5B does not silently launch formal cluster tests merely because provenance exists somewhere in the workspace.

If the Phase-5A decision indicates a directly provenance-bearing score surface, Phase 5B flags that surface for a separate Phase-5C cluster-cardinality/shared-unit validation. Formal cluster bootstrap or paired randomization is authorized only after that gate passes.

If no directly provenance-bearing claim surface exists, Phase 5C is unnecessary and Phase 5 may close with:

- five-seed robustness summaries;
- denominator-based Wilson intervals where appropriate;
- explicitly conditional row-level intervals;
- descriptive paired/common-case method comparisons;
- no population-level row-bootstrap p-values.

## Multiple testing

No new family of formal null-hypothesis tests is created in Phase 5B. Therefore no new BH or Bonferroni adjustment is applied here.

If Phase 5C becomes necessary, its comparison family and correction rule must be frozen before refreshed p-values are inspected.

## Outputs

The Phase-5B builder writes compact results under:

`outputs/13_jisa_q1_revision/phase5_statistical_refresh/`

Expected files:

- `blind_primary_3pct_five_seed_summary.csv`;
- `blind_primary_3pct_paired_seed_deltas.csv`;
- `blind_primary_3pct_paired_delta_summary.csv`;
- `blind_primary_3pct_equal_holdout_summary.csv`;
- `visible_selected_methods_inventory.csv`;
- `inferential_scope.json`;
- `build_manifest.json`.

No row-level score file, model, prepared dataset, or private local path is copied into the tracked repository.

## Interpretation boundary

The statistical refresh is conditional on the fixed trained models and benchmark construction unless an analysis explicitly resamples model seeds or defensible provenance clusters. Five-seed variation addresses training randomness. Row/Wilson intervals address fixed-test-set denominator uncertainty. Partition sensitivity addresses dependence assumptions. These are different uncertainty sources and must not be collapsed into one number.