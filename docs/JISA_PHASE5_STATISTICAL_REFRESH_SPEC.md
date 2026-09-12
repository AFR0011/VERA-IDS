# JISA Phase 5 — Statistical Refresh

Status: PREFLIGHT OPEN  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Purpose

Phase 5 replaces the manuscript's current reliance on row-level bootstrap p-values as the inferential centerpiece with the strongest statistically defensible uncertainty and paired-comparison procedures supported by the retained provenance.

The statistical refresh is an evidence-structure problem first and a resampling problem second. The sampling unit must be chosen from documented data provenance rather than from convenience or raw row count.

## Scientific questions

1. What is the strongest defensible independent or clustered sampling unit available for each retained result surface?
2. Where such units exist in adequate numbers, how uncertain are the reported effect sizes under cluster-aware resampling?
3. For paired rejector comparisons, do conclusions remain under a paired randomization/permutation procedure defined on shared independent units?
4. Where independent grouping cannot be reconstructed, which intervals must remain explicitly conditional row-level uncertainty rather than population-level inference?

## Hard rules

- Do not treat packet/flow rows as independent experimental replicates merely because they are numerous.
- Do not invent source-file/day/unit identifiers from row ordering or filenames when provenance does not support the mapping.
- Bootstrap is primarily an interval/effect-size tool, not the default source of paired p-values.
- Formal paired method tests use a predeclared paired randomization/permutation procedure where a defensible shared grouping unit exists.
- Paired methods must use the same units and the same resampling/randomization draw within a comparison.
- Test labels do not determine the grouping unit, test family, resample count, stopping rule, or multiple-testing correction.
- Multiple-testing families are frozen before reading refreshed p-values.
- Existing Wilson intervals for binomial rates may remain as descriptive denominator-based intervals where appropriate; they answer a different question from cluster-aware uncertainty.
- If the number of defensible clusters is too small for a stable inferential procedure, report that limitation instead of manufacturing precision.

## Candidate statistical surfaces

The preflight must inventory at least:

- Protocol-B validation-visible rejector comparisons used in the manuscript;
- CICIoT2023 five-seed unsupported-family / rejector surfaces;
- Phase-3 heldout-validation-blind rejection results;
- CICIDS2017 partition-sensitivity evidence where relevant;
- residual failure-destination/sink summaries if their uncertainty is retained in the main paper.

Not every surface requires a new significance test. The primary goal is to support the manuscript's central comparative claims with defensible uncertainty while reducing redundant null-hypothesis testing.

## Provisional inferential design, subject to provenance preflight

If a sufficiently granular shared source unit is recoverable:

- uncertainty intervals: cluster bootstrap with whole provenance units resampled with replacement;
- paired method tests: paired cluster-level randomization/permutation of method labels or paired signed unit-level contrasts, depending on metric decomposition;
- draws: target at least 10,000 randomization/permutation draws for formal paired tests when computationally feasible;
- confidence level: 95%;
- multiple-testing reporting: both Bonferroni family-wise control and Benjamini-Hochberg FDR, applied to explicitly declared comparison families.

If only a very small number of units is available, exact or enumerated paired randomization should be preferred when feasible, but the manuscript must emphasize low unit count.

If no valid grouping map exists for a surface, retain/recompute row-level bootstrap intervals only as **conditional on the observed test rows and fixed fitted model**, and do not present their p-values as population-level inferential evidence.

## Preflight gate

Before implementing any refreshed statistics, inventory:

- provenance/group columns present in prepared parquet schemas;
- source/file/day/unit mapping artifacts retained outside the row tables;
- row-level prediction/score files and whether they preserve a joinable provenance key;
- number and size distribution of candidate clusters where reconstructable;
- whether compared methods share exactly the same rows/units;
- whether CICIDS2017 and CICIoT2023 support different grouping strengths;
- whether group labels survive into the Phase-3 score surface or can be linked without ambiguous joins.

The preflight is read-only and may write only under `.release-audit/`.

## Stop conditions

Stop before implementing inferential tests if:

- the provenance-to-score join is ambiguous;
- candidate group identifiers are derived from test outcomes;
- compared methods do not share the same evaluable units;
- a claimed cluster unit is merely a surrogate inferred from row ordering;
- the chosen multiple-testing family depends on the observed refreshed p-values.

## Manuscript wording boundary

Cluster-aware intervals and randomization tests estimate uncertainty relative to the retained grouping structure and fixed trained models unless the procedure explicitly includes model retraining. They do not estimate full training-instability uncertainty. Five-seed reruns remain the evidence for seed sensitivity where available.
