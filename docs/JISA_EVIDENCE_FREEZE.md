# JISA Revision — Evidence Freeze and Manuscript Authority

Status: EVIDENCE FROZEN FOR MANUSCRIPT RECONSTRUCTION  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Purpose

This document marks the end of the claim-bearing experiment/reanalysis stage for the JISA revision. Manuscript reconstruction must now use the frozen phase documents and machine-readable outputs listed below. No additional model fitting, threshold tuning, resampling, or rejector-denominator redesign is authorized unless a new provenance contradiction is discovered during claim-to-source audit.

## Central article claim

The manuscript is an evaluation-methodology paper, not a new-detector or state-of-the-art classifier paper. Its central result is that conclusions about intrusion-detection quality can change materially as evaluation moves from conventional closed-set component metrics to composed end-to-end behavior, operating-constraint-matched comparisons, held-out-family evaluation, validation-blind rejector selection, rejection-cost accounting, residual failure destinations, and partition-validity sensitivity.

Competitive closed-set performance is retained as a validity check so that the evaluation effects cannot be dismissed as artifacts of obviously weak classifiers.

## Frozen evidence hierarchy

### 1. Controlled direct-vs-two-stage comparison

Authority: `docs/JISA_PHASE1_RESULTS_FROZEN.md`

Key boundary: architecture ranking depends on dataset and operating policy. Direct multiclass superiority under ordinary argmax can shrink, disappear, or reverse under a common validation-side operating constraint. This is a controlled system comparison, not a pure causal architecture ablation.

### 2. Published-reference provenance repair

Authority: `docs/JISA_PHASE2_RESULTS_FROZEN.md`

Published source metrics, VERA source-inspired profile results, and VERA primary results must remain separate provenance classes. Published-vs-VERA differences are contextual, not causal matched-protocol performance drops.

### 3. Canonical held-out-family evidence

Primary dataset: CICIoT2023.

Authority: `docs/JISA_PHASE3B_RESULTS_FROZEN.md`

Primary blind surface: five seeds, six held-out families, four rejectors, 3% nominal validation rejection budget. Held-out-family validation observations are unavailable to rejector selection. No single rejector dominates all held-out families; detectability is strongly family-dependent and rejector-dependent.

The result is named heldout-validation-blind or validation-blind. It must not be called fully inductive unknown-family blindness because historical unsupervised preprocessing exposure is retained by design.

Historical validation-visible results remain separate contextual evidence and are not paired seed-by-seed with the blind surface.

### 4. CICIDS2017 partition-validity sensitivity

Authority: `docs/JISA_PHASE4_RESULTS_FROZEN.md`

CICIDS2017 is a partition-sensitivity result, not the canonical unsupported-family benchmark. The recovered contiguous-within-day and group-safe/exact-identity-isolated constructions represent competing admissible dependence assumptions. Family-level and aggregate shifts may be reported, but no causal duplicate-removal claim is authorized.

### 5. Statistical interpretation

Authority: `docs/JISA_PHASE5_RESULTS_FROZEN.md`

No Phase-5D cluster inference is authorized because the exact raw rejection-score rows do not retain a defensible provenance cluster column. Five-seed summaries quantify training/selection randomness only. Row-bootstrap intervals, where retained, are conditional on the fixed observed test rows and fitted model. Historical row-bootstrap p-values are not the revised manuscript's population-level inferential centerpiece.

### 6. Validation-visible rejector common-case reporting

Authority: `docs/JISA_PHASE5E_RESULTS_FROZEN.md`

At the primary 3% validation budget, direct rejector means must be computed on jointly validation-feasible holdouts and accompanied by feasibility coverage.

Frozen common-case sets:

- CICIDS2017: Botnet, BruteForce, Web/App (3 holdouts);
- CICIoT2023: BruteForce, DoS, Other, Scan/Recon (4 holdouts).

No aggregate cross-budget rejector-mean curve is authorized because the fixed all-method/all-budget feasible holdout intersection is empty for both datasets.

## Figure/table reconstruction rules

1. Closed-set/direct comparison should use five-seed dot/error or compact table reporting and distinguish argmax from validation-constraint-matched operation.
2. Reference-profile comparison must be rebuilt from the corrected provenance surface and label published, source-inspired VERA, and VERA-primary values separately.
3. The main heldout-validation-blind figure should show CICIoT2023 holdout-specific UDR by rejector with five-seed variability at the primary 3% budget.
4. Historical validation-visible rejector aggregates must use the frozen common-case denominator and show feasibility coverage.
5. The previous aggregate multi-budget mean curve must be removed or replaced by case-level trajectories / feasibility-coverage reporting.
6. CICIDS2017 partition sensitivity should be shown as a compact paired-family plot or table.
7. Residual sink/failure-destination evidence remains an important main-text result where its retained provenance is unchanged.
8. Threshold-independent separability versus useful operating-point rejection remains a key result where the original score provenance is retained.

## Manuscript wording rules

- Use `held-out family`, `unsupported-family evaluation`, `heldout-validation-blind selection`, or `validation-blind` as appropriate.
- Do not claim live zero-day detection.
- Do not claim universal rejector superiority.
- Do not claim architecture alone causes direct-vs-two-stage differences.
- Do not call source-inspired profiles exact reproductions unless independently verified.
- Do not describe row-bootstrap p-values as population-level inference.
- Do not treat CICIDS2017 partition shifts as caused solely by duplicate removal.
- Prefer `competitive with recent high-performing closed-set IDS reports` over unqualified state-of-the-art language.

## Next stage

The claim-bearing evidence stage is complete. The next work is manuscript reconstruction:

1. final claim-to-source map;
2. Methods wording update;
3. Results reconstruction from frozen outputs;
4. figure/table rebuild;
5. substantive Discussion section;
6. Abstract/Introduction/Conclusion tightening after results are stable;
7. final JISA formatting and submission audit.
