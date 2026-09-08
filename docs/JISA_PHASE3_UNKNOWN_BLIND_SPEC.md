# JISA Phase 3 — Unknown-Blind Protocol-B Specification

Status: FROZEN CONCEPTUAL DESIGN; PREFLIGHT REQUIRED BEFORE EXECUTION
Date: 2026-09-08
Branch: `jisa-q1-revision`
Primary dataset: CICIoT2023
Seeds: 123, 124, 125, 126, 127

## Purpose

This phase measures how Protocol-B conclusions change when the held-out attack family
is unavailable not only during training but also during validation-based selection.
The existing Protocol-B result remains a separate **validation-visible** surface.
The new result is an **unknown-blind threshold/selection** surface. It is not described
as live zero-day detection.

## Non-negotiable visibility rule

For a holdout family H, rows belonging to H in the validation split are unavailable
to every operation that can influence the final operating point or selected model
profile. They may not influence:

1. Stage-1 threshold selection;
2. Stage-1 validation criteria used for profile selection;
3. Stage-2/rejector threshold selection;
4. calibration or score-normalization fitting used by the blind lane;
5. rejector method/operating-point selection;
6. feasibility decisions;
7. tie-breaking among candidate model profiles.

Held-out-family validation rows may be scored only after all blind-lane choices for
that run are frozen, for diagnostic reporting that is clearly marked evaluation-only.
They cannot retroactively alter selection.

Test labels and test scores are evaluation-only throughout.

## Training surface

The training definition remains strict LOAO:

- the holdout family is removed from Stage-1 attack training;
- the holdout family is absent from Stage-2 training;
- represented attack families remain the known-family set;
- preprocessing is fitted on training data only;
- the existing prepared CICIoT2023 Protocol-B partition is reused without
  repartitioning.

This phase changes validation visibility, not the train/test partition.

## Candidate profile policy

Historical five-seed Protocol-B evidence contains three candidate profiles per
holdout and reports stable selected profiles across seeds. Historical profile
selection used validation Stage-2 macro-F1 first, Stage-1 AUROC second, then a
lexical tie-break. Stage-2 validation macro-F1 is computed on represented attack
families only, while the historical Stage-1 AUROC may include the held-out family.

Therefore historical selected profiles may be frozen into the blind lane **only if a
preflight audit proves that every seed × holdout winner is uniquely determined by
Stage-2 macro-F1 alone**, with a strictly positive top-vs-runner-up margin. If any
winner requires the Stage-1 AUROC or lexical tie-break, the blind lane must rerun all
candidate profiles for that seed/holdout and select using heldout-free validation
metrics only.

No historical test metric may select a blind-lane profile.

## Stage-1 operating-point selection

The blind lane preserves the corresponding Protocol-B Stage-1 search policy but
applies it only to benign plus represented-family validation rows. The held-out
family is removed before threshold-search inputs are constructed.

Unless a pre-execution provenance audit proves a different historical setting is the
relevant comparison surface, preserve the native Protocol-B search definition:

- target benign-FPR candidates: 0.01 and 0.02;
- minimum family support: 100;
- objectives: minimum family recall and p10 family recall;
- p10 quantile: 0.10;
- 40 threshold sweep points;
- tie-break: objective, then total attack TPR.

The selected threshold is frozen before any held-out-family validation or test
metric is examined.

## Rejection/selective-prediction budgets

The manuscript-facing primary overall rejection budget is 3%.
Sensitivity budgets are 1%, 5%, and 10%.

Known-traffic constraints are preserved across visible and blind surfaces:

- benign-to-family FPR <= 0.02;
- benign rejection <= 0.10;
- false-unknown rate among all known traffic <= 0.05;
- false-unknown rate among known attacks <= 0.10.

In the blind lane, these constraints are computed only from represented-family and
benign validation traffic. Unknown-detection rate is never a selection objective,
because no true-unknown validation rows are available to the selector.

The historical repository also contains a separate active 1% rejection configuration.
That file is not allowed to silently redefine this JISA experiment. Phase 3 uses an
explicit new JISA config/output namespace and records 3% as primary, with 1/5/10%
reported as sensitivity surfaces.

## Rejector methods

The planned comparison retains the manuscript rejector families where their exact
score definitions can be reproduced without hidden test/unknown tuning:

- maximum confidence;
- margin;
- entropy;
- family-conditional/conformal-inspired rejection.

The family-conditional method is descriptive/conformal-inspired unless the exact
conditions for a formal conformal guarantee are satisfied; no such guarantee is
assumed by this specification.

Before implementation, the existing open-set baseline code must be audited for the
exact order of calibration, Stage-1 gating, score computation, feasibility filtering,
and threshold selection. Any step that currently consumes held-out-family validation
rows must receive a blind-specific implementation rather than being reused unchanged.

## Output boundary

All Phase-3 outputs must be local under:

`outputs/13_jisa_q1_revision/phase3_unknown_blind/`

Row-level validation/test scores may be written there only when needed for the
predeclared rejector/statistical analyses. They are local regeneration artifacts and
must remain git-ignored. Tracked evidence must contain only compact aggregate
summaries, hashes/provenance, code, and specifications.

Historical Protocol-B/open-set outputs are read-only.

## Required reporting

At minimum, report per seed × holdout × budget × rejector:

- unknown detection rate on test;
- false-unknown rate among all known test traffic;
- false-unknown rate among known attacks;
- overall rejection rate;
- benign rejection rate;
- benign-to-family FPR;
- supported-label system macro-F1;
- unknown-vs-known AUROC/AUPR where the score definition is threshold-independent;
- selected Stage-1 threshold;
- selected rejector threshold/policy;
- feasibility status.

Primary manuscript comparisons use the 3% budget. The 1/5/10% surfaces are
sensitivity evidence and must not be averaged with the primary result.

## Visible-vs-blind comparison

The comparison unit is matched by dataset, holdout family, seed, rejector, and
budget. Report blind minus validation-visible differences rather than comparing
unmatched feasible subsets. Coverage/feasibility is itself an outcome and must be
reported separately.

## Preflight gate

Before implementing or training the blind lane, run
`scripts/jisa_phase3_preflight.py` against the local five-seed Protocol-B artifacts.
The preflight must establish:

- all five seeds exist;
- all six CICIoT2023 holdouts are present per seed;
- strict Stage-1 LOAO is active;
- the candidate profile matrix is complete;
- whether historical winners are uniquely determined by heldout-free Stage-2
  validation macro-F1 alone;
- the prepared CICIoT2023 Protocol-B directory referenced by local manifests exists.

If unique-primary profile selection fails for any seed/holdout, do not freeze the
historical winner into the blind lane. Expand the blind rerun to all candidate
profiles for the affected cases.
