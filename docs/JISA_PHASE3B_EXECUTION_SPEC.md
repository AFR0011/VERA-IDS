# JISA Phase 3B — Heldout-Validation-Blind Rejector Replay

Status: FROZEN BEFORE FIRST PHASE-3B REJECTOR RESULT
Date: 2026-09-12
Branch: `jisa-q1-revision`
Parent specification: `docs/JISA_PHASE3_EXECUTION_ADDENDUM.md`

## Input surface

Phase 3B consumes only the frozen Phase-3A score artifacts under
`outputs/13_jisa_q1_revision/phase3_validation_blind/score_generation/`.
No model retraining is performed. The selected Stage-1 threshold stored by Phase 3A
is fixed before rejector selection.

The Phase-3A deep structural audit passed all 30 CICIoT2023 seed x holdout cases.
Those score files are therefore treated as read-only evidence inputs.

## Visible-comparison availability

The Phase-3B preflight found 42 historical visible-score candidate directories but
zero artifacts that could be paired unambiguously by seed x holdout with the complete
new five-seed surface. Therefore the primary Phase-3B execution proceeds as a
five-seed heldout-validation-blind robustness analysis.

No manuscript claim may describe a five-seed paired visible-vs-blind rejector
comparison unless an explicitly matched visible surface is regenerated later.
Historical validation-visible results may still be shown as separately labelled
contextual evidence, but not as paired differences.

## Methods

Four rejector families are replayed after the frozen Stage-1 attack gate:

1. maximum confidence;
2. top-two margin;
3. normalized entropy;
4. family-conditional / conformal-inspired confidence thresholding.

The family-conditional method remains descriptive/conformal-inspired. No formal
coverage guarantee is claimed.

## Validation-only selection

Held-out-family validation rows are absent from the Phase-3A validation score files.
Rejector operating points are therefore selected only from benign plus represented-
family validation traffic.

Nominal overall rejection budgets are 1%, 3%, 5%, and 10%, with 3% primary.
Every selected point must also satisfy:

- benign-to-family FPR <= 0.02;
- benign rejection rate <= 0.10;
- false-unknown rate among all known validation traffic <= 0.05;
- false-unknown rate among represented validation attacks <= 0.10.

Because all Phase-3A validation rows are known traffic, the 5% all-known
false-unknown constraint can make the 10% nominal rejection budget effectively no
larger than 5%. This is intentional and must not be relaxed after observing test
results.

Selection objective for every method and budget:

1. maximize validation rejection rate subject to all constraints and the nominal
   budget;
2. break exact rejection-rate ties by higher supported-label validation macro-F1;
3. then lower false-unknown rate among all known traffic;
4. then deterministic parameter order.

Unknown-detection rate is never a validation objective.

## Exact empirical operating points for global rejectors

For maximum-confidence, margin, and entropy rejection, Phase 3B does not use an
arbitrary coarse threshold grid. It sorts the attack-gated validation observations by
method-specific uncertainty and evaluates every distinct score boundary, including
no rejection. Tied uncertainty scores are never split: all observations at a selected
boundary are rejected together.

Uncertainty definitions:

- maximum confidence: `1 - p_max`;
- margin: `1 - (p_(1) - p_(2))`;
- entropy: normalized Stage-2 entropy.

A selected empirical boundary uses an inclusive rule (`uncertainty >= boundary`) so
that the complete tied score group is reproduced exactly on validation and test.
This yields the most selective feasible empirical validation point under the frozen
constraints without introducing a post-hoc threshold grid.

## Family-conditional operating points

For each represented true family, the calibration threshold is the empirical
quantile of the validation probability assigned to that true family. Candidate alpha
values are frozen at 0.000, 0.005, ..., 0.100. At inference, an attack-gated row is
rejected when its maximum Stage-2 probability is strictly below the threshold learned
for its predicted family.

Family-conditional alpha is selected by the same validation-only feasibility and
selection rule as the global methods.

## Metrics

For every seed x holdout x method x budget, report at least:

- feasibility status;
- validation rejection rate and constraint metrics at the selected point;
- test unknown-detection rate;
- test false-unknown rate among all known traffic;
- test false-unknown rate among known attacks;
- test overall rejection rate;
- test benign rejection rate;
- test benign-to-family FPR;
- test supported-label macro-F1;
- test accuracy;
- selected Stage-1 threshold;
- selected rejector parameter/policy.

For maximum confidence, margin, and entropy, unknown-vs-known AUROC and AUPR are
reported from their threshold-independent uncertainty scores on the full test set.
They are not reported for the family-conditional rule because its uncertainty surface
depends on alpha-specific family thresholds.

Supported-label macro-F1 is computed over labels with nonzero true support in the
split. Thus validation excludes `Unknown` because the blind validation surface has no
true Unknown observations, while test includes `Unknown` when it has support.

## Output boundary

Phase-3B outputs are local under:

`outputs/13_jisa_q1_revision/phase3_validation_blind/rejector_replay/`

Each case stores compact selected-point results only. Phase-3A row-level score files
remain the immutable local evidence source. No additional row-level predictions are
written by Phase 3B.

## Pilot gate

Before opening the complete 30-case replay:

1. run BruteForce seed 123 to exercise the XGB-derived score surface;
2. run Botnet seed 123 to exercise the RF-derived score surface;
3. verify all four budgets and all four rejectors are represented;
4. inspect feasibility and ensure validation constraints are obeyed exactly;
5. only then open the resumable 30-case replay.
