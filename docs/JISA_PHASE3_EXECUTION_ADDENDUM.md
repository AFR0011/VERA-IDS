# JISA Phase 3 Execution Addendum — Validation-Blind Score Generation

Status: FROZEN BEFORE FIRST PHASE-3 MODEL EXECUTION  
Date: 2026-09-08  
Branch: `jisa-q1-revision`  
Parent specification: `docs/JISA_PHASE3_UNKNOWN_BLIND_SPEC.md`

## Why this addendum exists

The Phase-3 preflight and tie audit established the exact historical candidate-profile
structure before any blind result was generated:

- five seeds (123–127), six CICIoT2023 holdouts, three historical candidate profiles
  per seed/holdout;
- 30/30 candidate groups complete and strict Stage-1 LOAO;
- only the five BruteForce groups have a unique Stage-2-validation-macro-F1 winner;
- the remaining 25 groups contain an exact two-way tie between
  `rf_class_weight_balanced` and `rf_inv_family_clipped`;
- the third profile is strictly lower on the blind-safe primary Stage-2 criterion in
  every tied group;
- the two tied RF candidates share the same Stage-2 model definition.

Therefore the Phase-3 execution surface requires 55 Stage-1 candidate fits but only
30 Stage-2 fits. Non-top historical profiles are not rerun merely to resolve a
Stage-1 tie that they cannot win under the frozen lexicographic selection rule.

## Important preprocessing finding

A source audit of the historical Protocol-B runner found that the shared
`fit_preprocessor` routine is called before Stage-1/Stage-2 LOAO filtering. The
preprocessor computes numeric statistics and categorical maps from the full Protocol-B
training partition. Thus held-out-family feature rows can contribute to unsupervised
preprocessing statistics even though the held-out family is subsequently removed
from Stage-1 and Stage-2 model fitting.

This matters for terminology. The primary Phase-3 experiment is therefore described
as **heldout-validation-blind selection**, not as fully inductive unknown-family
training. The primary lane deliberately preserves the historical preprocessing
surface so that the causal comparison against the recorded validation-visible
Protocol B changes validation visibility rather than changing both preprocessing and
selection at once.

A stricter preprocessing-blind sensitivity lane may be added separately. It must not
be merged with the primary validation-visibility contrast.

## Phase 3A — score generation and profile selection

Each seed × holdout case is executed as follows.

1. Reuse the frozen Protocol-B train/validation/test partition and historical row
   caps: 1,500,000 training rows, 700,000 validation rows, full test.
2. Reproduce the historical unsupervised preprocessor fitting behavior on the full
   training partition. Record this exposure explicitly in every case manifest.
3. Restrict model fitting to strict LOAO:
   - Stage 1 removes held-out-family attacks from training;
   - Stage 2 trains only represented attack families.
4. Remove all held-out-family rows from validation before any selection metric is
   computed.
5. Use historical represented-family Stage-2 validation macro-F1 as the primary
   candidate criterion. Only the historical top-equivalence set is eligible.
6. Where the Stage-2 criterion ties, resolve the tie by Stage-1 AUROC computed only
   on benign plus represented-family validation traffic; then apply lexical candidate
   identity as a deterministic final tie-break.
7. Select the Stage-1 operating threshold on the same heldout-free validation set
   using the native historical Protocol-B family-aware search:
   - target benign-FPR candidates: 0.01 and 0.02;
   - minimum family support: 100;
   - objectives: minimum family recall and p10 family recall;
   - p10 quantile: 0.10;
   - 40 sweep points;
   - tie-break: objective, then total attack TPR.
8. Freeze the selected profile and Stage-1 threshold before scoring the test set.
9. Persist heldout-free validation scores and full test scores locally for Phase 3B
   rejector replay. Held-out validation rows are not needed by the blind selector and
   are not persisted by the primary runner.

No test label or test score may influence profile selection or Stage-1 threshold
selection.

## Phase 3B — rejector replay

Phase 3B will consume the frozen Phase-3A score files without retraining models.
Rejector definitions remain maximum confidence, top-two margin, entropy, and the
family-conditional/conformal-inspired rule. Thresholds will be selected only from
heldout-free validation traffic.

The selection principle is frozen before test replay: for each method and rejection
budget, choose the most selective feasible validation operating point, meaning the
largest validation rejection rate not exceeding the budget and all known-traffic
constraints. Ties are resolved by higher supported-label validation macro-F1, then
lower false-unknown rate among all known traffic, then deterministic parameter order.
Unknown-detection rate is never a validation objective in this lane.

Primary budget: 3%. Sensitivity budgets: 1%, 5%, 10%. Known-traffic constraints:

- benign-to-family FPR <= 0.02;
- benign rejection <= 0.10;
- false-unknown rate among all known traffic <= 0.05;
- false-unknown rate among represented attacks <= 0.10.

The 10% nominal budget may become effectively tighter because the 5% all-known
false-unknown constraint remains active; this is an intended consequence of the
predeclared feasibility rules, not a reason to relax them post hoc.

## Output boundary

Primary Phase-3 outputs are local under:

`outputs/13_jisa_q1_revision/phase3_validation_blind/`

Phase 3A score generation lives under `score_generation/`; Phase 3B aggregate
rejector results will live under `rejector_replay/` and `summary/`.

Row-level score files and preprocessors remain local/ignored regeneration artifacts.
No fitted Stage-1 or Stage-2 classifier is persisted by the Phase-3A runner.
Historical Protocol-B and open-set outputs remain read-only.

## Pilot gate

Before a full 30-case run:

1. dry-run and execute BruteForce seed 123 (unique XGB profile path);
2. audit its selected profile, hidden-validation count, Stage-1 threshold and score
   files;
3. dry-run and execute Botnet seed 123 (two-way RF tie path);
4. verify that the Stage-2 model is shared and the profile tie is resolved using only
   heldout-free Stage-1 AUROC;
5. only then open the resumable full matrix.
