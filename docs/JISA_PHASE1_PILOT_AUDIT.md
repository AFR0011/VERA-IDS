# JISA Phase 1 Pilot Audit

Status: PILOT NON-CANONICAL; IMPLEMENTATION CORRECTED BEFORE BATCH EXECUTION
Date: 2026-09-07
Pilot item: CICIDS2017 / RF / seed 123

## Pilot result

The first controlled-direct pilot completed successfully and confirmed that the
prepared Protocol-A data, preprocessing, direct classifier training, validation-only
thresholding, and test evaluation all execute end to end.

Pilot test metrics:

- ordinary argmax macro-F1: 0.917574
- ordinary argmax accuracy: 0.993746
- ordinary argmax benign-to-family FPR: 0.000202
- FPR-gated macro-F1: 0.813414
- FPR-gated accuracy: 0.988835
- FPR-gated benign-to-family FPR: 0.012826
- FPR-gated attack-to-benign rate: 0.000321

The pilot selected threshold 0.00333333333333 with validation benign FPR 0.011853
and validation attack recall 0.999498.

These values are retained only as a diagnostic pilot and are not claim-bearing JISA
results.

## Audit finding

The pilot runner selected the most-permissive exact benign-score boundary satisfying
the 1.5% validation benign-FPR ceiling. For a fixed estimator this is a sensible
operating rule and tends to maximize gate recall, but it did not literally execute
the frozen Phase-1 specification's 200-point family-aware threshold sweep and
lexicographic tie-break.

Because this discrepancy was detected after only one pilot and before any batch
execution or manuscript use, the pilot is marked non-canonical rather than silently
accepted.

## Correction

Canonical Phase-1 execution now uses
`scripts/jisa_phase1_run_controlled_direct_familyaware.py`.

The corrected runner preserves the same training/evaluation implementation but
selects the validation gate by:

1. enforcing benign FPR <= 0.015;
2. maximizing minimum gate recall across validation attack families meeting the
   frozen minimum support;
3. tie-breaking by total validation attack recall;
4. final tie-breaking by the highest threshold;
5. using a 200-point validation score sweep augmented with the exact feasible
   benign-FPR boundary so numerical discretization cannot omit the boundary.

Test labels remain unavailable to selection.

## Required action

Rerun CICIDS2017 / RF / seed 123 with `--force` using the corrected canonical runner.
Only after that rerun passes audit may Phase-1 batch execution proceed.
