# JISA Phase 1 Pilot Audit

Status: CORRECTED CANONICAL PILOT ACCEPTED
Date: 2026-09-07
Pilot item: CICIDS2017 / RF / seed 123

## Initial pilot

The first controlled-direct pilot completed successfully and confirmed that the
prepared Protocol-A data, preprocessing, direct classifier training, validation-only
thresholding, and test evaluation all execute end to end.

The initial runner selected the most-permissive exact benign-score boundary satisfying
the 1.5% validation benign-FPR ceiling. Because the frozen Phase-1 specification
explicitly described a family-aware validation sweep, that initial execution was
marked diagnostic rather than claim-bearing and was rerun with the corrected
canonical selector before any batch execution.

## Corrected canonical pilot

The canonical rerun used
`scripts/jisa_phase1_run_controlled_direct_familyaware.py` and selected the same
threshold and produced the same prediction metrics as the initial pilot:

- selected validation threshold: 0.00333333333333
- validation benign FPR: 0.011853
- validation attack recall: 0.999498
- ordinary argmax test macro-F1: 0.917574
- ordinary argmax test accuracy: 0.993746
- ordinary argmax benign-to-family FPR: 0.000202
- constraint-matched test macro-F1: 0.813414
- constraint-matched test accuracy: 0.988835
- constraint-matched benign-to-family FPR: 0.012826
- constraint-matched attack-to-benign rate: 0.000321

The corrected result is the canonical Phase-1 result for CICIDS2017 / RF / seed 123.

## Why the threshold remained identical

For one fixed classifier, gate recall for every attack family is monotone as the
attack threshold is relaxed. Therefore, among thresholds satisfying the same benign
FPR ceiling, maximizing minimum eligible-family gate recall and then total attack
recall selects the most permissive feasible threshold, except where tied/quantized
scores require choosing the next representable boundary. The explicit family-aware
sweep is retained because it matches the frozen experimental specification and
records family-level operating behavior, but the identical pilot/canonical threshold
is expected rather than evidence that the corrected selector failed to run.

For the RF model, the 300-tree probability grid creates tied/quantized score values;
consequently the closest feasible validation operating point realizes 1.1853% benign
FPR rather than exactly 1.5%.

## Interpretation boundary

The secondary direct surface is best described as **validation-constraint matched**
rather than as having identical realized test FPR. Both direct and two-stage systems
select their operating point under the same validation benign-FPR ceiling, while the
realized test FPR is reported rather than forced to match after observing test labels.
This preserves test-set isolation.

## Gate

The RF pilot is accepted. The next execution gate is one XGBoost pilot on the same
CICIDS2017 split to verify the CUDA/XGBoost path and probability-based gate before
batching remaining seeds and datasets.
