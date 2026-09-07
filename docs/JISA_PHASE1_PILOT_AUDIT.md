# JISA Phase 1 Pilot Audit

Status: RF/XGB CICIIDS2017 PILOTS AND CICIOT2023 SCALE PILOT ACCEPTED; BATCH EXECUTION OPEN
Date: 2026-09-07

## Initial RF pilot

The first controlled-direct pilot completed successfully and confirmed that the
prepared Protocol-A data, preprocessing, direct classifier training, validation-only
thresholding, and test evaluation all execute end to end.

The initial runner selected the most-permissive exact benign-score boundary satisfying
the 1.5% validation benign-FPR ceiling. Because the frozen Phase-1 specification
explicitly described a family-aware validation sweep, that initial execution was
marked diagnostic rather than claim-bearing and was rerun with the corrected
canonical selector before any batch execution.

## Corrected canonical RF pilot

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

## Why the RF threshold remained identical

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

## Canonical XGBoost pilot

CICIDS2017 / XGB / seed 123 completed successfully under the same canonical
family-aware runner:

- fit time: 138.7 s
- selected validation threshold: 0.000198852107862
- validation benign FPR: 0.012822
- validation attack recall: 0.999576
- ordinary argmax test macro-F1: 0.914292
- ordinary argmax test accuracy: 0.996646
- ordinary argmax benign-to-family FPR: 0.000324
- constraint-matched test macro-F1: 0.813588
- constraint-matched test accuracy: 0.989469
- constraint-matched benign-to-family FPR: 0.012110
- constraint-matched attack-to-benign rate: 0.000502

The XGBoost runtime emitted a device-mismatch prediction warning because the fitted
booster is on CUDA while prediction input remains CPU-backed sparse data. XGBoost
fell back to DMatrix prediction. This is a performance/memory warning rather than a
selection or metric-validity failure; the run completed normally and the warning did
not alter the frozen estimator configuration or test-isolation rules.

The CICIDS2017 / XGB / seed 123 result is accepted as canonical Phase-1 evidence.

## CICIoT2023 scale pilot

CICIoT2023 / RF / seed 123 completed successfully at the full frozen Phase-1 scale:
1,500,000 sampled training rows, 650,000 validation rows, and all 911,053 test rows.

- fit time: 1888.7 s
- selected validation threshold: 0.86
- validation benign FPR: 0.014887
- validation attack recall: 0.975541
- ordinary argmax test macro-F1: 0.917892
- ordinary argmax test accuracy: 0.979920
- ordinary argmax benign-to-family FPR: 0.100606
- constraint-matched test macro-F1: 0.882629
- constraint-matched test accuracy: 0.969551
- constraint-matched benign-to-family FPR: 0.016857
- constraint-matched attack-to-benign rate: 0.024639

The realized test benign FPR of 1.6857% is above the 1.5% validation constraint. This
is not a protocol violation because the operating threshold was selected on validation
only and the test set is evaluation-only. The difference is reported as held-out
operating-point drift rather than corrected after observing test labels.

The CICIoT2023 result also illustrates why ordinary argmax and validation-constraint-
matched surfaces must remain separate. Ordinary argmax achieves stronger macro-F1 but
allows a 10.06% benign-to-family false-positive rate, whereas the predeclared validation
constraint reduces that test false-positive rate to 1.69% at the cost of lower macro-F1
and higher attack-to-benign error. No superiority claim between direct and two-stage
architectures is made from this pilot alone; the matched comparison requires the full
seed matrix and corresponding two-stage evidence.

The CICIoT2023 / RF / seed 123 scale pilot is accepted as canonical Phase-1 evidence.

## Interpretation boundary

The secondary direct surface is best described as **validation-constraint matched**
rather than as having identical realized test FPR. Both direct and two-stage systems
select their operating point under the same validation benign-FPR ceiling, while the
realized test FPR is reported rather than forced to match after observing test labels.
This preserves test-set isolation.

The macro-F1 movement from ordinary argmax to the validation-constraint-matched
surface should not be interpreted as a model defect. The gate deliberately changes
the class-error trade-off under a fixed validation operating constraint. Depending on
the dataset, this may increase or decrease realized benign false positives relative to
ordinary multiclass argmax because the two decision rules are different.

## Gate

The CICIDS2017 RF and XGB seed-123 pilots and the CICIoT2023 RF seed-123 scale pilot
are accepted. Phase-1 batch execution is now open for the remaining dataset/model/seed
items. Execution should remain sequential and resumable; completed items are skipped
by the canonical runner unless `--force` is supplied. Do not parallelize RF and XGB
items on the same workstation because CICIoT2023 RF already exercises substantial CPU
and memory resources, while XGBoost uses the GPU and CPU-backed prediction matrices.
