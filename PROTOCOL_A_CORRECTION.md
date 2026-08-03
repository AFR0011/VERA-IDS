# Protocol A Macro-F1 Correction Supplement

Release: `v2026.08`
Copyright © 2026 Ali Farrokhnejad
License: CC BY 4.0

This repository supplement corrects the closed-set Protocol A system macro-F1
denominator. It does not alter the thesis body. The thesis remains a historical
record of the value produced by the declared-output-label calculation.

## Correction

Protocol A has supported ground-truth labels `Benign` plus the attack families.
Its abstention policy can also emit `Unknown`, but no Protocol A test row has a
true `Unknown` label. The historical calculation averaged per-class F1 over the
declared output vocabulary, including that unsupported label. Because its F1 is
zero, the extra term lowered the macro average.

The primary metric is now:

`system_macro_f1_supported_labels = mean(F1(label) for label in supported_labels)`

Each class F1 is still computed from the complete confusion matrix. Therefore a
supported example predicted as `Unknown` remains a false negative for its true
class; only the unsupported `Unknown` class is excluded from the final average.

The former value is retained without reinterpretation as
`system_macro_f1_declared_output_labels_historical`. Protocol B is unchanged:
valid LOAO folds contain genuine true-`Unknown` support, so `Unknown` remains in
their supported averaged label set.

## Corrected primary results

| Dataset | Model | Policy | Supported-label macro-F1 | Historical declared-output macro-F1 |
| --- | --- | --- | ---: | ---: |
| CICIDS2017 | RF | strict | 0.818902771736530 | 0.716539925269464 |
| CICIDS2017 | RF | strict_tau | 0.850374216458442 | 0.744077439401136 |
| CICIDS2017 | XGB | strict | 0.699910088488132 | 0.612421327427115 |
| CICIDS2017 | XGB | strict_tau | 0.731398318262722 | 0.639973528479881 |
| CICIoT2023 | RF | strict | 0.897154513910465 | 0.785010199671656 |
| CICIoT2023 | RF | strict_tau | 0.897142171999474 | 0.784999400499540 |
| CICIoT2023 | XGB | strict | 0.891279663605775 | 0.779869705655053 |
| CICIoT2023 | XGB | strict_tau | 0.888809454169177 | 0.777708272398030 |

External Protocol A, five-seed summaries, reference-profile comparisons,
flat-versus-two-stage tables, and figures in this release were regenerated from
the same corrected field.

## Evidence and provenance

The path-free evidence table contains 136 matrices: 34 selected runs × four test
policies (`strict`, `cascade`, `strict_tau`, and `cascade_tau`). Each record
contains ordered labels, integer counts, surface metadata, the original CSV
SHA-256, and a stable evidence ID. It contains no filesystem paths.

- Evidence: `outputs/evidence/protocol_a_confusion_matrices.jsonl`
- Evidence SHA-256: `1c34102c3bdc593896e8c20ed1579631ac44b57ae3e7fffa46dc3f5c249b17b5`
- Deterministic verification:

```bash
python scripts/recompute_protocol_a_metrics.py --check
python scripts/verify_manifests.py
```

Rebuilding the evidence table from protected research artifacts is an internal
release operation. Public users can reproduce every corrected number directly
from the tracked aggregate counts without datasets or model binaries.

## Interpretation

The correction changes the denominator, not predictions, thresholds, seeds,
splits, confusion counts, accuracy, or model selection. Comparisons in this
release use the supported-label value. The historical field exists solely to
reconcile the repository edition with earlier reported numbers.
