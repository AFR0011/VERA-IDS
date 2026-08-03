# Results

All values are compact aggregate evidence. No row-level predictions are included.

## Protocol A primary results

The primary system macro-F1 averages supported labels. The historical declared-
output-label value is retained in an explicit separate column.

| Dataset | Model | Policy | Supported-label macro-F1 | Accuracy |
| --- | --- | --- | ---: | ---: |
| CICIDS2017 | RF | strict | 0.818902771736530 | 0.986258384257585 |
| CICIDS2017 | RF | strict_tau | 0.850374216458442 | 0.985119330909276 |
| CICIDS2017 | XGB | strict | 0.699910088488132 | 0.958819572680488 |
| CICIDS2017 | XGB | strict_tau | 0.731398318262722 | 0.958089817053437 |
| CICIoT2023 | RF | strict | 0.897154513910465 | 0.972182737996582 |
| CICIoT2023 | RF | strict_tau | 0.897142171999474 | 0.972167371162819 |
| CICIoT2023 | XGB | strict | 0.891279663605775 | 0.970414454482890 |
| CICIoT2023 | XGB | strict_tau | 0.888809454169177 | 0.964150274462627 |

Authoritative table: `outputs/summaries/protocol_a_core_summary.csv`. Definitions,
historical values, and source hashes are in
[PROTOCOL_A_CORRECTION.md](PROTOCOL_A_CORRECTION.md).

## Protocol B

Valid LOAO folds have genuine true-`Unknown` support. Their macro-F1 therefore
includes `Unknown`, alongside `Benign` and the known attack families. Results are
reported only for support-eligible folds in
`outputs/summaries/protocol_b_best_per_holdout.csv`.

The recorded UNSW-NB15 Protocol B setup is an invalid diagnostic because benign
validation/test support is zero. It is excluded from rankings, comparisons,
figures, and positive claims.

## External Protocol A

NSL-KDD supported-label macro-F1 spans 0.5791–0.6657 across the selected RF/XGB
strict and strict_tau policies. UNSW-NB15 spans 0.5338–0.5386. These are external
stress-test results, not open-set claims.

## Repeated seeds and reference profiles

Five-seed summaries now contain both
`system_macro_f1_supported_labels` and
`system_macro_f1_declared_output_labels_historical` rows. Reference-profile
comparisons use the supported-label Protocol A value and preserve the historical
field for reconciliation. Reference profiles are descriptive parameter profiles,
not byte-identical reimplementations of third-party studies.

## Interpretation limits

Cross-surface metric values are comparable only when their task, support, label
set, split, and selection rules match. Historical models are unavailable, and no
downloadable-model reproduction claim is made.
