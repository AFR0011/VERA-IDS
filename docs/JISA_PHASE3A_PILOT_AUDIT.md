# JISA Phase 3A Pilot Audit

Status: BOTH PILOT PATHS ACCEPTED; FULL 30-CASE MATRIX COMPLETED LOCALLY; STRUCTURAL AUDIT PENDING
Date: 2026-09-12
Branch: `jisa-q1-revision`
Parent execution spec: `docs/JISA_PHASE3_EXECUTION_ADDENDUM.md`
Runner: `scripts/jisa_phase3a_generate_validation_blind_scores.py`

## BruteForce seed 123

The first Phase-3A pilot exercised the unique-profile XGB path for CICIoT2023 with `BruteForce` held out.

Observed completion record:

- selected candidate: `xgb_inv_family_clipped`;
- blind-validation Stage-1 AUROC: `0.9976213198709024`;
- selected Stage-1 threshold: `0.9417504815234587`;
- blind validation rows: `698024`;
- hidden held-out validation rows: `1976`;
- loaded training rows: `1500000`;
- full test rows: `1190899`;
- selected Stage-2 blind-validation macro-F1: `0.976999477904962`;
- held-out validation used for selection: `false`;
- test used for selection: `false`;
- preprocessor exposure: `historical_full_train_unsupervised_feature_exposure`;
- fitted Stage-1/Stage-2 models persisted: `false`.

The candidate-profile selection file contains one eligible member of the historical top Stage-2 equivalence set, `xgb_inv_family_clipped`, and marks it selected. The recomputed blind-validation Stage-2 macro-F1 agrees with the historical represented-family Stage-2 value to stored precision, as expected because the historical Stage-2 validation metric was already restricted to represented families.

The XGBoost CUDA/CPU DMatrix warning observed during prediction is an execution/performance warning only and does not change the frozen scientific protocol.

## Botnet seed 123

The second pilot exercised the two-candidate RF tie path for CICIoT2023 with `Botnet` held out.

Observed completion record:

- historical top-equivalence set: `rf_class_weight_balanced`, `rf_inv_family_clipped`;
- both candidates recomputed the same blind-validation Stage-2 macro-F1: `0.9219984587949888`;
- `rf_class_weight_balanced` blind-validation Stage-1 AUROC: `0.9969831286430979`;
- `rf_inv_family_clipped` blind-validation Stage-1 AUROC: `0.9965140597387833`;
- selected candidate: `rf_class_weight_balanced`;
- selected Stage-1 threshold: `0.9166666666666666`;
- blind validation rows: `599278`;
- hidden held-out validation rows: `100722`;
- loaded training rows: `1500000`;
- full test rows: `1190899`;
- held-out validation used for selection: `false`;
- test used for selection: `false`;
- preprocessor exposure: `historical_full_train_unsupervised_feature_exposure`.

The two RF candidates share the same Stage-2 definition, and the runner caches Stage-2 models by Stage-2 signature. Their identical stored Stage-2 blind-validation macro-F1 is consistent with that shared definition. The candidate tie is resolved by Stage-1 AUROC computed only on the heldout-free validation surface; `rf_class_weight_balanced` wins by approximately `0.00046907` AUROC. The Stage-1 threshold is then selected after the profile choice is frozen.

## Full matrix completion checkpoint

On 2026-09-12 the local Phase-3A score-generation summary reported `completed cases: 30/30` for all five seeds and all six CICIoT2023 holdouts. The summary showed a stable selected profile by holdout across all five seeds:

- `Botnet`: `rf_class_weight_balanced`;
- `BruteForce`: `xgb_inv_family_clipped`;
- `DDoS`: `rf_class_weight_balanced`;
- `DoS`: `rf_class_weight_balanced`;
- `Other`: `rf_class_weight_balanced`;
- `Scan/Recon`: `rf_class_weight_balanced`.

All 30 cases reported the full test size `1190899`. The summary is an execution checkpoint, not yet the frozen Phase-3A evidence surface. A separate structural/deep audit must verify the case artifacts, selection rule, held-out validation exclusion, score schemas, test-row support, and model-persistence boundary before Phase 3B begins.

## Current gate

Both required pilot paths are accepted and the full 30-case matrix has completed locally. Phase 3B remains blocked until `scripts/jisa_phase3a_audit_results.py --deep` passes all 30 cases.
