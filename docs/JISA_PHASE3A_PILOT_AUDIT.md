# JISA Phase 3A Pilot Audit

Status: BRUTEFORCE UNIQUE-PROFILE PATH ACCEPTED; BOTNET TIE-PATH PENDING
Date: 2026-09-08
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

## Pilot verdict

The unique-profile path is accepted. The next required pilot is CICIoT2023 `Botnet`, seed 123, which must exercise the two-candidate RF tie path and prove that:

1. both `rf_class_weight_balanced` and `rf_inv_family_clipped` are evaluated;
2. their Stage-2 definition is shared/reused;
3. profile selection is resolved by Stage-1 AUROC on heldout-free validation traffic only;
4. the Stage-1 threshold is selected only after the blind profile choice is frozen;
5. no held-out validation or test information enters selection.

The full 30-case Phase-3A matrix remains gated until the Botnet tie-path pilot passes.
