# JISA Phase 0 Findings

Status: FROZEN
Date: 2026-09-07
Branch: `jisa-q1-revision`

## Local evidence recovered

The local owner workspace contains sufficient evidence to proceed without treating the public compact repository as the only source of historical run data.

- 42 complete Protocol-B validation/test score pairs with scenario manifests were recovered from local JISA/finalization output surfaces.
- 114 joblib/model-extension artifacts were found; the five-seed Protocol-B directories specifically retain 90 `preprocessor.joblib` files but no saved Stage-1/Stage-2 classifier estimators or row-level probability score files.
- The repeated-seed CICIoT2023 Protocol-B surface contains 5 seeds (123–127), 6 holdouts, and 3 candidate profiles per holdout, for 90 candidate runs.
- Validation-selected winners are stable across all five seeds for every holdout: BruteForce selects `xgb_inv_family_clipped`; Botnet, DDoS, DoS, Other, and Scan/Recon select `rf_class_weight_balanced`.
- Selection is based on validation Stage-2 macro-F1, then validation Stage-1 AUROC, then lexical run name. Test metrics are not used for profile selection.
- The processed parquet rows do not carry source/unit provenance columns. Split reports and source/unit planning metadata remain available separately and may support reconstructed cluster identifiers for later statistical analysis.

## Consequence for the blind Protocol-B experiment

A new blind result cannot be obtained by merely replaying the five-seed selected aggregate outputs, because those seed runs did not preserve the row-level probability scores or trained Stage-1/Stage-2 estimators required to reconstruct alternative operating points.

The blind lane must therefore perform targeted reruns on CICIoT2023. The reruns may reuse the frozen split/support manifests and the already-stable candidate-profile policy, but all selection steps that are allowed to inspect the held-out family in the validation-visible lane must be redefined so held-out-family validation observations are excluded from operating-point selection.

## Scientific boundary for the blind lane

For the new result to be described as blind with respect to the held-out family, held-out-family validation observations must be excluded from every decision that can inspect validation labels or predictions for that family, including:

1. Stage-1 threshold selection;
2. rejector/tau threshold selection;
3. rejector method operating-point selection;
4. any candidate/profile selection rule used for the blind lane.

The held-out family remains absent from Stage-1 and Stage-2 training exactly as in strict LOAO. Test remains evaluation-only.

The existing validation-visible Protocol B is preserved unchanged as a separate evidence surface.

## Consequence for Phase 1

The controlled direct-multiclass comparison remains a new experiment surface. Matched RF and XGB flat classifiers must be trained on the same Protocol-A prepared partitions as the corresponding two-stage models, with both ordinary argmax reporting and a validation-selected benign/attack gate derived from `1 - p(Benign)` for operating-point matching.

## Phase 0 closure

Phase 0 is complete. No historical scientific outputs were changed. Subsequent experiments must write only to a JISA-specific local output namespace and must not overwrite historical public-release or thesis/finalization outputs.
