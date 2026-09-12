# JISA Phase 3A Results Frozen — Heldout-Validation-Blind Score Generation

Status: FROZEN AFTER DEEP AUDIT PASS
Date: 2026-09-12
Branch: `jisa-q1-revision`
Parent execution specification: `docs/JISA_PHASE3_EXECUTION_ADDENDUM.md`
Runner: `scripts/jisa_phase3a_generate_validation_blind_scores.py`
Audit: `scripts/jisa_phase3a_audit_results.py --deep`

## Frozen completion state

The complete CICIoT2023 Phase-3A matrix contains 30/30 seed × holdout cases:

- seeds: 123, 124, 125, 126, 127;
- holdouts: Botnet, BruteForce, DDoS, DoS, Other, Scan/Recon;
- full test rows per case: 1,190,899;
- no held-out-family validation observations enter profile or Stage-1 threshold selection;
- no test observation enters selection;
- historical full-training-partition unsupervised preprocessing exposure is preserved deliberately;
- row-level validation/test score files remain local regeneration artifacts;
- fitted Stage-1 and Stage-2 classifiers are not persisted.

The deep structural audit passed. Phase 3A is therefore frozen and must not be rerun or modified merely to improve downstream rejector results. Any later change to preprocessing, profile selection, Stage-1 threshold policy, row caps, split definitions, or score construction constitutes a new experiment surface.

## Selected profile stability

The selected profile is stable across all five seeds for every holdout:

- Botnet: `rf_class_weight_balanced`;
- BruteForce: `xgb_inv_family_clipped`;
- DDoS: `rf_class_weight_balanced`;
- DoS: `rf_class_weight_balanced`;
- Other: `rf_class_weight_balanced`;
- Scan/Recon: `rf_class_weight_balanced`.

The five non-BruteForce holdouts were resolved from an exact two-way Stage-2-validation tie between the two RF Stage-1 weighting variants using heldout-free Stage-1 AUROC. BruteForce had a unique Stage-2 primary winner.

## Frozen Stage-1 threshold ranges across seeds

Observed selected thresholds are narrow within each holdout and differ materially across holdouts:

- Botnet: approximately 0.916667–0.920000;
- BruteForce: approximately 0.940971–0.943207;
- DDoS: approximately 0.913333–0.916667;
- DoS: approximately 0.916667–0.920000;
- Other: approximately 0.816667–0.830000;
- Scan/Recon: approximately 0.866667–0.870000.

These values are descriptive outputs of the predeclared heldout-free Stage-1 validation search, not post hoc thresholds.

## Phase 3B boundary

Phase 3B may consume the frozen `val_known_scores.csv.gz`, `test_scores.csv.gz`, and selected Stage-1 thresholds. It may not retrain or alter Phase-3A models.

Before rejector replay, a dedicated Phase-3B preflight must determine whether matching five-seed validation-visible score pairs already exist locally. The manuscript comparison specification requires visible-vs-blind comparisons to be matched by seed, holdout, rejector, and budget wherever such a paired surface is claimed. If the necessary validation-visible scores do not exist, the missing comparison must not be fabricated from unmatched historical summaries.
