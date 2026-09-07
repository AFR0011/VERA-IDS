# JISA Phase 1 Controlled Direct Comparison — Execution Specification

Status: FROZEN BEFORE FIRST TEST EXECUTION
Date: 2026-09-07
Parent specification: `docs/JISA_PHASE1_SPEC.md`
Runner: `scripts/jisa_phase1_run_controlled_direct.py`
Config: `config/jisa_phase1_controlled_direct.yml`

## Fixed data surface

- Prepared root: `processed_V5/A_stratified`
- Datasets: CICIDS2017, CICIoT2023
- Seeds: 123, 124, 125, 126, 127
- No repartitioning or preprocessing changes are permitted.
- Preprocessing is fitted on training data only.
- Training-row caps match the established Protocol-A Stage-1 sampling budget:
  - CICIDS2017: 900,000
  - CICIoT2023: 1,500,000
- Validation caps match the established Stage-1 validation budget:
  - CICIDS2017: 500,000 (the prepared validation split has only 315,174 rows, so all are used)
  - CICIoT2023: 650,000
- The full prepared test split is evaluated.

The row caps are sampling budgets inside the already frozen split. They do not define new train/validation/test partitions.

## Fixed model profiles

### RF direct

- 300 trees
- unrestricted depth
- minimum leaf size 1
- `sqrt` feature sampling
- balanced-subsample class weighting
- seed-specific `random_state`

### XGB direct

- 800 estimators
- depth 6
- learning rate 0.05
- subsample 0.8
- column subsample 0.8
- L2 regularization 1.0
- multiclass soft probabilities
- histogram tree method on CUDA
- balanced training sample weights
- seed-specific `random_state`

These are fixed established VERA-IDS-family settings. The comparison is model-family matched, not a claim that a single multiclass model can have hyperparameters literally identical to both stages of a two-model cascade.

## Two reporting surfaces from the same fitted direct model

1. `argmax`: ordinary closed-set multiclass argmax.
2. `fpr_matched`: validation-selected benign/attack gate with `p_attack = 1 - p(Benign)`.

The primary validation benign-FPR ceiling is 0.015. The selected threshold is the lowest empirical threshold that satisfies that ceiling, with deterministic tie handling. Because lowering this scalar gate threshold monotonically increases every attack-family gate recall, the lowest feasible threshold is also the threshold that maximizes the predeclared minimum-family gate-recall objective under the FPR constraint.

Test labels are never inspected during threshold selection.

## Metrics

For validation and test, save aggregate only:

- supported-label system macro-F1;
- accuracy;
- weighted F1;
- binary attack F1;
- benign-to-family false-positive rate;
- attack-to-benign rate;
- per-family classification report;
- confusion matrix;
- family-level gate recall for the FPR-matched surface.

No row-level predictions or probabilities are persisted by this runner.

## Output boundary

All heavy/local outputs are written under:

`outputs/13_jisa_q1_revision/phase1_controlled_direct/`

Historical Protocol-A, Protocol-B, seed-reliability and manuscript-finalization outputs are read-only and are not overwritten.

## Execution order

1. Compile and dry-run the runner.
2. Pilot CICIDS2017 RF seed 123.
3. Audit the pilot output before any matrix run.
4. Run remaining RF/XGB × dataset × seed items only after the pilot gate passes.
