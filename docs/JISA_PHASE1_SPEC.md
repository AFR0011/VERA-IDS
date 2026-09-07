# JISA Phase 1 Experiment Specification

Status: FROZEN BEFORE TEST EXECUTION
Date: 2026-09-07
Experiment: controlled direct-multiclass comparison
Primary purpose: establish that VERA-IDS conclusions are observed under conventionally strong models and quantify architecture/operating-point effects under matched data and validation discipline.

## Datasets and partitions

Use the existing Protocol-A `A_stratified` prepared partitions for CICIDS2017 and CICIoT2023. Do not regenerate or repartition the datasets for this experiment.

Seeds: 123, 124, 125, 126, 127.

## Comparator families

Train matched direct-multiclass classifiers using:

- Random Forest (RF)
- XGBoost (XGB)

These are matched to the two-stage model families used by VERA-IDS. The existing broader competitive direct baseline remains a separate literature-competitiveness surface and is not merged with this controlled comparison.

## Hyperparameter policy

The direct RF/XGB models must use a predeclared, fixed model-family configuration derived from the corresponding established VERA-IDS model settings rather than selecting among a broader model zoo on test performance.

The same feature preparation and train-only preprocessing rules used by Protocol A apply.

## Two direct operating surfaces

For each trained direct model, report both surfaces from the same fitted estimator.

### A. Conventional direct multiclass

Prediction is ordinary multiclass argmax across `Benign` and the supported attack-family labels.

This surface represents conventional closed-set IDS reporting.

### B. FPR-matched direct multiclass

Define the direct attack score as:

`p_attack_direct = 1 - p(Benign)`

Select an attack/benign threshold on validation only using the same target benign-FPR principle as the corresponding two-stage Stage-1 gate. The primary target is 0.015. Among feasible thresholds, prefer the threshold that maximizes the predeclared family-recall objective; ties use total attack TPR and then a deterministic threshold tie-break.

At test time:

- if `p_attack_direct < threshold`, predict `Benign`;
- otherwise predict the highest-probability attack family after excluding the Benign class from the argmax.

No test labels may alter the threshold.

## Metrics

For both direct surfaces report at minimum:

- system macro-F1 over supported true labels;
- accuracy;
- weighted F1;
- binary attack F1 induced from system predictions;
- benign-to-family false-positive rate;
- per-family F1;
- selected validation threshold for the FPR-matched surface;
- realized validation and test benign FPR;
- realized validation and test attack recall;
- family recall distribution on validation/test where support permits.

## Comparisons

The main controlled comparison is family-matched:

- RF direct vs RF two-stage;
- XGB direct vs XGB two-stage.

Report ordinary direct argmax and FPR-matched direct separately. Do not make causal architecture claims from the broader validation-selected ensemble/model-zoo winner.

The broader competitive direct baseline is retained only as a distinct evidence layer showing that the benchmark surface supports performance comparable with strong conventional IDS classifiers.

## Repetition and uncertainty

Repeat the matched direct models for seeds 123–127. Summarize mean, sample SD, min, and max by dataset/model/surface. Seed variation is model-fitting variation conditional on the fixed prepared Protocol-A split; it is not a data-resampling confidence interval.

## Output namespace

Write heavy local outputs only under:

`outputs/13_jisa_q1_revision/phase1_controlled_direct/`

No historical output path may be overwritten.

Tracked compact summaries may be created only after the local results have been audited and accepted.

## Stop conditions

Stop before test evaluation if any of the following occurs:

- the direct and two-stage lanes are found to use different prepared Protocol-A partitions;
- feature/label leakage differs between the two lanes;
- threshold selection inspects test labels;
- label taxonomies do not align;
- a requested model setting cannot be reconstructed unambiguously from the repository.
