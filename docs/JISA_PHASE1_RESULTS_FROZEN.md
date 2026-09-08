# JISA Phase 1 Controlled Direct Comparison — Frozen Results

Status: FROZEN
Date: 2026-09-08
Branch: `jisa-q1-revision`
Canonical local output root: `outputs/13_jisa_q1_revision/phase1_controlled_direct/`
Canonical runner: `scripts/jisa_phase1_run_controlled_direct_familyaware.py`
Audit runner: `scripts/jisa_phase1_audit_results.py`

## Completion gate

The local Phase-1 audit passed with:

- experiment items: 20/20
- five-seed summary groups: 8/8
- audit status: PASS

Each dataset × model-family × reporting-surface group contains seeds 123–127.
No row-level predictions/probabilities or fitted direct classifiers are part of the
tracked evidence surface.

## Five-seed direct-multiclass results

| Dataset | Model | Surface | n | Macro-F1 mean | SD | Accuracy mean | Benign-family FPR mean |
|---|---|---:|---:|---:|---:|---:|---:|
| CICIDS2017 | RF | argmax | 5 | 0.921358 | 0.003812 | 0.993736 | 0.000203 |
| CICIDS2017 | RF | validation-constraint matched | 5 | 0.827165 | 0.020731 | 0.990519 | 0.010743 |
| CICIDS2017 | XGB | argmax | 5 | 0.922426 | 0.010550 | 0.996879 | 0.000319 |
| CICIDS2017 | XGB | validation-constraint matched | 5 | 0.813275 | 0.005994 | 0.988266 | 0.013975 |
| CICIoT2023 | RF | argmax | 5 | 0.918056 | 0.000169 | 0.979985 | 0.099305 |
| CICIoT2023 | RF | validation-constraint matched | 5 | 0.884390 | 0.001985 | 0.969863 | 0.015768 |
| CICIoT2023 | XGB | argmax | 5 | 0.878210 | 0.001520 | 0.978862 | 0.069141 |
| CICIoT2023 | XGB | validation-constraint matched | 5 | 0.873196 | 0.001865 | 0.973377 | 0.016332 |

The `validation-constraint matched` surface is selected on validation only under the
predeclared benign-FPR ceiling. Realized test FPR is reported, not forced to equal
the validation target.

## Descriptive comparison with genuine five-seed two-stage Protocol-A results

The historical `protocol_a_two_stage` seed-reliability surface contains genuine
five-seed strict two-stage results. The supported-label system macro-F1 means are:

| Dataset | Model | Two-stage strict mean | Direct argmax mean | Direct constrained mean |
|---|---|---:|---:|---:|
| CICIDS2017 | RF | 0.820022 | 0.921358 | 0.827165 |
| CICIDS2017 | XGB | 0.706029 | 0.922426 | 0.813275 |
| CICIoT2023 | RF | 0.897147 | 0.918056 | 0.884390 |
| CICIoT2023 | XGB | 0.891738 | 0.878210 | 0.873196 |

Corresponding descriptive mean differences (direct minus two-stage) are:

- CICIDS2017 RF: +0.101336 under argmax; +0.007143 under the constrained surface.
- CICIDS2017 XGB: +0.216397 under argmax; +0.107246 under the constrained surface.
- CICIoT2023 RF: +0.020909 under argmax; -0.012757 under the constrained surface.
- CICIoT2023 XGB: -0.013528 under argmax; -0.018542 under the constrained surface.

These are descriptive differences of five-seed means, not the final paired
inferential analysis. Formal paired seed-level/cluster-level inference, if retained,
belongs to the later statistical-refresh phase.

## Scientific interpretation boundary

The result supports the article's evaluation-methodology claim: architecture ranking
is conditional on both dataset and operating policy. A direct multiclass system can
appear substantially superior under ordinary closed-set argmax reporting, while the
advantage can shrink, disappear, or reverse after imposing a common validation-side
benign-FPR operating constraint.

The comparison is model-family controlled, not a pure causal architecture ablation.
The direct RF/XGB estimators use fixed predeclared VERA-IDS-family settings, while the
two-stage system retains its stage-specific validation-selected configurations.
Therefore the manuscript may compare end-to-end system behavior on the same prepared
partitions and operating discipline, but it must not claim that architecture alone
caused the observed differences.

## Phase-1 closure

Phase 1 is closed and frozen. Subsequent JISA work must treat the above values as
read-only claim-bearing evidence unless a new explicitly versioned experiment surface
is approved. Historical Protocol-A/Protocol-B outputs remain untouched.
