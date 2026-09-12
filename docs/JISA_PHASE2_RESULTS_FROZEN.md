# JISA Phase 2 Results Frozen — Reference-Profile Provenance Repair

Status: FROZEN AFTER BUILDER PASS  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Closure

The corrected reference-profile builder passed after resolving the historical source-inspired Protocol-A values from the tracked claim-bearing summary surface rather than the plan-only local Protocol-A summary.

Phase 2 is therefore frozen. No historical reference output is overwritten.

## Manuscript-facing provenance classes

The corrected JISA surface keeps three evidence classes separate:

1. `published_source_metric` — metric transcribed from the cited source paper with source task/taxonomy recorded;
2. `source_inspired_profile_result` — VERA result for a paper-inspired model profile under VERA Protocol A;
3. `vera_primary_result` — primary VERA RF/XGB Protocol-A result on the corresponding VERA system metric surface.

The historical values previously stored in `closed_set_reference_values` are preserved separately as `framework_compatible_reference` values and are prohibited from being labelled as paper-reported metrics.

## Main corrected comparison rows

The builder produced three manuscript-facing comparison rows:

- Adewole / CICIDS2017 / XGB;
- Adewole / CICIoT2023 / XGB;
- Neto / CICIoT2023 / RF, using the paper's 8-class result as the primary source anchor.

The source anchors, source-inspired VERA results, and VERA primary results remain descriptively juxtaposed only. Differences between source-paper values and VERA values are not interpreted as causal performance drops because task, taxonomy, split, sampling, preprocessing, averaging, and operating policy are not fully matched.

## Corrected source anchors

- Adewole CICIDS2017 multiclass XGBoost: source F1 `0.9987`.
- Adewole CICIoT2023 binary XGBoost: source F1 `0.9855`.
- Neto CICIoT2023 RF 8-class: source F1 `0.71928904`.
- Neto CICIoT2023 RF 34-class: source F1 `0.714021981` (catalogued, not primary main-text anchor).
- Neto CICIoT2023 RF binary: source F1 `0.965279544` (catalogued, not primary main-text anchor).

## Frozen build boundary

The corrected builder consumes tracked frozen summaries and source anchors only. No model retraining is required for Phase 2. Any later change to source-paper transcription, source-task choice, or source-inspired VERA metrics constitutes a new provenance revision and must not silently replace this surface.

## Manuscript implication

Table 3 / Figure 3 may be retained only after being rebuilt from the corrected provenance surface. Captions and prose must explicitly distinguish published source metrics from VERA source-inspired profile results and VERA primary results.
