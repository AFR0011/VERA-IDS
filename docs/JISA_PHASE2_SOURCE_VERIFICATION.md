# JISA Phase 2 — Published-Source Verification

Status: FROZEN SOURCE TRANSCRIPTION; LOCAL PROFILE BUILD PENDING  
Date: 2026-09-12  
Branch: `jisa-q1-revision`  
Parent specification: `docs/JISA_PHASE2_REFERENCE_PROVENANCE_SPEC.md`

## Purpose

The Phase-2 provenance audit correctly identified that the historical
`closed_set_reference_values` cannot be treated as paper-reported metrics. One
legacy row remained unresolved by the first audit: the Adewole et al. CICIoT2023
binary row. That row is now source-verified.

The historical `config/reference_framework_eval.yml` remains unchanged as provenance
of the previous reconstruction surface. The corrected JISA surface is generated
separately.

## Verified Adewole et al. source metrics

Source: Kayode S. Adewole, Andreas Jacobsson, and Paul Davidsson,
"Intrusion Detection Framework for Internet of Things with Rule Induction for Model
Explanation," Sensors 2025, 25(6), 1845. DOI: 10.3390/s25061845.

### CIC-IDS2017, multiclass, XGBoost

Table 7 reports:

- accuracy: `0.9988`;
- precision: `0.9987`;
- recall: `0.9988`;
- F1-score: `0.9987`;
- AUC-ROC: `0.9999`;
- MCC: `0.9961`.

### CICIoT2023, binary, XGBoost

Table 5 reports:

- accuracy: `0.9854`;
- precision: `0.9856`;
- recall: `0.9854`;
- F1-score: `0.9855`;
- AUC-ROC: `0.9306`;
- MCC: `0.8483`.

Therefore the historical VERA values `accuracy = 0.988768` and
`macro_f1 = 0.9394575205434568` are not the paper-reported Adewole CICIoT2023 XGBoost
binary metrics and must not be labelled as such.

## Verified Neto et al. source metrics

Source: the CICIoT2023 dataset paper, Table 6.

Random Forest reports:

- 8-class accuracy: `0.994368173`;
- 8-class F1-score: `0.71928904`;
- 34-class accuracy: `0.99164365`;
- 34-class F1-score: `0.714021981`;
- binary accuracy: `0.99680798`;
- binary F1-score: `0.965279544`.

For the JISA manuscript-facing comparison, the 8-class RF result is the primary Neto
source anchor because it is the closest of the paper's reported multiclass surfaces
to a coarse attack-category comparison. The 34-class and binary results remain in
the provenance catalog and must be available in notes/supplementary material. This
choice does not make the source task directly comparable to the VERA taxonomy.

## Legacy-row verdict

All three historical `closed_set_reference_values` rows are now confirmed to be
VERA-internal/framework-compatible reference values rather than paper-reported source
metrics:

1. Adewole / CICIDS2017 / family: not source-reported;
2. Adewole / CICIoT2023 / binary: not source-reported;
3. Neto / CICIoT2023 / family: not source-reported.

They remain historically preserved but are prohibited from the JISA manuscript under
a `published_source_metric` label.

## Manuscript-facing comparison rule

The corrected Table 3 / Figure 3 surface must keep the following provenance classes
separate:

1. `published_source_metric` — exact source-paper metric and source task;
2. `source_inspired_profile_result` — local VERA result from the paper-inspired model
   profile under VERA Protocol A;
3. `vera_primary_result` — primary VERA RF/XGB Protocol-A result on the same VERA
   system metric surface.

No numeric difference from class 1 to classes 2 or 3 is interpreted as a causal
performance drop because the source task, taxonomy, split, sampling, preprocessing,
averaging, and operating policy are not fully matched.

## Next gate

Run the JISA Phase-2 corrected-summary builder. It must consume existing local
reference-profile outputs only and must not retrain models. The builder must fail
closed if the expected local source-inspired profile rows are missing or ambiguous.
