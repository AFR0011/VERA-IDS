# JISA Phase 2 — Reference-Profile Provenance Repair

Status: FROZEN BEFORE REBUILD OF TABLE/FIGURE 3  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Purpose

The current reference-profile surface mixes three conceptually different quantities:

1. metrics actually reported in the cited source paper;
2. framework-compatible/reconstructed profile values stored in VERA-IDS;
3. VERA-IDS results obtained by evaluating a source-inspired model profile through VERA protocols.

These categories must be separated before Table 3 / Figure 3 are retained in the JISA manuscript. A source-paper metric may not be relabelled as a VERA-compatible reproduction, and a VERA-compatible profile value may not be presented as the metric reported by the source paper.

## Verified source anchors

### Adewole et al. 2025 — Sensors 25(6), 1845

Source: `https://www.mdpi.com/1424-8220/25/6/1845`

For CIC-IDS2017 multiclass classification with all feature sets, the paper's Table 7 reports XGBoost approximately:

- accuracy: `0.9988`;
- precision: `0.9987`;
- recall: `0.9988`;
- F1-score: `0.9987`.

The legacy VERA configuration value `macro_f1 = 0.9205930209401167` is therefore not the paper-reported source F1 and must not be labelled as such.

### Neto et al. 2023 — CICIoT2023 dataset paper

Source: `https://pmc.ncbi.nlm.nih.gov/articles/PMC10346235/`

The paper's Table 6 reports Random Forest results:

- 8-class accuracy: `0.994368173`;
- 8-class F1-score: `0.71928904`;
- 34-class accuracy: `0.99164365`;
- 34-class F1-score: `0.714021981`;
- binary accuracy: `0.99680798`;
- binary F1-score: `0.965279544`.

The legacy VERA configuration value `macro_f1 = 0.904974505109935` is therefore not the paper-reported source RF F1 for either the 8-class or 34-class source task and must not be labelled as such.

## Required provenance categories

Every retained reference row must carry an explicit provenance class:

- `published_source_metric`: value transcribed from the cited paper, with source task/taxonomy and averaging definition recorded;
- `framework_compatible_reference`: a VERA-internal reconstructed or compatibility value that is not claimed to be source-reported;
- `source_inspired_profile_result`: a VERA result produced by a model profile reconstructed/inspired from the paper and evaluated under a VERA protocol;
- `vera_primary_result`: the corresponding primary VERA model/system result under the same VERA metric surface when a contextual comparison is appropriate.

## Hard comparison boundary

A numeric difference between a published source metric and a VERA result is descriptive only unless dataset representation, taxonomy, split construction, sampling, preprocessing, metric averaging, and operating policy are matched. The JISA manuscript must not call such differences performance gains/losses attributable to the evaluation framework.

The reference comparison is retained to demonstrate that conventional literature-facing metrics can look strong while the same broad model family/profile behaves differently under stricter VERA evaluation surfaces. It is not an exact replication study.

## Table 3 / Figure 3 design

The main-text comparison should visually distinguish at least:

- source-reported closed-set metric;
- VERA source-inspired profile under Protocol A / framework-compatible surface;
- VERA primary comparison where meaningful.

Task/taxonomy labels must be printed directly in the table or caption. CICIoT2023 source RF results must specify whether the source value is 8-class or 34-class. A single unlabeled 'source F1' value is not admissible.

## Historical preservation

`config/reference_framework_eval.yml` and its historical outputs are evidence of the previous reconstruction surface and should not be silently rewritten in place. Phase 2 should create a JISA-specific provenance map and corrected manuscript-facing summaries while preserving historical artifacts.

No model rerun is required merely to repair source-metric provenance. Existing source-inspired VERA profile runs may be reused if their local provenance and metric definitions pass audit.
