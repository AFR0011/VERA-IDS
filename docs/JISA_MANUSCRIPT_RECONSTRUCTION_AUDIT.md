# JISA manuscript reconstruction audit

Status: MAIN MANUSCRIPT RECONSTRUCTED  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Reconstruction completed

The JISA main manuscript was rebuilt from the frozen evidence hierarchy rather than incrementally preserving the pre-freeze v5 Results narrative.

Main structural changes:

- retained the evaluation-methodology title and contribution framing;
- rewrote Abstract, Introduction, Methods, Results, Conclusion;
- added a substantive Discussion section with limitations and scope of inference;
- added the five-seed controlled direct-vs-two-stage result as a main evidence surface;
- rebuilt the literature/reference-profile comparison using corrected published source anchors;
- made CICIoT2023 heldout-validation-blind rejection the canonical unsupported-family evidence;
- separated historical validation-visible rejection from blind selection;
- replaced unequal-feasible-subset rejector means with the Phase-5E jointly feasible common-case comparison;
- removed the previous aggregate cross-budget method-mean curve because no fixed all-method/all-budget common holdout set exists;
- retained residual destination analysis as a main security-failure diagnostic;
- reframed CICIDS2017 as partition-validity sensitivity rather than canonical open-set evidence;
- removed population-level interpretation of historical row-bootstrap p-values;
- added the explicit limitation that raw rejection-score rows do not retain a defensible provenance cluster for cluster-aware inference.

## Main figures in reconstructed manuscript

1. VERA-IDS evaluation chain (retained schematic).
2. Five-seed controlled two-stage vs direct argmax vs validation-constraint-matched direct comparison.
3. Corrected published/source-inspired/VERA-primary reference provenance comparison.
4. CICIoT2023 heldout-validation-blind UDR by holdout and rejector with seed variability.
5. Historical validation-visible unknown-vs-known AUROC versus selected UDR.
6. Validation-visible 3% common-case rejector UDR/FUR comparison.
7. Residual failure-destination structure.
8. CICIDS2017 recovered-vs-group-safe partition sensitivity.

## Main tables

1. Dataset roles, sizes, and validity boundaries.
2. Closed-set component/system results plus broader competitive direct benchmark.
3. Corrected provenance-separated source/source-inspired/VERA-primary comparison.
4. CICIoT2023 blind 3% holdout-by-rejector five-seed results.
5. Historical validation-visible max-confidence separability/operating-point results.
6. Validation-visible 3% common-case rejector summary with feasibility coverage.
7. Residual failure destinations and concentration.

## Current manuscript size

The rendered reconstruction is approximately 22 pages in the working Times New Roman single-column format, with about 5,430 words before References, 8 figures, 7 tables, and 42 retained references.

The current file is a scientific/content reconstruction, not an Elsevier typesetting template. Submission-system formatting should be applied only after the scientific text and Supporting Information are synchronized.

## Remaining synchronization tasks before submission

1. Update Supporting Information so the heldout-validation-blind selection rule, Phase-5 statistical boundary, corrected source metrics, and common-case rejector reporting match the reconstructed main manuscript.
2. Recheck any supplementary tables that still contain the historically mislabelled source F1 values or unequal-feasible-subset method means.
3. Ensure figure source files and final machine-readable tables copied into the submission package correspond to the frozen JISA revision surfaces.
4. Run a final citation/DOI and Elsevier/JISA submission-metadata audit after Supporting Information synchronization.
5. Add corresponding-author contact metadata if required by the final submission form/template; it is not invented in the reconstruction because no verified email was supplied in the source document.

## Scientific submission verdict after reconstruction

The revised manuscript is now framed as an IDS evaluation-validity contribution rather than a routine application of existing ML models. Its strongest JISA-relevant evidence is the combination of operating-policy-dependent system ranking, validation-blind unsupported-family rejection, rejection-cost/common-case analysis, residual destination structure, and partition-validity sensitivity.

The remaining work is editorial/synchronization work, not a justification for additional experiments unless a new provenance contradiction is discovered.
