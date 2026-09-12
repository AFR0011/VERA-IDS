# JISA manuscript claim-to-source map

Status: MANUSCRIPT RECONSTRUCTION AUTHORITY  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

This map links the reconstructed JISA manuscript's principal claims to the frozen evidence surfaces. It is intentionally narrower than the complete repository inventory: only claims retained in the main article are listed.

| Main-text claim | Manuscript location | Primary authority | Allowed wording / boundary |
|---|---|---|---|
| Closed-set component performance is strong but staged end-to-end macro-F1 is lower because gate errors propagate | Results 4.1, Table 2 | `outputs/summaries/protocol_a_core_summary.csv`; historical primary Protocol-A outputs | Component metrics and system metrics must remain distinct |
| Direct-vs-two-stage ranking depends on dataset and operating policy | Methods 3.3; Results 4.1; Fig. 2 | `docs/JISA_PHASE1_RESULTS_FROZEN.md`; `outputs/13_jisa_q1_revision/phase1_controlled_direct/` | Controlled system comparison, not a pure causal architecture ablation |
| Published source metrics are not the same evidence class as VERA source-inspired or VERA-primary metrics | Methods 3.7; Results 4.2; Table/Fig. 3 | `docs/JISA_PHASE2_RESULTS_FROZEN.md`; corrected Phase-2 summary | Contextual comparison only; no matched-protocol degradation claim |
| CICIoT2023 heldout-validation-blind UDR is strongly family- and rejector-dependent | Methods 3.4–3.5; Results 4.3; Fig./Table 4 | `docs/JISA_PHASE3B_RESULTS_FROZEN.md`; frozen Phase-3B five-seed surface | Call the lane heldout-validation-blind / validation-blind, not fully inductive or live zero-day |
| No single rejector dominates all omitted families under blind selection | Results 4.3; Discussion 5.2 | `docs/JISA_PHASE3B_RESULTS_FROZEN.md` | Descriptive family-specific conclusion; no universal superiority claim |
| Similar known-traffic FUR can coexist with substantially different UDR | Results 4.3 | Phase-3B 3% five-seed means | Supports rejector/family dependence, not population inference |
| Threshold-independent unknown-vs-known separability does not determine useful UDR at a constrained operating point | Results 4.4; Fig./Table 5 | historical validation-visible max-confidence score surface; `outputs/summaries/open_set_unknown_known_curves.csv` and retained tradeoff outputs | Historical validation-visible evidence, separate from blind surface |
| Validation-visible rejector means must use jointly feasible common holdouts | Results 4.4; Fig./Table 6 | `docs/JISA_PHASE5E_RESULTS_FROZEN.md`; `outputs/13_jisa_q1_revision/phase5_visible_common_case/` | Always report feasibility coverage with common-case means |
| No aggregate cross-budget method-mean curve is authorized | Results 4.4; Discussion | `docs/JISA_PHASE5E_RESULTS_FROZEN.md` | Global all-method/all-budget feasible holdout intersection is empty in both datasets |
| Validation-selected rejection costs can generalize imperfectly to test | Results 4.4 | historical budget-audit outputs under `outputs/12_jisa_finalization/16_rejector_budget_audit/` | Test exceedance is a generalization diagnostic; do not retune on test |
| Residual held-out failures concentrate into stable known destinations | Results 4.5; Fig./Table 7 | retained failure-destination outputs under `outputs/12_jisa_finalization/12_failure_destinations/` | Descriptive mapping only; no semantic-equivalence claim |
| CICIDS2017 unsupported-family conclusions are partition-sensitive | Methods 3.4; Results 4.6; Fig. 8 | `docs/JISA_PHASE4_RESULTS_FROZEN.md`; frozen local Phase-4 outputs | Competing admissible dependence assumptions; no duplicate-removal causal claim |
| Strong external binary discrimination can coexist with weaker end-to-end family performance | Results 4.6 | `outputs/summaries/external_protocol_a_summary.csv` | External stress evidence, not Protocol-B unknown-family evidence |
| Five-seed SD quantifies training/selection randomness, not population sampling uncertainty | Methods 3.6; Discussion 5.5 | `docs/JISA_PHASE5_RESULTS_FROZEN.md` | Do not relabel seed variability as deployment uncertainty |
| Row-level bootstrap intervals, where retained, are conditional on fixed observed test rows/models | Methods 3.6; Discussion 5.5 | `docs/JISA_PHASE5_RESULTS_FROZEN.md` | Historical row-bootstrap p-values are not population-level inferential evidence |
| Cluster-aware rejection-score inference is not authorized because exact raw score rows lack admissible provenance clusters | Methods 3.6; Discussion 5.5 | Phase-5C decision; `docs/JISA_PHASE5_RESULTS_FROZEN.md` | State the limitation rather than inventing an independent sampling unit |

## Central manuscript claim

The article's central supported claim is methodological: conclusions about IDS quality can change materially as evaluation moves from closed-set component reporting to composed-system behavior, operating-constraint-matched comparison, held-out-family conditions, validation-blind rejection selection, rejection-cost accounting, residual failure destinations, and partition-validity sensitivity.

The manuscript must not be reframed as a new-detector paper, a state-of-the-art accuracy paper, or evidence of live zero-day detection.
