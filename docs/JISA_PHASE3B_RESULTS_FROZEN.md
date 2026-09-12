# JISA Phase 3B Results Frozen — Heldout-Validation-Blind Rejector Replay

Status: FROZEN AFTER FULL AUDIT PASS  
Date: 2026-09-12  
Branch: `jisa-q1-revision`  
Parent execution specification: `docs/JISA_PHASE3B_EXECUTION_SPEC.md`  
Runner: `scripts/jisa_phase3b_replay_validation_blind_rejectors.py`  
Audit: `scripts/jisa_phase3b_audit_results.py`

## Frozen completion state

The complete CICIoT2023 Phase-3B matrix contains 30/30 seed × holdout cases and was replayed from the frozen Phase-3A score surface without model retraining.

- seeds: 123, 124, 125, 126, 127;
- holdouts: Botnet, BruteForce, DDoS, DoS, Other, Scan/Recon;
- rejectors: entropy, family-conditional/conformal-inspired, margin, maximum confidence;
- nominal rejection budgets: 1%, 3%, 5%, and 10%;
- primary budget: 3%;
- held-out-family validation observations are absent from rejector selection;
- test information does not enter rejector selection;
- all 24 holdout × method groups at the primary 3% budget have 5/5 feasible seeds;
- the full Phase-3B audit passed.

The Phase-3B result surface is therefore frozen. Changing the score surface, Stage-1 threshold, rejector definition, validation constraint, budget, alpha grid, boundary handling, or selection objective constitutes a new experiment surface.

## Primary 3% five-seed results

Five-seed mean unknown-detection rate (UDR), all-known false-unknown rate (FUR), and supported-label system macro-F1 are:

| Holdout | Rejector | UDR mean | All-known FUR mean | Macro-F1 mean |
|---|---|---:|---:|---:|
| Botnet | entropy | 0.996825 | 0.024374 | 0.859768 |
| Botnet | family-conditional | 0.998878 | 0.027197 | 0.892750 |
| Botnet | margin | 0.989806 | 0.024199 | 0.874196 |
| Botnet | maximum confidence | 0.993436 | 0.024218 | 0.868662 |
| BruteForce | entropy | 0.323357 | 0.023836 | 0.794982 |
| BruteForce | family-conditional | 0.000000 | 0.026705 | 0.795293 |
| BruteForce | margin | 0.323357 | 0.023838 | 0.794962 |
| BruteForce | maximum confidence | 0.323357 | 0.023840 | 0.794967 |
| DDoS | entropy | 0.387261 | 0.025121 | 0.749338 |
| DDoS | family-conditional | 0.704969 | 0.025021 | 0.826059 |
| DDoS | margin | 0.277948 | 0.024609 | 0.735729 |
| DDoS | maximum confidence | 0.306977 | 0.024723 | 0.742778 |
| DoS | entropy | 0.101455 | 0.024751 | 0.733626 |
| DoS | family-conditional | 0.242726 | 0.024914 | 0.796453 |
| DoS | margin | 0.085353 | 0.024630 | 0.744191 |
| DoS | maximum confidence | 0.088073 | 0.024640 | 0.739356 |
| Other | entropy | 0.324344 | 0.025197 | 0.723754 |
| Other | family-conditional | 0.120905 | 0.027083 | 0.682709 |
| Other | margin | 0.322996 | 0.025542 | 0.721344 |
| Other | maximum confidence | 0.323990 | 0.025371 | 0.722253 |
| Scan/Recon | entropy | 0.503461 | 0.025953 | 0.771023 |
| Scan/Recon | family-conditional | 0.399441 | 0.026730 | 0.816511 |
| Scan/Recon | margin | 0.492815 | 0.025797 | 0.768618 |
| Scan/Recon | maximum confidence | 0.498893 | 0.026045 | 0.769788 |

## Scientific interpretation boundary

The frozen primary result supports the following manuscript-level conclusions:

1. Held-out-family detectability remains strongly family-dependent when held-out-family validation observations are unavailable to rejector selection.
2. Rejector ranking is also family-dependent. No single uncertainty rule dominates all omitted families.
3. Comparable known-traffic false-unknown rates can coexist with very different unknown-detection rates, so the family-level UDR differences are not explained merely by one method rejecting vastly more known traffic.
4. Family-conditional rejection can be highly effective for some holdouts (for example DDoS and DoS) while failing badly for others (for example BruteForce and Other).
5. The five-seed result is a robustness surface for heldout-validation-blind selection, not a paired causal estimate of the effect of hiding unknown validation observations.

## Visible-comparison boundary

The Phase-3B preflight found no complete seed-resolved historical validation-visible score surface that can be paired unambiguously with the new five-seed blind result by seed × holdout × method × budget.

Therefore:

- historical validation-visible Protocol-B results remain separately labelled contextual evidence;
- no five-seed paired blind-minus-visible effect size is authorized;
- no formal paired test between the historical visible and new blind surfaces is authorized;
- regenerating a matched visible surface would constitute a separate additional experiment and is not required for the validity of this robustness analysis.

## Budget interpretation

The nominal 10% budget is not guaranteed to produce 10% validation rejection. The independently frozen 5% all-known false-unknown constraint can bind first, making the selected 10% operating point equal to or close to the 5% point. This behavior is intentional and must not be post-hoc relaxed.

## Manuscript placement

Recommended main-text use:

- primary 3% five-seed holdout × rejector UDR figure with seed variability;
- concise table or text reporting representative UDR/FUR/macro-F1 values;
- explicit statement that visible and blind surfaces are not paired.

Recommended supplementary use:

- full 1%, 3%, 5%, and 10% budget sensitivity;
- complete per-seed selected operating points and feasibility metrics;
- threshold-independent AUROC/AUPR for global rejectors.

The result should be framed as an evaluation-validity finding, not as evidence that one rejector is universally superior.
