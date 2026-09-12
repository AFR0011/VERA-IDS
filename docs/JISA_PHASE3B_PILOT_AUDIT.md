# JISA Phase 3B Pilot Audit

Status: BOTH PILOT PATHS ACCEPTED; FULL PHASE-3B MATRIX OPEN
Date: 2026-09-12
Branch: `jisa-q1-revision`
Parent execution spec: `docs/JISA_PHASE3B_EXECUTION_SPEC.md`
Runner: `scripts/jisa_phase3b_replay_validation_blind_rejectors.py`

## Purpose

The Phase-3B pilot gate required one XGB-derived score case and one RF-derived score case before opening the full five-seed x six-holdout rejector replay. The pilot verifies all four rejector families, all four budgets, validation-only feasibility selection, and the intended interaction between the nominal 10% budget and the frozen 5% all-known false-unknown constraint.

## BruteForce seed 123

The XGB-derived pilot produced 16/16 feasible selected points (four methods x four budgets).

At the primary 3% budget:

- entropy: validation rejection `0.029999`, test UDR `0.324053`, test all-known FUR `0.023690`, test supported-label macro-F1 `0.795236`;
- family-conditional: validation rejection `0.026929`, test UDR `0.000000`, test all-known FUR `0.026597`, test supported-label macro-F1 `0.795266`;
- margin: validation rejection `0.029999`, test UDR `0.324053`, test all-known FUR `0.023698`, test supported-label macro-F1 `0.795201`;
- maximum confidence: validation rejection `0.029999`, test UDR `0.324053`, test all-known FUR `0.023697`, test supported-label macro-F1 `0.795212`.

At the 5% and nominal 10% budgets, the three global rejectors selected the same 5% validation operating point, while family-conditional selected `0.047200`. This is consistent with the frozen 5% all-known validation false-unknown ceiling. The nominal 10% budget therefore does not force a larger rejection rate.

The equality of the BruteForce test UDR values for entropy, margin, and maximum-confidence at several budgets is not itself a protocol error: the three methods may select different uncertainty boundaries yet induce the same rejected unknown subset at those operating points. The manuscript should not imply identical score functions from this equality.

## Botnet seed 123

The RF-derived pilot also produced 16/16 feasible selected points.

At the primary 3% budget:

- entropy: validation rejection `0.029999`, test UDR `0.996578`, test all-known FUR `0.024215`, test supported-label macro-F1 `0.859851`;
- family-conditional: validation rejection `0.027572`, test UDR `0.998884`, test all-known FUR `0.026297`, test supported-label macro-F1 `0.893637`;
- margin: validation rejection `0.029991`, test UDR `0.990550`, test all-known FUR `0.024231`, test supported-label macro-F1 `0.874356`;
- maximum confidence: validation rejection `0.029988`, test UDR `0.993451`, test all-known FUR `0.024289`, test supported-label macro-F1 `0.868007`.

At the 5% and nominal 10% budgets, each method again selected the same operating point across those two nominal budgets, consistent with the frozen 5% all-known validation false-unknown constraint.

The large differences between BruteForce and Botnet are scientifically plausible and are retained as results rather than normalized away. They show that the usefulness of a rejection rule remains strongly family-dependent under heldout-validation-blind selection.

## Pilot verdict

Both required Phase-3B pilot paths pass:

1. XGB-derived score surface: BruteForce seed 123;
2. RF-derived score surface: Botnet seed 123;
3. all four rejector families represented;
4. all four nominal budgets represented;
5. all 32 pilot method-budget points feasible on the blind validation surface;
6. the 10% budget is effectively capped by the predeclared 5% all-known validation FUR constraint, as expected;
7. no unknown-detection metric enters validation selection.

The complete 30-case Phase-3B replay is open. Completed pilot cases must remain resumable/skippable without `--force`. After all 30 cases finish, the five-seed summary must be audited before manuscript claims are frozen.

## Comparison boundary

The Phase-3B preflight found no seed-resolved historical validation-visible score surface that can be paired with these five-seed blind results. Therefore these results remain a separate heldout-validation-blind robustness analysis. Historical validation-visible results may be discussed as separately labelled context only. No five-seed paired visible-vs-blind effect size is authorized by this pilot audit.
