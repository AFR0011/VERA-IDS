# JISA Phase 5A — Provenance Joinability Audit

Status: EXECUTION SPEC FROZEN  
Date: 2026-09-12  
Branch: `jisa-q1-revision`  
Parent: `docs/JISA_PHASE5_STATISTICAL_REFRESH_SPEC.md`

## Purpose

The Phase-5 preflight may report `READY_FOR_PHASE5_DESIGN=True` when provenance-bearing metadata exists somewhere in the retained workspace. That condition is intentionally weaker than proving that a claim-bearing row-level score surface can be linked to a defensible provenance unit.

Phase 5A therefore resolves the stronger question before any refreshed inferential test is run:

> Can the row-level observations used by a claim-bearing statistical surface be linked unambiguously to an independently documented source/file/day/unit/group identifier?

No model fitting, threshold selection, rejector selection, bootstrap, permutation test, or p-value calculation is authorized in Phase 5A.

## Important Phase-3 constraint

The row-level `row_id` written by `build_score_frame` in `src/ids_eval_framework/_native/protocol_b_open_set.py` is generated as `np.arange(len(frame))` after the evaluation frame has already been assembled. It is therefore a local positional identifier for that score file, not source provenance.

Consequently:

- `row_id` alone is **not** an admissible cluster identifier;
- `row_id` alone is **not** an admissible join key to source/file/day metadata;
- a Phase-3 score surface is cluster-joinable only if it contains direct provenance columns or an explicit retained mapping artifact links a non-synthetic score key to provenance;
- reconstructing a mapping from row order, deterministic sampling order, or an inferred correspondence is prohibited.

This rule follows the parent Phase-5 requirement not to invent provenance from row ordering.

## Audit hierarchy

For each row-level claim surface, assign exactly one evidence class:

1. `direct_provenance` — the score/prediction file itself contains a documented source/file/day/unit/group column;
2. `verified_explicit_mapping` — a non-synthetic key in the score file joins one-to-one or many-to-one, without ambiguity, to a retained mapping table that contains documented provenance;
3. `candidate_mapping_unverified` — plausible keys/mapping artifacts exist but an exact unambiguous join has not been demonstrated;
4. `conditional_row_only` — no admissible provenance link exists; row-level uncertainty may only be described as conditional on the observed fixed test rows and fitted model.

A generic sequential `index`, `row_id`, or equivalent is not accepted merely because a similarly named field exists elsewhere.

## Required Phase-3 decision

The audit must explicitly report whether the heldout-validation-blind Phase-3 score surface supports cluster-aware row resampling. If not, the statistical refresh must not manufacture cluster-aware Phase-3 p-values. Five-seed variability remains the primary robustness evidence for that surface, while row-level/Wilson intervals may be retained with their conditional interpretation.

## Other surfaces

The audit should also inventory validation-visible rejector score surfaces and any other row-level claim-bearing files discovered by the Phase-5 preflight. A surface may be cluster-ready even if Phase 3 is not. The eventual Phase-5B execution specification must distinguish these evidence classes rather than forcing one inferential procedure across all results.

## Stop conditions

Stop and classify a surface as unverified rather than joining if:

- the only common key is synthetic positional `row_id`/`index`;
- the mapping depends on row order;
- the mapping table is not retained locally;
- the candidate key is non-unique in a way that prevents an unambiguous many-to-one provenance assignment;
- provenance is inferred from labels, predictions, or test outcomes;
- mapping requires changing or rerunning a fitted model before a separate rerun specification is frozen.

## Outputs

Phase 5A writes only under `.release-audit/jisa_phase5_joinability/`:

- `surface_joinability.csv`;
- `mapping_table_candidates.csv`;
- `decision.json`.

The decision file is an audit artifact, not a manuscript result.

## Gate

`READY_FOR_PHASE5B_SPEC=True` means the evidence classes are sufficiently resolved to freeze the actual statistical execution plan. It does **not** mean every surface supports cluster-aware inference.
