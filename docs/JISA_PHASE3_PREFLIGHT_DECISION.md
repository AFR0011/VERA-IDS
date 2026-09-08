# JISA Phase 3 Preflight Decision

Status: PRE-EXECUTION DECISION; NO BLIND TEST RESULTS INSPECTED
Date: 2026-09-08
Branch: `jisa-q1-revision`
Parent: `docs/JISA_PHASE3_UNKNOWN_BLIND_SPEC.md`

## Preflight result

The corrected local-artifact preflight recovered the intended CICIoT2023 repeated-seed
Protocol-B surface:

- seeds 123–127: 5/5;
- holdouts: Botnet, BruteForce, DDoS, DoS, Other, Scan/Recon;
- candidate groups: 30/30;
- three candidates per group: 30/30;
- strict Stage-1 LOAO: true;
- prepared Protocol-B data: available;
- stable historical winner identity across all five seeds for each holdout: true.

However, only 5/30 seed-by-holdout groups had a unique winner under the blind-safe
primary profile-selection criterion, represented-family Stage-2 validation macro-F1.
Those five groups are the BruteForce holdout. In the other 25 groups, the top Stage-2
validation macro-F1 is tied, so the historical Stage-1 AUROC tie-break cannot be
reused because it may include held-out-family validation observations.

Therefore the historical winner is **not globally blind-safe** and the first preflight
correctly returned `historical_profile_reuse_blind_safe: False`.

## Narrowing rule for reruns

The parent specification states that ambiguous groups require rerunning candidate
profiles under heldout-free selection. Before execution, this is clarified as follows.

Profile selection remains lexicographic:

1. maximize represented-family Stage-2 validation macro-F1;
2. among candidates tied at the maximum primary value, apply the heldout-free
   Stage-1 validation tie-break defined for the blind lane;
3. apply a deterministic lexical tie-break only if the blind-safe secondary metric
   also ties.

A candidate that is **strictly below the maximum Stage-2 validation macro-F1** cannot
win under this frozen lexicographic rule regardless of any Stage-1 metric. Such a
candidate is already excluded by a blind-safe validation quantity and does not need
to be retrained merely to resolve a tie among higher-ranked candidates.

Accordingly:

- unique-primary groups require one targeted blind rerun of the unique primary
  profile so blind Stage-1/rejector thresholds can be selected;
- tied-primary groups require targeted blind reruns of every member of the top
  Stage-2 equivalence set;
- strictly dominated lower-primary candidates are not rerun for profile-selection
  purposes;
- if a purportedly dominated candidate is not separated from the top by more than
  the fixed numerical tolerance, it remains in the equivalence set and must be
  rerun.

This is a computational pruning rule only. It does not inspect test metrics, change
the primary criterion, change the profile search space after seeing blind outcomes,
or introduce a new model preference.

## Required follow-up audit

Run `scripts/jisa_phase3_tie_audit.py` against the selected historical seed aggregates.
It must record for every seed-by-holdout group:

- top-equivalence-set size and identities;
- gap from the top equivalence set to the next lower candidate;
- whether all top candidates remain strict LOAO;
- whether tied candidates share the same Stage-2 model definition, which may permit
  computational reuse of a single Stage-2 fit without changing predictions or
  selection.

Blind-lane training must not begin until that audit reports `EXECUTION_READY=True`.
