# JISA Phase 4 — CICIDS2017 Partition-Validity Analysis

Status: PREFLIGHT PASSED; BUILD OPEN  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Purpose

Phase 4 addresses a validity problem specific to the CICIDS2017 Protocol-B surface. The recorded data construction cannot simultaneously preserve the original source/day separation and guarantee complete isolation of exact processed representations across partitions. Therefore CICIDS2017 is not used as the canonical unsupported-family benchmark. Instead, it is used to show how admissible partition assumptions can materially change family-level conclusions.

## Scientific question

How sensitive are CICIDS2017 unsupported-family conclusions to the partition construction used to control cross-partition dependence?

## Required comparison

The intended comparison is between two already-established partition constructions whose local artifacts were recovered by the Phase-4 preflight:

1. **Recovered contiguous-within-day variant**: preserves the recorded source/day structure more closely but does not claim complete exact-representation isolation.
2. **Group-safe / exact-identity-isolated sensitivity variant**: removes cross-partition exact-representation overlap by grouping exact identities, but may sacrifice source/day independence because exact-identity groups can connect source units.

These are competing validity assumptions, not a clean treatment/control experiment.

## Group-safe candidate-selection clarification

The preflight established that the group-safe local `summary/best_per_holdout.csv` is not a one-row-per-holdout surface. It retains two RF Stage-1 weighting finalists for each of the six holdouts. Therefore it must not be treated as already uniquely selected.

For Phase 4, one group-safe winner per holdout is selected **without using test labels or test metrics**, using the same frozen validation-only hierarchy already used for Protocol-B profile selection elsewhere in VERA-IDS:

1. maximum validation Stage-2 macro-F1;
2. tie-break maximum validation Stage-1 AUROC;
3. final deterministic tie-break lexical run name.

This selection rule is applied only to the two retained group-safe finalists for each holdout. Unknown-detection rate, test macro-F1, test accuracy, and all other test outcomes are prohibited from winner selection. The selected run name and validation metrics must be preserved in the Phase-4 output for auditability.

If the retained group-safe finalist table does not contain the validation Stage-2 macro-F1 and validation Stage-1 AUROC needed to apply this rule, the build must stop rather than infer a winner from test performance.

## Hard interpretation boundary

- Do not attribute performance changes causally to duplicate removal alone.
- Do not call one partition universally correct and the other invalid.
- Do not combine the two surfaces into one pooled estimate.
- Report both aggregate and family-level shifts.
- Preserve the exact holdout taxonomy used by each surface and compare only common holdouts.
- Test labels remain evaluation-only. Phase 4 reuses frozen candidate results and performs no model retraining or post-hoc test-based retuning.
- If the group-safe evidence is incomplete or cannot be linked to a documented construction, stop rather than reconstructing numbers from manuscript prose.

## Preflight gate

The Phase-4 preflight passed locally on 2026-09-12:

- recovered CICIDS2017 six-holdout result surface present;
- group-safe / identity-isolated six-holdout candidate result surface present;
- group-safe split-construction and support artifacts present;
- family-level unknown-detection metrics traceable to local result files.

No new model training is authorized by this phase.

## Planned manuscript outputs after evidence audit

Produce a compact table or paired family plot showing the six common holdouts under each partition construction, plus aggregate summaries including mean UDR, mean absolute family-level shift, maximum family-level shift, and family-rank correlation. The discussion must emphasize instability of the security conclusion under plausible partition assumptions rather than presenting one split as a benchmark score correction.
