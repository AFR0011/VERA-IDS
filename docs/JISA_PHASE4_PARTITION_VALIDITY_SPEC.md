# JISA Phase 4 — CICIDS2017 Partition-Validity Analysis

Status: PREFLIGHT OPEN  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Purpose

Phase 4 addresses a validity problem specific to the CICIDS2017 Protocol-B surface. The recorded data construction cannot simultaneously preserve the original source/day separation and guarantee complete isolation of exact processed representations across partitions. Therefore CICIDS2017 is not used as the canonical unsupported-family benchmark. Instead, it is used to show how admissible partition assumptions can materially change family-level conclusions.

## Scientific question

How sensitive are CICIDS2017 unsupported-family conclusions to the partition construction used to control cross-partition dependence?

## Required comparison

The intended comparison is between two already-established partition constructions, if the complete local artifacts are recoverable:

1. **Recovered contiguous-within-day variant**: preserves the recorded source/day structure more closely but does not claim complete exact-representation isolation.
2. **Group-safe / exact-identity-isolated sensitivity variant**: removes cross-partition exact-representation overlap by grouping exact identities, but may sacrifice source/day independence because exact-identity groups can connect source units.

These are competing validity assumptions, not a clean treatment/control experiment.

## Hard interpretation boundary

- Do not attribute performance changes causally to duplicate removal alone.
- Do not call one partition universally correct and the other invalid.
- Do not combine the two surfaces into one pooled estimate.
- Report both aggregate and family-level shifts.
- Preserve the exact holdout taxonomy used by each surface and compare only common holdouts.
- Test labels remain evaluation-only. Phase 4 should reuse frozen results where possible rather than retune models after observing sensitivity outcomes.
- If the group-safe evidence is incomplete or cannot be linked to a documented construction, stop rather than reconstructing numbers from manuscript prose.

## Preflight gate

Before building a manuscript-facing sensitivity table, inventory local artifacts and confirm:

- the recovered CICIDS2017 Protocol-B result surface is present;
- the group-safe / identity-isolated result surface is present;
- split-construction metadata or reports exist for both surfaces;
- family-level unknown-detection metrics can be traced to local result files;
- train/validation/test row counts can be traced to preparation/split artifacts;
- the exact-overlap/source-group tradeoff is documented by local audit evidence rather than inferred from performance values.

No new model training is authorized by this preflight. If a missing artifact requires regeneration, a separate execution specification must be frozen first.

## Planned manuscript outputs after evidence audit

If the evidence passes, produce a compact table or paired family plot showing the common holdouts under each partition construction, plus aggregate summaries such as mean UDR and mean absolute family-level shift. The discussion must emphasize instability of the security conclusion under plausible partition assumptions rather than presenting one split as a benchmark score correction.
