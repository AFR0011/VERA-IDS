# JISA Phase 4 — Results Frozen

Status: FROZEN AFTER LOCAL BUILD + AUDIT PASS  
Date: 2026-09-12  
Branch: `jisa-q1-revision`

## Scope

Phase 4 evaluates CICIDS2017 only as a partition-validity sensitivity surface. It does not replace CICIoT2023 as the canonical unsupported-family benchmark.

The comparison uses two already-computed CICIDS2017 Protocol-B constructions:

1. recovered contiguous-within-day Protocol B;
2. group-safe / exact-identity-isolated sensitivity Protocol B.

The group-safe summary retained two RF Stage-1 weighting finalists per holdout. Phase 4 selects one finalist per holdout using only the frozen validation hierarchy:

1. maximum validation Stage-2 macro-F1;
2. tie-break maximum validation Stage-1 AUROC;
3. final tie-break lexical run name.

No test outcome enters this selection.

## Frozen local outputs

The claim-bearing machine-readable Phase-4 outputs are generated locally under:

`outputs/13_jisa_q1_revision/phase4_partition_validity/`

Required files:

- `cicids_partition_sensitivity_by_holdout.csv`
- `cicids_partition_sensitivity_summary.json`
- `group_safe_validation_selection_audit.csv`
- `build_manifest.json`

The independent audit is written under:

`.release-audit/jisa_phase4_results/`

The owner reported the Phase-4 audit as passed. Numeric manuscript claims must be transcribed from the frozen local machine-readable outputs above, not reconstructed from prose or memory.

## Claim boundary

Allowed interpretation:

- CICIDS2017 unsupported-family conclusions are sensitive to the admissible partition construction;
- both aggregate UDR and family-level shifts should be reported;
- family ordering may change under different defensible dependence assumptions.

Not allowed:

- attributing the observed changes causally to duplicate removal alone;
- presenting one partition as universally correct and the other as invalid;
- pooling the two partitions into one estimate;
- treating CICIDS2017 as the canonical open-set benchmark after the partition-validity finding.

## Audit requirements satisfied

The local audit verifies:

- exactly six common holdouts;
- exact holdout identity;
- aggregate-statistic recomputation from the family-level table;
- 12 group-safe finalists and six validation-selected winners;
- exactly one winner per holdout;
- selected run names match the paired sensitivity table;
- validation-only selection policy;
- build-input hashes match the manifest.

## Manuscript role

Phase 4 should appear as a compact partition-sensitivity result, ideally a paired family plot or concise table. Its purpose is to demonstrate that a security conclusion can depend materially on how cross-partition dependence is controlled, not to advertise an alternative benchmark score.
