# CICIDS2017 partition regeneration preflight

Updated 2026-11-19. Historical-recipe replay, user-adopted contiguous-assignment
metadata audit and raw-label reconciliation. No new split data or trained models
were produced.

## Recovered group-safe recipe

The original materializer was recovered from Git commit
`bff66918cbf28d6a1d70e17bcfaeb51a7f11057b`, historical path
`scripts/15_materialize_cicids2017_group_safe_surface.py`. Its exact byte hash is
`88c3091f14997f7cfc059d415284b00fcaf49b131ae859a7a1155b7136646e3d`.
The audit wrapper disables the historical materialization/deletion entrypoints
and calls only `build_assignment` at seed 123, support floor 200 and historical
chunk size 200,000. Input files are hashed before and after the calculation.

The original method hashes the 84 processed predictors, applies seed-mixed
60/20/20 buckets, then makes deterministic whole-group transfers to repair
family support. Groups containing conflicting family labels are not used as
repair donors. It does not preserve raw day/file separation.

The replay matched all 13 saved aggregate, support, feature-order and repair
checks in 15.856 seconds, using peak worker RSS 1,007,697,920 bytes. Available
RAM was telemetry only; the retained 4 GiB worker and disk/time/artifact guards
held. Completion and artifact hashes were independently rechecked.

| Quantity | Reproduced value |
| --- | ---: |
| Input parts | 24 |
| Rows | 2,099,879 |
| Processed predictor columns | 84 |
| Unique 64-bit feature-hash groups | 1,735,180 |
| Groups with conflicting family labels | 14 |
| Repair moves / rows | 109 / 109 |
| Training rows | 1,254,207 |
| Validation rows | 412,200 |
| Test rows | 433,472 |

New current-environment group-to-split checksum:
`44b48ca56a8240b938ebb5e02c93ab814c94f411a761f08d87403d4df3cc2c7a`.
The old evidence contains aggregate fingerprints, not this assignment checksum.
Matching aggregates therefore does not establish historical row-by-row identity.
This audit also does not prove collision-free equivalence of 64-bit feature
hashes or independent source groups. The 14 conflicting-label groups are retained
as historical evidence, not silently relabeled or dropped.

Evidence: [replay receipt](../experiments/cicids_partition/REPLAY_SUCCESS.json),
[audit wrapper](../experiments/cicids_partition/replay.py), and the unchanged
`outputs/12_jisa_finalization/04_group_safe_surface/` aggregate evidence.
The wrapper is an internal archival audit requiring recovered local source bytes;
it is not advertised as a clean-clone public data-preparation workflow.
The release-content test permits exactly this wrapper's recorded SHA-256 as an
internal archival exception. The general ban on historical dynamic loading in
public workflows remains enforced; changed wrapper bytes fail the exception.

## Contiguous-source construction: rule adopted 2026-11-18

The historical recovered surface uses 20 contiguous units: four segments from
each of five raw day files. Its saved exhaustive best assignment has 12 training,
four validation and four test units. All six attack families pass the adopted
200-row support floors in that saved assignment. The packaged exhaustive search
does not consult an RNG; changing only its seed cannot produce five new partitions.

The local `processed_V5/B_day_file/CICIDS2017` directory is instead the older
whole-file construction with zero selected holdouts. It must not be substituted
for the recovered contiguous surface. The recovered metadata remains in the
protected research snapshot and has only been read.

**User-adopted assignment rule:** retain the
same ordered 20 units and their boundaries, draw seeded random permutations,
assign the first 12 units to training, next four to validation and last four to
test, and accept the first assignment with at least 200 rows of every retained
family (including benign) in each split. Stop after 10,000 attempted assignments
per seed and report failure explicitly. There are 8,817,900 possible unordered
12/4/4 assignments before support filtering. This rule samples support-eligible
assignments rather than maximizing the historical support score; that change in
the partition distribution was explicitly approved by the user on 2026-11-18.

The approved first check uses partition seeds 123–127 at the metadata level
only: exact assignments, support tables, deterministic regeneration and actual
distinctness. No classifier fitting, split materialization, new label mapping,
raw-day independence claim or performance-based partition choice. Duplicate
assignments across seeds would be reported, not silently redrawn. Model seed is
a separate variable even when its numeric value happens to match a partition seed.

### Metadata audit result

All five seeds produced distinct feasible assignments and regenerated identically.
Each split meets the 200-row floor for all seven retained families, including
benign. The limiting family is Web/App; its 691 rows make admissibility restrictive.
No failed seed or duplicate was discarded. Counts were independently reconciled
using Python integers against the saved per-unit family table.

| Partition seed | First valid draw | Train rows | Validation rows | Test rows | Minimum family count (train / val / test) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 123 | 168 | 1,223,347 | 432,067 | 444,465 | 241 / 213 / 237 |
| 124 | 237 | 1,179,709 | 475,710 | 444,460 | 237 / 241 / 213 |
| 125 | 107 | 1,213,390 | 442,024 | 444,465 | 213 / 241 / 237 |
| 126 | 263 | 1,213,390 | 444,462 | 442,027 | 213 / 237 / 241 |
| 127 | 244 | 1,179,717 | 475,708 | 444,454 | 241 / 237 / 213 |

The bounded audit completed in 0.147 seconds with peak RSS 59,392,000 bytes.
[Compact receipt](../experiments/cicids_partition/CONTIGUOUS_ASSIGNMENT_SUCCESS.json)
contains exact unit assignments, support tables, source hashes, RNG/version,
per-seed assignment checksums and verification scope. Its SHA-256 is
`9129d5d1b3e18f0b784d0ce23d9d55cb9dec7b6da8a78ba539c0c1b903f85fc7`.
[Implementation](../experiments/cicids_partition/seeded_assignments.py) writes
immutable per-seed checkpoints, preserves the original shared deadline on
recovery, and can finish an interrupted completion receipt without replacing
the existing result. Read-only verification replays every seed.

This establishes metadata feasibility only. Raw-row materialization, predictor
and label identity checks, model fitting and outcome comparisons remain undone.
The five realizations are dependent partitions of the same corpus; 200 rows do
not establish independent source support. No full campaign is authorized here.

The group-safe and contiguous-source constructions remain separate surfaces.
Saved duplicate-provenance evidence links all 20 contiguous units through shared
processed-feature hashes; preserving those units and removing all such overlap
cannot be asserted simultaneously. No full campaign is frozen by this preflight.

### Local raw-label reconciliation — 2026-11-19

A bounded label-only pass over the five local raw day files reproduced all
twenty saved per-unit retained-family counts, row boundaries and dropped-row
totals using the existing native label mapper. Header inspection found 89 unique
canonical columns in each source. No predictor values were parsed, split files
written or models fitted. File hashes were checked before/after each scan and
again during independent receipt reconciliation. Per-file checkpoints preserve
completed scans under the original deadline.

The audit completed in 10.527 seconds with peak worker RSS 144,826,368 bytes.
[Compact receipt](../experiments/cicids_partition/RAW_LABEL_RECONCILIATION_SUCCESS.json)
records the five current source hashes, twenty support comparisons and the
native preparation-source hash. This establishes agreement of labels/support
with the saved metadata; it does not prove historical raw-byte identity or
predictor equality. Predictor reconstruction was checked separately below.

### Exact prepared-value reconciliation — 2026-11-19

The five raw-file headers reproduce the saved ordered schema: 84 predictors
and three target columns. The existing preparation drops the same recorded
identifier/time/port columns, retains the two coarse port buckets, maps labels,
and drops the historical `Other` family. No new blacklist, mapper or imputation
rule was introduced. The feature-description CSV and `_plus` sources are outside
the five-source input contract.

A bounded read-only audit reconstructed each of the twenty contiguous units
and compared it with the corresponding historical prepared part identified by
the saved split manifest. **All 2,099,879 rows matched in order across all 87
columns, using exact parsed numeric/text equality with no tolerance.** No
nonfinite values were found in reconstructed numeric columns. Storage dtype
differences were confined to the two categorical port buckets and the binary
target; their values matched. This verifies parsed values, not byte-identical
CSV serialization or historical raw-file identity.

The audit completed in 41.531 seconds, with peak worker RSS 455,258,112 bytes,
under the existing 900-second/4-GiB worker limits and disk/artifact guards.
Available RAM remained telemetry only. Per-unit checkpoints retain completed
comparisons under the original deadline. Source, reference, metadata and runner
hashes were independently reconciled after completion. The
[compact receipt](../experiments/cicids_partition/PREDICTOR_RECONCILIATION_SUCCESS.json)
has SHA-256 `2e8e9f22fcd65f0b477c6e1881e2841d2fdc65b0fae41cb392ea1b34a657ba7e`.
It records source-unit boundaries, retained-position hashes, prepared-part hashes
and exact-value checks. No predictor arrays, new splits or fitted models were
written by this audit.

This removes the processed-value reconstruction uncertainty for these inputs.
It does not remove shared feature identities across units or establish
independent attack executions. Identical duplicate rows also prevent a claim of
unique historical provenance from value equality alone.

On 2026-11-20 the user approved materializing one seed-123 partition, using its
already recorded first-valid assignment and the exact verified unit contents.
The twenty historical compressed parts total 227,120,777 bytes, so separate
hash-checked copies can preserve values without reserializing or linking mutable
files into the protected snapshot. A materializer must additionally bind each
output part to its source unit and retained raw-row positions, verify the
12/4/4 assignment and support totals, and reject checkpoint/source drift. This
materializer preserves the approved counts: 1,223,347 TRAIN, 432,067 validation
and 444,465 test rows. The batch has a 900-second deadline, 4 GiB worker ceiling,
5 GiB artifact cap and the existing disk floors; available host RAM is telemetry
only. It performs no model fitting or scoring and is scheduled after DOC finishes.

The [materializer](../experiments/cicids_partition/materialize.py) and
[generated-data tests](../tests/test_cicids_materialize.py) are implemented.
All 11 focused tests passed, covering interrupted copies and checkpoint
publication, independent source/output files, corruption rejection, exact native
raw-label offsets, support and deadline enforcement. The metadata definition
reproduces all 20 approved units and the exact split counts.

Materialization completed on 2026-11-20 in **31.518 seconds**, with peak worker
RSS 189,419,520 bytes and peak observed artifacts 243,971,253 bytes. All 20 copied
parts match their reconciled source hashes and are independent files. Independent
verification checked all 2,099,879 retained raw-row identities, their bounds and
uniqueness across splits, and the current raw-source hashes. This establishes
source-row separation, not independence of feature-identical observations or
attack executions. No classifier was fitted or scored.

[Compact completion and provenance](../experiments/cicids_partition/MATERIALIZATION_SUCCESS.json),
SHA-256 `266b0c46ac5062e0c773e6176a6ad64b72ac00b0f369737c0e0dea296aa88eea`,
records the exact support totals, assignment and per-part bindings. The required
repository checks are running for this new materializer.
