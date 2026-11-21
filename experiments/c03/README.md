# C03 paired validation-information pilot

This folder holds the local, prospective C03 tools, scientific policy and
synthetic acceptance suites. The pilot and full campaign are not execution-frozen.

## No-fit producer foundation (Batch 1)

The Batch 1 modules add metadata-only planning, injected train/validation reader
interfaces with separate sample identities, the five-component dependency map,
and immutable checkpoint contracts. The command-line entrypoint remains no-fit;
Batch 2 adds the synthetic-only lifecycle described below.

From the repository root:

```powershell
# Read-only metadata plan; no observations opened or directories created
.\.venv\Scripts\python.exe -B -m experiments.c03.producer
# Explicitly create a new plan-only run in the approved external location
.\.venv\Scripts\python.exe -B -m experiments.c03.producer --initialize my-run-id
# Verify existing run/configuration and any completed synthetic checkpoints
.\.venv\Scripts\python.exe -B -m experiments.c03.producer --resume my-run-id
.\.venv\Scripts\python.exe -B -m pytest tests/test_c03_producer_foundation.py -p no:cacheprovider
```

The approved artifact root is the sibling `VERA-IDS-local-runs/c03/` directory.
Existing run IDs cannot be overwritten. Source/environment changes block resume.
An interrupted component blocks implicit reuse. Explicit recovery can copy verified
completed work into a fresh attempt (see below); the failed attempt is preserved.
OS-owned locks release on process exit. Legacy sentinel locks still require review.
Complete components are opaque bytes with checksums; Batch 1 does not load models.
Input hashes are caller declarations until a future production feeder establishes
them from actual tensor/row bytes. The stores are single-writer, local contracts,
not a sandbox or tamper-proof signature system.

Publication uses flushed temporary files and atomic no-clobber hard links, tested
on local Windows NTFS. A filesystem without hard-link support fails closed.
Directory metadata power-loss durability is not asserted. The producer subfolder
and its tests have separate provenance; the earlier relocation snapshot remains
unchanged. See the [implementation contract](../../docs/RECKONING_C03_SCORE_PRODUCER_CONTRACT_DRAFT.md).

## Validation production and sealing (Batch 2)

`producer.validation.produce_validation` connects verified train/validation
Parquet files to the actual historical preprocessor, sampler, model constructors
and weighting helpers. It materializes shared CSR tensors and source-row IDs,
fits five sequential components (one shared RF Stage-2), persists model and score
receipts, then calls the existing selector and publishes both arms in one object.
`producer.sealing.verify_seal` verifies code, input/model/score checksums, both
arms and the final seal without resolving any test paths.

Execution is deliberately **synthetic-only**: generated fixtures outside this
repository, at most 10,000 rows per source split, at most eight trees per model,
one worker, CPU XGB, and an explicit synthetic plan. These test-only caps do not
replace the scientific policy. The CLI has no benchmark fitting command. Seals
record `evaluation_authorized=false`: they cannot authorize scientific test
evaluation. Batch 3 adds a separate, prospective synthetic evaluation contract.

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_c03_producer_validation.py tests/test_c03_producer_foundation.py -p no:cacheprovider
```

The tests generate disposable Parquet sources and tiny fitted RF/XGB checkpoints;
they do not use benchmark observations. Native XGB serialization and local
joblib RF serialization are checked before reloading. Hashes detect drift, not
malicious replacement of both files and their receipts; do not accept arbitrary
downloaded model files as local checkpoints.

Use `produce_validation(..., resume=True)` for a complete synthetic run: input
artifacts and scores are verified and models are not refitted. Incomplete stages
fail closed and require a new run. Existing Batch 1 run files and provenance are
preserved. Source changes intentionally make the older plan fail current resume;
do not rewrite it. `BATCH2_PROVENANCE.json` records the successor implementation.

## Separate post-seal evaluator (Batch 3)

`producer.evaluator` verifies the complete selection seal before resolving test
paths or loading models. A new synthetic plan must bind the ordered test-file
inventory, byte hashes, row counts and evaluator revision before selection is
sealed. Older seals without that contract cannot be upgraded. Test inputs remain
external generated fixtures, capped at 10,000 rows; no benchmark mode exists.

```powershell
# Only for an already sealed synthetic run with a prospective evaluator contract
.\.venv\Scripts\python.exe -B -m experiments.c03.producer.evaluator --run EXTERNAL_RUN_DIRECTORY
.\.venv\Scripts\python.exe -B -m pytest tests/test_c03_producer_evaluator.py tests/test_c03_producer_validation.py tests/test_c03_producer_foundation.py -p no:cacheprovider
```

Both arms consume the same ordered test population, including when they select
different profiles. The evaluator writes `evaluation.json` and a final checksum
receipt without overwriting an existing result. It retains fixed seven-class
confusion matrices, macro-F1, accuracy, explicit rate numerators/denominators,
Stage-1 attack AUROC/average precision and visible-minus-blind effects for all
16 budget/method pairs. Unsupported populations and infeasible paired effects
remain undefined with reasons. No uncertainty estimates or scientific claims
are produced. `BATCH3_PROVENANCE.json` records this implementation separately.

## Layout

The approved TRAIN-only preparation readiness probe is separate from the earlier
small RF probe and from scientific execution:

```powershell
.\.venv\Scripts\python.exe -B -m experiments.c03.engineering.preparation_probe --inspect
.\.venv\Scripts\python.exe -B -m experiments.c03.engineering.preparation_probe --probe
.\.venv\Scripts\python.exe -B -m pytest tests/test_c03_preparation_probe.py -p no:cacheprovider
```

It verifies the 239 TRAIN sources, fits the historical preprocessor and preserves
the 1.5m-row sampling/strict supervised masks, without classifier fits or validation/
test reads. It requires 6 GiB available RAM and 20 GiB free disk before launch;
sampled stops are 4 GiB worker-tree RSS, 2 GiB available RAM, 5 GiB artifacts,
10 GiB free disk and 900 seconds. Descendants are terminated and reaped on stops
or monitor failures. These are sampled thresholds with possible transient
overshoot. A refused start does not create a worker; no automatic retry occurs.
Fresh external directories preserve diagnostics and cannot serve as scientific
checkpoints. Full training remains subject to a separate execution review.

The preparation probe now calls the unchanged historical transform on 8,192-row
blocks. A counting pass sizes the final CSR arrays exactly; a second pass fills
them without keeping a list of chunk tensors or a full selected feature frame.
Both final training tensors remain live. Differential tests compare the complete
CSR byte representation, row identities and labels against the original path.
The shared implementation now lives in `producer/transforms.py` and is used by
the synthetic producer for both training tensors and validation. The engineering
probe uses the same function. The protected historical transformer and all
resource limits remain unchanged. Source changes require fresh plans; do not
rewrite old seals or promote engineering artifacts to scientific checkpoints.

The separately approved combined preparation probe exercises the producer's
TRAIN/validation preparation, serialization and verified reload in an explicit
`ENGINEERING_TRAIN_VALIDATION_PREPARATION_ONLY` namespace:

```powershell
.\.venv\Scripts\python.exe -B -m experiments.c03.engineering.combined_preparation_probe --inspect
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.combined_preparation_probe --probe
.\.venv\Scripts\python.exe -B -m pytest tests/test_c03_combined_preparation_probe.py -p no:cacheprovider
```

This verifies 239 TRAIN and 65 validation sources, keeps the approved 1.5m/700k
samples, prepares and saves all three tensors, releases the original arrays and
reloads them with identity checks. Classifier fits, selection, test data and GPU
work remain excluded. The same 900-second, 6-GiB start, 4-GiB worker, 2-GiB free
RAM and 5-GiB artifact limits apply. Diagnostic row-level arrays stay in a fresh
external run directory and cannot authorize scientific fitting or evaluation.
Incomplete runs stay incomplete; there is no automatic retry or reuse.

Both preparation probes print the run directory and live progress to stderr:
stage changes and a heartbeat every 30 seconds show elapsed time against the
time budget, worker RSS, available RAM and artifact size. The final JSON remains
on stdout. These are observed measurements, not completion percentages or ETAs.

## Approved full-row RF sizing probe

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.rf_stage2_sizing_probe --probe
# Verify an already completed result without refitting
.\.venv\Scripts\python.exe -B -m experiments.c03.engineering.rf_stage2_sizing_probe --resume EXTERNAL_PROBE_DIRECTORY
```

This engineering-only probe imports the verified combined preparation's Stage-2
array (1,201,422 rows, 46 float64 features), with pinned original receipts and
byte/tensor/label/row/weight hashes. It fits eight RF trees with one worker and
seed 133, retaining timing and tree-storage diagnostics only. It performs no
validation/test scoring and retains no model. The same 900-second and memory/disk
limits apply, including loading and verification. Stage changes and 30-second
resource heartbeats appear on stderr. A completed result is reusable; an
interrupted fit has no mid-tree checkpoint and requires a fresh authorized attempt.
Ordinary producer resume still requires the same source revision. This explicit
import neither promotes diagnostic arrays nor certifies full scientific fitting.

The separately approved two-worker comparison uses this fixed command:

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.rf_two_worker_sizing_probe --probe
# Verify a completed two-worker result without fitting again
.\.venv\Scripts\python.exe -B -m experiments.c03.engineering.rf_two_worker_sizing_probe --resume EXTERNAL_PROBE_DIRECTORY
```

It pins the successful one-worker baseline, requires identical imported inputs
and effective model settings except `n_jobs=2`, and retains timing/tree-storage
comparisons. Limits, progress, failure preservation and completed-result recovery
are unchanged. A single timing comparison does not establish stable speedup or
full-model feasibility; equal tree sizes do not prove identical fitted models.

The approved Stage-1 batch measures both historical RF weighting profiles with
eight trees, two workers and seed 123 each, in fresh sequential processes. Its
900-second budget covers both profiles together. It imports only the verified
Stage-1 TRAIN array; the native class-weight and inverse-family-weight paths stay
distinct. The same RAM/disk gates apply before each worker; no model or scores
are saved.

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.rf_stage1_sizing_probe --probe
# Verify a completed batch without fitting
.\.venv\Scripts\python.exe -B -m experiments.c03.engineering.rf_stage1_sizing_probe --resume EXTERNAL_BATCH_DIRECTORY
# Only after approval for a new attempt: preserve completed profiles and restart unfinished work
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.rf_stage1_sizing_probe --fork EXTERNAL_BATCH_DIRECTORY
```

Each successful profile commits a separate diagnostic receipt. Explicit recovery
copies and verifies completed diagnostic files into a fresh run, preserves failed
attempts, and refuses active locks or corrupted completed work. It does not reuse
an interrupted fit or grant another time budget automatically. If no profile
completed, a fresh approved probe is needed. This is engineering evidence, not
a scientific checkpoint or full-forest resource qualification.

## Synthetic CUDA readiness

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.cuda_readiness_probe --probe
# Completed-result verification does not require GPU execution
.\.venv\Scripts\python.exe -B -m experiments.c03.engineering.cuda_readiness_probe --resume EXTERNAL_BATCH_DIRECTORY
# Only after approval for a new attempt: reuse completed component diagnostics
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.cuda_readiness_probe --fork EXTERNAL_BATCH_DIRECTORY
```

This separate scope fits only generated data: binary and five-class XGBoost
objectives, 4,096 rows/46 CSR features each, eight boosting rounds, two CPU workers
and CUDA device 0. Fresh sequential workers share a 300-second budget. It starts
with at least 3 GiB available host RAM and 6 GiB free VRAM; stops above 2 GiB
worker RSS, below 2 GiB available RAM, above 4 GiB whole-device used VRAM or below
2 GiB free VRAM. Disk gates are 2 GiB start/1 GiB floor, with 64 MiB new artifacts.
These synthetic limits do not change any full-row probe limits.

GPU queries must succeed within two seconds. The approved single-NVIDIA-GPU
setup is required; ambiguous device enumeration/remapping is rejected. Fitted
booster configuration is checked after each round and at completion; CPU fallback
is rejected. Progress includes sampled host and device-wide GPU memory. Diagnostic
configuration, warnings, traces and receipts are retained, but models, generated
arrays, predictions and quality metrics are not. A completed component can be
reused only after verification; failed fits require an explicit fresh attempt.
This measures CUDA runtime compatibility, not benchmark capacity or performance.

## Full-row XGBoost sizing

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.xgb_full_row_sizing_probe --probe
# Verify completed diagnostics without fitting or GPU queries
.\.venv\Scripts\python.exe -B -m experiments.c03.engineering.xgb_full_row_sizing_probe --resume EXTERNAL_BATCH_DIRECTORY
# Only after approval for another attempt: reuse completed component diagnostics
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.xgb_full_row_sizing_probe --fork EXTERNAL_BATCH_DIRECTORY
```

The approved engineering batch imports only each component's verified TRAIN
array: Stage-1 (1,270,025 rows) then Stage-2 (1,201,422 rows), 46 float64 features,
original weights and seeds 123/133. It fits eight boosting rounds per component,
with two CPU workers and CUDA device 0, in fresh sequential processes under one
900-second budget. Eight five-class rounds mean 40 individual trees.

Full-row limits apply: 6 GiB available host RAM to start each worker, 4 GiB worker
RSS ceiling and 2 GiB available floor; GPU 6 GiB free to start, 4 GiB whole-device
used ceiling and 2 GiB free floor; disk 20 GiB to start, 10 GiB floor, 5 GiB total
new artifacts. Source/preparation/CUDA evidence and tensor/label/row/weight hashes
are checked. CPU fallback, changed GPU identity and missing telemetry fail closed.
Progress and completed-component recovery are retained; no automatic retries.
No validation/test arrays, scores, quality metrics, prepared-array copies or
fitted models are retained. Completion qualifies only these short engineering
fits, not the full scientific 800/1,400-round models or scoring lifecycle.

## Isolated workers and explicit recovery

The synthetic component runner starts a fresh process for each of the five model
components. Each loads only its own training stage and the common validation
arrays. Parent monitoring covers the process tree; component locks prevent
recovery while an orphaned worker still owns that component. Progress, resource
traces and per-attempt results are retained. The current runner permits only
tiny synthetic CPU plans; it cannot fit benchmark models.

```powershell
# Continue a same-revision synthetic run with a complete preparation checkpoint
.\.venv\Scripts\python.exe -B -m experiments.c03.producer.isolated --resume EXTERNAL_RUN_DIRECTORY
# Explicitly preserve completed work and retry interrupted components in a new run
.\.venv\Scripts\python.exe -B -m experiments.c03.producer.isolated --fork EXTERNAL_RUN_DIRECTORY --run-id recovery-01
# Resume an engineering preparation checkpoint in a fresh diagnostic attempt
.\.venv\Scripts\python.exe -B -m experiments.c03.engineering.combined_preparation_probe --resume EXTERNAL_PROBE_DIRECTORY
```

Recovery verifies source/environment, plan, input, model and score hashes before
reuse. Corrupt completed checkpoints are rejected. Recovery forks retain the
original scope and record their parent and copied-file hashes; incomplete fork
copies cannot run. No deletion, overwriting, automatic retry or scope promotion
occurs. A fully written frozen selection can recover its missing final receipt
without repeating selection or changing decisions.

Checkpoints are at **complete preparation and complete model-component boundaries**.
An interrupted fit restarts that component from the beginning; no warm-start or
mid-tree/mid-boosting continuation is claimed. An interrupted preparation without
its final receipt must restart. Old code revisions are not silently upgraded;
earlier probe artifacts remain archival engineering evidence. Atomic writes and
OS locks handle tested process failures, but do not promise filesystem/power-loss
durability. Keep sufficient disk space for the retained source and recovery copy.

- `c03_selector_reference.py`, `c03_paired_runner_core.py`: validation selectors.
- `c03_*gate.py`, `c03_runner_disk_boundary_smoke.py`, `test_c03_*.py`:
  model-free integration and acceptance checks.
- `c03_candidate_fit_contract.py`, `c03_feature_proxy_preflight.py`,
  `c03_source_file_assignment_audit_v3.py`: candidate and source preflights.
- `c03_source_unit_integrity_audit.py` and its `v2` successor: retained earlier
  audit attempts; v3 handles the actual legacy file-assignment schema.
- `c03_input_content_snapshot.py`: byte inventory and relocation verification.
- `c03_rf_bounded_memory_probe.py`: bounded engineering memory measurement.
- `engineering/guarded_memory_probe.py`: use this supervisor for the authorized
  bounded probe on Windows. The original supervisor monitors only the virtual
  environment launcher PID; this one includes and terminates worker descendants.
  Engineering provenance and supervisor tests are recorded separately in that
  folder; they are not covered by the earlier relocation snapshot.
- `C03_APPROVED_SELECTOR_POLICY.json`: unchanged approved scientific policy.
- `docs/`: historical review, acceptance report and handoff, preserved verbatim.
  Their old root-level command examples are historical; use commands below.
- `RELOCATION.json`: original-to-current file hashes and the successor snapshot.
- `../../.release-audit/c03/`: ignored local manifest, logs, exact pre-move
  backup (`layout-before/`) and relocated input snapshot. Not public evidence.

## Commands

Run from the repository root using its existing Python 3.12 environment.
Standalone scripts still expect the repository working directory.

```powershell
# No-fit candidate and selector preflights
.\.venv\Scripts\python.exe -B experiments/c03/c03_candidate_fit_contract.py
.\.venv\Scripts\python.exe -B experiments/c03/c03_paired_runner_core.py --preflight --verify-bytes

# All 47 existing synthetic cases, including the historical Stage-1 adapter
.\.venv\Scripts\python.exe -B -c "import sys,unittest; sys.path.insert(0,'experiments/c03'); unittest.main(module=None,argv=['c03-tests','-v','test_c03_selector_reference','c03_stage1_integration_gate','c03_pretest_sealing_gate','c03_runner_disk_boundary_smoke','test_c03_paired_runner_core'])"

# Memory inspection only; does not start the probe or fit anything
.\.venv\Scripts\python.exe -B experiments/c03/c03_rf_bounded_memory_probe.py --inspect
```

Do not launch `--probe` as a cleanup check. Its original resource and execution
boundaries still apply; the complete scientific model fit remains separate.

## Provenance after relocation

The original input manifest remains byte-for-byte unchanged in ignored local
storage. It includes the original root-level paths and code hashes. Path updates
to a few tools necessarily produce new code hashes. The relocation record pins
the separate successor snapshot; verification checks identical data entries,
unchanged scientific policy and historical package sources, and exact current
tool bytes. It does not claim that edited tools match their historical hashes.

The inventory includes the new path helper and all current C03 Python files and
historical C03 notes. A subsequent code change requires a documented successor
snapshot; do not silently replace the original baseline or relax a failed gate.
No training, scientific results or model-performance claims are created by this
organization change.
