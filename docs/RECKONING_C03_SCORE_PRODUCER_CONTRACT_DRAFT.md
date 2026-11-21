# C03 score producer and sealed evaluator — reconstructed review draft

Updated: 2026-11-18. **The user authorized autonomous continuation of the fixed
C03 pilot, including necessary full-size fitting. The broader scientific campaign
remains unfrozen. The producer implemented below has no selection or test reader;
those require their own prospective, verified execution contract.**

The full producer completed all eleven preparation/fit/score checkpoints in
15,977.139 seconds (4 h 26 m 17 s), with independent completion verification on
November 18. Peak worker RSS was 3.6918 GiB, peak whole-device GPU usage 0.9492 GiB,
and retained artifacts 6.4344 GiB. Each of the five full-size models scored the
same 700,000 validation rows. These are execution results, not test-performance
claims. The fixed producer plan remains
`6b54cfd10448ff4f4cf9cc601e3269debe14f95e78999b0378ffba955d057bdd`.

The November 18 continuation adds `experiments/c03/postseal/`, a separate
prospective contract that binds the completed producer, approved policy, evaluator
revision and original test-file inventory before selecting either arm. Both
decisions are published together and sealed before any test source is resolved.
Selected components are reloaded in isolated workers; test features are transformed
in 8,192-row batches without refitting. Every component must reproduce the same
ordered source-row fingerprint, labels and class support. The existing fixed
seven-class metrics and visible-minus-blind effects are retained unchanged.

This stage has one shared 3,600-second ceiling and a 5 GiB new-artifact budget,
with the existing 4 GiB worker-RSS, GPU and disk guards and available-RAM telemetry.
Recovery reuses completed component scores under the original deadline. A partial
selection or analysis publication can finish its final receipt only after
verification; existing decision/result bytes cannot be overwritten. The original
producer plan and its completion records remain immutable. Test execution follows
generated-data verification and repository checks; no full-campaign or inferential
claim is added by this fixed-case evaluation contract.

2026-11-17 execution-authority update: the user removed available-RAM start and
runtime gates and authorized autonomous necessary experiments to advance the plan.
Earlier numerical RAM gates and per-attempt approval statuses below describe
their historical batches. Successor work retains worker-RSS, GPU, disk, time and
artifact limits; available RAM remains telemetry. Necessary fixed-protocol C03
work may advance without repeated operational permission. Unresolved scientific
choices and new campaign/dataset definitions still require clarification before
execution. Active agent work ends at exhaustion or reset of the current Codex
five-hour quota; already-started bounded jobs may finish and checkpoint.

The successor lifecycle rehearsal completed all ten fit/score phases in 515.889 s:
five eight-tree/round models, independently reloaded, each scoring 700,000 validation
rows. Peak sampled worker RSS was 1,984,114,688 bytes, and peak whole-device VRAM
979,369,984 bytes. Both XGBoost reload/score phases verified CUDA execution.
`experiments/c03/engineering/LIFECYCLE_RAM_POLICY_SUCCESS.json` preserves the
source and checkpoint hashes; the original November 14 provenance is unchanged.

The fixed scientific producer is `python -m experiments.c03.pilot.runner --run`.
It performs fresh preparation and requires its numeric tensors, labels, weights
and row identities to match the previously verified historical recipe. It uses
three 300-tree RF components and the historical 800/1,400-round XGBoost components,
two CPU workers, GPU 0, and seeds 123/133. Its operational envelope is ten hours,
15 GiB new artifacts, 30 GiB free disk at start, and the retained 4 GiB worker RSS,
10 GiB disk floor and original GPU limits. The larger artifact envelope accounts
for fresh preparation plus full models; the deadline is a ceiling, not a runtime
promise. Available RAM is telemetry only. Completed fit/score receipts are reusable
within the original deadline; interrupted fits restart and interrupted preparation
requires a new run. The producer ends with validation score checkpoints, before
selection. Engineering models are not promoted into scientific results.

This reconstructs the missing 2026-10-25 design contract from the approved C03
policy, canonical decision log and current pinned source. It is not the original
file and does not claim its hash or exact contents. Proposed implementation
choices for later batches below require review. No new method, threshold, sample cap or experiment
is adopted by this document.

## Basis and existing evidence

- Canonical scope: [Reckoning master](RECKONING_MASTER_CAMPAIGN_PLAN.md).
- Latest decisions: [decision log](https://docs.google.com/document/d/1BNNXb3zCEvhKLsmnn8K-4UOR9GaZrqhYQ5SEcnZ9Amk/edit).
- Approved policy: `experiments/c03/C03_APPROVED_SELECTOR_POLICY.json`, SHA-256
  `bf56f938d1fb193d2b383deeeec29f9d84759a4be829f6a0011f1bc20fe5e099`.
- Source revision: `ca13f1097f851ce6693658ccc0c6dd35480fabd6`.
- Original input snapshot: `13e667a86a6872ff6192d1662085383e2b526752ceb783f8fc372b51487de9f4`;
  subsequent layout provenance is separate in `experiments/c03/RELOCATION.json`.
- 47 existing synthetic selector/integration cases passed after relocation.
  They do not validate production fitting, model checkpoint reload or real
  test-reader separation.
- On 2026-11-06 the authorized TRAIN-only engineering probe completed 8-tree RF
  fits at 30k and 80k rows in 1.59 s and 6.14 s; tree storage 1.75 and 3.58 MiB.
  Peak sampled combined process-tree RSS: 0.599 GiB; minimum sampled available
  physical RAM: 3.788 GiB; overall wall time 14.52 s. These are small-sample
  measurements, not certification of a 1,201,422-row, 300-tree fit. No quality
  metrics, validation/test observations or persistent models were produced.

## Scientific invariants already approved

One pilot: CICIoT2023, fixed prepared Protocol B partition, Botnet held out,
model seed 123. The intervention changes validation-observation availability:
blind K versus visible K+U. Both arms share the fit, candidate universe, known
validation observations and ordered test population.

| Surface | Required identity/count |
| --- | --- |
| Prepared training | 239 files; 3,676,338 rows |
| Shared capped training | 1,500,000 rows; sampling seed 123 |
| Strict Stage-1 supervised training | Expected 1,270,025 rows; exclude Botnet attacks |
| Stage-2 supervised training | Expected 1,201,422 represented-attack rows |
| Shared capped validation | 700,000 rows; sampling seed 124 |
| Validation views | K 599,278; U 100,722 |
| Post-seal test | Full identical ordered 1,190,899 rows |

Counts above are expectations recorded in prior audits, not recomputed by this
draft. Verify them at execution and stop on mismatch. Never silently drop a
family, resample until support improves, or change the seven-class universe.

Candidate IDs remain `xgb_inv_family_clipped`, `rf_inv_family_clipped`, and
`rf_class_weight_balanced`. Load exact archived parameter dictionaries through
the existing candidate contract. There are five unique fitted components:
three Stage-1 models with seed 123, one XGB Stage-2 and one shared RF Stage-2
with seed 133. Preserve original weighting and model-constructor semantics.
RF native class weighting is determined by the constructor, not guessed from
the profile's human-readable name.

## Preprocessing and sampling requirements

Use the historical training-stream preprocessing procedure, including its own
statistics cap, first-part feature inference, part order and RNG. This is not
the engineering probe's sample-fitted scaler. In the current helper,
`fit_preprocessor` infers from the first part's first 5,000 rows, accumulates
statistics until `max_stat_rows`, and may then sample the accumulated numeric
rows. Record the effective helper configuration and exact source-row coverage.
Do not describe this as using every training row. It is also not an H-excluded
representation: historical unsupervised exposure to Botnet is retained.

The historical `collect_split_frame` repeatedly combines its current buffer with
the next part and downsamples using sequential RNG draws. It is explicitly not
a uniform reservoir. Preserve its exact order and draws for the 1.5m/700k caps.
Attach source-part digest and original row offset through a separate identity
channel; never include those identifiers as predictors. Fixture comparisons
must show that identity tracking leaves sampled feature/label values and order
identical to the unmodified loader.

Freeze feature names/order, dtypes, sparse format, class encoding and numeric
transforms. Do not downcast, change clipping, substitute a sampler or calibrate
probabilities for memory savings. Such changes need separate scientific review.

## Proposed execution architecture

1. **Inspect/plan (no fit by default).** Verify source, policy, environment,
   resource configuration and requested namespace. Emit an execution plan only;
   do not open benchmark observations, load serialized models or start fitting.
2. **Prepare and produce validation scores (explicit later execution).** Fit
   one preprocessor; derive samples and support ledger; build Stage-1 and shared
   Stage-2 tensors. Fit each component sequentially, checkpoint it, score the
   common ordered validation population, release in-memory model state, and
   verify score provenance. No test observations or score files are accessible
   through the production loader at this stage.
3. **Select and persist both arms.** Supply all three candidate validation
   bundles to the existing core. Validate source-row identity in addition to
   its current position-vector hashes. Persist complete decisions for both arms,
   all four methods and all four budgets, including infeasibility.
4. **Separate post-seal evaluator process.** Verify the persisted seal and every
   referenced policy/code/score/model identity before resolving test paths.
   Then score the same full ordered test population for each distinct selected
   candidate, apply each arm's frozen decisions and produce paired outputs.

Never call historical `run_one_combo`: it loads test at lines 966–968 before
validation selection. Reuse individual audited helpers only. Application-level
loader restrictions plus process separation are not an OS access-control sandbox;
do not advertise them as one. Content hashing of frozen source files is a
distinct byte-provenance operation, not permission to decode test observations.

## Proposed checkpoint, resume and seal contract

Use an explicitly selected, local output root outside the repository, containing
immutable run IDs. Reject existing run IDs unless an explicit verified resume is
requested. Do not overwrite historical outputs or reuse historical model files.

Each completed component records input row/tensor/label/weight hashes,
preprocessor hash, effective constructor parameters, seed, environment,
code identity and output checksums. The two RF profiles reference exactly one
stored Stage-2 tensor and fitted component; equal parameter dictionaries alone
are insufficient. Resume checks all identities before reuse; interrupted or
checksum-invalid components remain incomplete and cannot unlock evaluation.

Proposed formats: JSON manifests, non-object NumPy arrays with pickle disabled,
sparse arrays with explicit dtype/shape metadata, XGB native serialization, and
locally produced sklearn/joblib checkpoints only after provenance verification.
No third-party or historical pickle/joblib is loaded. Exact checkpoint format
and trust checks are part of the implementation review.

Write each artifact to a temporary file on the same volume, flush it, then
publish it atomically. Write the final both-arm seal last. It binds decisions,
validation score/input hashes, profile mapping, policy, source inventory,
producer/evaluator revisions and environment. A content digest detects drift;
it is not a cryptographic identity signature. A changed decision or provenance
requires a new run, not silently updating the seal.

## Evaluator outputs and interpretation

Retain per-arm, per-method, per-budget feasibility and selection traces; ordered
test identity/label checksums; exact denominators; confusion matrices; UDR;
known-all and known-attack FUR; benign-to-family FPR; benign rejection; composed
macro-F1; accuracy; realized rejection; and score-defined AUC metrics only with
explicitly appropriate score/positive-class definitions.

Use fixed seven-class semantics and approved rejection inequalities/transforms.
Numerical paired effects are visible minus blind. If either arm is infeasible,
report the feasibility transition and an undefined numeric delta with reason.
Do not impute a fallback score or use test outcomes to revise selection.

This single fixed-case pilot does not establish population-level causality,
capture independence, absence of feature proxies or cross-environment stability.
Row support and disjoint source-path metadata are not independent captures.
No new inferential claims or confidence-interval policy is introduced here.

## Resource boundaries

Use sequential processes and process-tree monitoring. The original probe's
single-PID monitoring missed the Windows virtual-environment redirector's child;
the engineering supervisor corrects that while preserving the original worker.
Summed RSS can double-count shared pages and is a conservative sampled measure.

Production RF/XGB resources are still unvalidated. The small probe's 3 GiB
worker limit is not a proposed full-fit limit. Device, worker count, physical
memory floor, per-component limit, wall-time budget and checkpoint disk budget
must be reviewed before fitting. No automatic sample/tree/depth reduction,
device fallback, or larger experiment follows from the successful small probe.

## Proposed implementation batches and acceptance

**Batch 1 — no-fit producer foundation (approved 2026-11-06).** Implement explicit
configuration and provenance validation, isolated loader interfaces, sample-ID
tracking, component DAG and checkpoint state machine using synthetic fixtures.
No real preprocessing fit, benchmark scan, model fit or test evaluation.
Acceptance: exact sampler equivalence on fixtures, no test-loader access,
five-component shared-RF mapping, corruption/interruption/resume rejection,
source/config drift rejection, and no-fit CLI default.

**Batch 2 — production score generation and persistent sealing (approved; synthetic verification).** Implement
historical transforms/models and validation scoring, but exercise with synthetic
fixtures until separate resource and execution approval. Acceptance includes
shared tensor/model identity, actual fitted-class probability-column alignment,
hidden-U invariance of blind decisions and atomic both-arm seal behavior.

**Batch 3 — separate evaluator (approved; synthetic verification).** Test missing/tampered seal rejection before
any test-loader call, misaligned/reordered rows, distinct selected profiles,
infeasible cases, signed deltas and exact metric denominators with synthetic data.

**Execution review.** Reconcile source/environment/provenance and all acceptance
evidence; approve machine/resources, output location and one-cell fit separately.
Expansion to other holdouts/seeds remains under the campaign freeze process.

## Decision status

1. User approved Batch 1's no-fit scope and checkpoint/resume foundation.
   User subsequently directed proceeding with Batches 2 and 3 and resuming after
   interruption. This does not authorize fitting/scoring benchmark observations.
2. User approved sibling `VERA-IDS-local-runs/c03/` for local artifacts.
   Batch 1 may initialize a plan-only run there; synthetic tests use temporary
   directories. No real model artifacts are produced by this batch.
3. Full-fit hardware and resource limits remain deferred; no choice is inferred
   from the small memory probe. Any larger sizing experiment needs its own scope.

## Batch 2 implementation boundary

The producer now derives tensor/row/label/weight hashes from actual prepared
fixture arrays, persists them alongside validation truth and ordered IDs, uses
historical model constructors and serializes five models, reuses the one RF2
component, verifies fitted probability-column identities, and persists both-arm
selection plus a final content seal. Only generated synthetic inputs have been
executed. The benchmark execution entrypoint remains absent pending resource and
execution review; the plan-only CLI still performs no observation reads.

The Batch 2 seal is explicitly synthetic and cannot authorize test evaluation.
It binds producer revision, policy, preparation, models, scores and validation
identity, and records no evaluator revision. Batch 3 must provide and prospectively
bind its own evaluator version rather than upgrading an existing synthetic seal.

## Batch 3 implementation boundary

The separate evaluator accepts only a new synthetic seal that prospectively
binds its revision and ordered test source inventory. It verifies the seal before
test-path resolution, loads verified local model checkpoints, scores one shared
ordered test population, and publishes paired descriptive results plus a final
checksum receipt. Older synthetic seals remain unusable for evaluation.

Verification covers a separate-process invocation, missing/tampered seals before
test access, source and row identity drift, distinct selected profiles, exact
rejection boundaries, metric denominators and infeasible-arm deltas. The only
AUC outputs are explicitly defined Stage-1 raw attack-score AUROC and average
precision; absent binary classes produce undefined values. Full-fit resources,
process isolation and benchmark execution remain a separate review checkpoint.

## Execution review — 2026-11-07

Read-only reconciliation matched the current plan and every Batch 3 source hash:
`185025edbd9a1f154c3bf820fa3e01ef722c1ea2bc04bd6c526c392883dfd70a`.
The installed dependency versions still match the pinned environment. The nine
passing Batch 3 checks and 183-test result remain applicable: no implementation
source changed during this review. Estimators were constructed to inspect their
effective parameters; no observations were decoded and no fit was invoked.
Local evidence: `.release-audit/c03/execution-review-20261107.json` and its
hardware snapshot. Stored inventory sizes were read as metadata, not rehashed
from current data bytes.

### Observed machine and effective defaults

| Item | Observation |
| --- | --- |
| CPU | Intel Core Ultra 9 275HX; 24 physical/logical cores reported |
| OS-visible physical RAM | 15.43 GiB total; 2.53 GiB available at inspection |
| Output-volume free space | Approximately 463.66 GiB at inspection |
| NVIDIA GPU | RTX 5060 Laptop; 8,151 MiB total and 7,590 MiB free VRAM reported by `nvidia-smi` |
| Historical RF constructors | 24 workers; 300 trees; unlimited depth |
| Historical XGB constructors | CUDA, histogram trees, 24 CPU workers; 800/1,400 trees |
| Historical preprocessing | 600,000-row statistics cap; ordered-prefix semantics preserved |

Available resources are a time-specific observation, not a reservation. GPU
enumeration does not prove that the pinned XGB runtime can complete a CUDA fit.
The available RAM is below even the earlier small probe's 3 GiB start threshold.
Compressed input sizes do not bound expanded frames, CSR tensors, identities,
estimator copies or trees. No full-run memory/time estimate is justified yet.

### Remaining implementation and evidence gaps

- Production preparation holds train/validation frames, both training tensors
  and validation tensors together; this footprint has only been tested on tiny
  fixtures. The evaluator also materializes the complete test frame and tensor.
- Five component fits are sequential within one Python process. The agreed
  fresh-worker/process-tree resource controls are not wired into that lifecycle.
- The existing supervisor covers only the original fixed small RF probe; it is
  not a full producer supervisor or a hard OS memory limit.
- All executable producer/evaluator scopes remain synthetic. A scientific mode
  needs explicit resource settings, source verification, prospective evaluator
  identity and tests of guarded interruption/resume before execution review closes.
- The one-worker CPU fixture settings do not approve replacing historical
  24-worker/CUDA defaults. Worker/device choices for model fitting remain open.

### Bounded TRAIN preparation readiness (approved 2026-11-07)

Recommendation: first establish whether the unchanged historical preparation
fits this machine under a conservative operating envelope. The user approved
implementing and running this engineering probe on the laptop, subject to the
start gate and limits below. It is not a scientific model fit.

1. Add a separate guarded preparation-probe entrypoint and verify process-tree
   stop/cleanup and incomplete-output handling with disposable synthetic workers.
   Keep existing synthetic gates and all earlier evidence intact.
2. Use a fresh external probe directory under the already approved
   artifact root. Verify the 239 TRAIN source identities before decoding them.
   Use the exact historical preprocessor, 1,500,000-row sampler/seed 123, strict
   Botnet-exclusion masks, tensor dtypes and row identities. Do not read validation
   or test observations, fit classifiers or report model-quality metrics.
3. Record preprocessing/sample/tensor stage timings, support counts, tensor byte
   sizes and hashes, peak process-tree RSS, minimum available RAM, artifact bytes,
   and completion or the exact stop reason. Retain diagnostic evidence externally;
   probe output is not a reusable scientific checkpoint.

Approved limits (operational choices, not measured requirements):

| Control | Approved value |
| --- | --- |
| Start gate | At least 6 GiB available physical RAM and 20 GiB free output-volume space |
| Memory stop | Sampled worker-tree RSS above 4 GiB, or available physical RAM below 2 GiB |
| Time stop | 900 seconds for the supervised probe, including source verification |
| Disk stop | Probe artifacts above 5 GiB, or output-volume free space below 10 GiB |
| Concurrency | One probe worker; no classifier or GPU work |
| Monitoring | Check at approximately 0.3-second intervals; terminate/reap worker descendants on violation |

These are sampled stop thresholds; transient overshoot is possible. Refuse to
start below the gate rather than closing user applications. A successful result
establishes TRAIN preparation feasibility within this envelope only. Failure
produces a diagnostic result and stops: no silent resampling, dtype change,
tree reduction, enlarged budget or automatic retry. A gate refusal requires a
fresh resource check after the user frees memory or selects another machine.

The user selected this laptop for the bounded readiness batch. Full training,
validation scoring and post-seal benchmark evaluation remain outside this batch. Their resource budgets
and worker/device choices should follow measured preparation evidence.

Implementation: `experiments/c03/engineering/preparation_probe.py`, with a
separate immutable request, source/test hashes, stage log, resource trace,
worker result and supervisor completion receipt. Only the supervisor can publish
the final completion receipt, and no probe artifacts are scientific checkpoints.
Earlier producer/selector sources and provenance remain unchanged. The run
retains both TRAIN tensors and their IDs/weights while measuring preparation;
it does not establish the larger joint TRAIN/validation or test footprint.

Verification: 17 focused tests and 200 full-suite tests passed; all nine required
repository commands passed. Additive evidence is recorded in
`experiments/c03/engineering/PREPARATION_PROVENANCE.json`. The approved launch
attempt on 2026-11-07 returned `START_REFUSED`: 4.11 GiB RAM available, below the
6 GiB start gate. No worker or benchmark observation read occurred. Preparation
feasibility remains unmeasured; the same approved probe can be invoked after
memory is freed, with a fresh resource check and unchanged limits.

Manual follow-up reviewed on 2026-11-08: the start gate passed with 6.24 GiB
available RAM. After 354.06 seconds, the monitor stopped the worker at 4.1196 GiB
summed RSS; minimum system available RAM remained 3.2236 GiB. The stage ledger
reached `stage1_tensor` after TRAIN byte verification, historical preprocessing,
sampling and support checks. It never reached Stage-2 tensor preparation and
has no completion receipt. Request source hashes still match the verified probe.
Local review evidence: `.release-audit/c03/preparation-manual-review-20261108.json`.
The current path exceeds the approved worker envelope during Stage-1 preparation;
the next review should examine allocation/copy overhead before proposing another
run or changing resource limits. No classifier fit or scientific result occurred.

On 2026-11-08 the user directed proceeding with the allocation review. The
engineering probe replaces full-selected-frame transformation with two passes
of the unchanged historical transformer over 8,192-row blocks, allocating one
exact-size final CSR. No dtype, sampled position, mask, seed, feature transform,
weight or resource limit changes. Both final tensors remain live. Numeric and
categorical fixtures (including nonfinite values and uneven block boundaries)
must match full-transform data/indices/indptr bytes exactly. This adjustment is
confined to the probe; original producer and historical source bytes remain
unchanged. Prior stopped-run evidence is preserved under its original hashes.

Revised implementation verification: 26 focused tests and 209 full-suite tests
passed; all nine repository checks passed. The subsequent guarded launch was
refused with 4.27 GiB RAM available, so no full-size memory improvement has yet
been measured. Successor record:
`experiments/c03/engineering/PREPARATION_MEMORY_PROVENANCE.json`.

Successful manual follow-up verified on 2026-11-08: the revised TRAIN-only probe
completed in 360.89 seconds, peaking at 2.8859 GiB sampled worker-tree RSS, with
minimum available system RAM 4.0977 GiB. Request/source hashes, final receipt,
result/supervisor hashes and preprocessor artifact bytes were verified. Exact
counts: 3,676,338 source rows; 1,500,000 sampled rows; Stage-1 tensor 1,270,025 x 46
and Stage-2 tensor 1,201,422 x 46, both float64 and retained simultaneously.
Immutable evidence: `experiments/c03/engineering/PREPARATION_MEMORY_SUCCESS.json`.
This establishes TRAIN preparation feasibility within the approved envelope only.
It does not establish joint TRAIN/validation preparation, full RF/XGB fitting or
post-seal test-evaluation feasibility. No classifier fit or validation/test read
occurred; the probe artifacts remain ineligible as scientific checkpoints.

## Bounded-transform integration and next measurement — 2026-11-08

Following the user's direction to proceed, the verified block transform is now
shared in `producer/transforms.py`. The synthetic producer uses it for Stage-1,
Stage-2 and validation preparation; the engineering probe imports that same
implementation. Historical helpers and scientific settings remain unchanged.
Differential verification checks all three prepared tensors against the original
full-frame transform. The producer still accepts only synthetic execution plans.
Old plans, seals and successful probe evidence retain their original revisions;
this integration receives separate successor provenance.

Integration verification: 96 focused tests and 210 full-suite tests passed;
all nine required repository checks passed. The extracted function body is
identical to the successful probe implementation. Successor evidence:
`experiments/c03/producer/BOUNDED_TRANSFORM_PROVENANCE.json`.

### Combined preparation probe (approved 2026-11-08)

The next minimum useful measurement is the full preparation lifecycle with TRAIN
and validation resident together, including array serialization and verified
reload. Measuring only a separate validation tensor would miss their combined
footprint. The user approved implementation and one guarded attempt of this scope.
It excludes classifier fits, scoring, selection, test
observations and GPU work.

- Reuse the same preparation implementation through a dedicated, explicitly
  engineering-only entrypoint; never label benchmark preparation as synthetic.
  The scientific fitting/evaluator gates remain closed.
- Verify the pinned 239 TRAIN and 65 validation files before reading them.
  Fit preprocessing on TRAIN only, preserve the 1.5m/700k sample caps and seeds
  123/124, and enforce Stage-1/Stage-2 counts 1,270,025/1,201,422 and validation
  K/U counts 599,278/100,722. No new validation-driven choice is made.
- Use a fresh external run with immutable source/configuration hashes. Exercise
  simultaneous preparation, serialization and reload identity verification;
  release the original live arrays before reload. Keep ordered IDs, targets,
  weights, tensor hashes, stage timings, memory/disk traces and failure evidence.
  These diagnostic arrays are not eligible scientific checkpoints.
- Retain the existing limits: one supervised worker; 900 seconds including
  verification; >=6 GiB available RAM and >=20 GiB free disk to start; stop above
  4 GiB worker-tree RSS, below 2 GiB available RAM, above 5 GiB run artifacts or
  below 10 GiB free disk. Stop thresholds are sampled; no automatic retry or
  budget/parameter adjustment. Refuse the start if resources are insufficient.
- First exercise the new scope boundary and interruption/reload checks on
  synthetic fixtures. Then attempt the engineering measurement once, subject
  to the live start gate.

Success would establish combined preparation/serialization/reload feasibility
only. Model-worker isolation and sizing remain a later checkpoint. RF/XGB worker
counts, CUDA acceptance and full-fit time/memory budgets are still unresolved;
do not infer their approval from a successful preparation measurement.

Implementation: `engineering/combined_preparation_probe.py` uses the shared
producer preparation and reload functions under a dedicated engineering plan
and artifact scope. The CLI fixes the original plan and source inventory; it
has no root, sample, classifier or budget overrides. The existing synthetic
fitting/selection gates reject engineering plans. The supervisor binds its
completion receipt to the combined scope and verifies saved artifact hashes.
Resource accounting tolerates temporary-file removal and counts hard-linked
publication names once. Stage logs distinguish sampling, transformation,
serialization, receipt creation and reload.

Verification: 108 focused tests and 222 full-suite tests passed; all nine required
repository commands passed. The guarded attempt returned `START_REFUSED` with
4.62 GiB available RAM, below the unchanged 6 GiB start gate. No worker or benchmark
read occurred. Combined preparation feasibility remains unmeasured. Evidence:
`experiments/c03/engineering/COMBINED_PREPARATION_PROVENANCE.json`. The same approved
command can be run manually after freeing RAM, without changing scope or limits.

Successful combined follow-up verified on 2026-11-08: preparation, serialization
and identity-verified reload completed in 437.062 seconds. Peak sampled worker
RSS was 3.7078 GiB; minimum available RAM 3.2346 GiB; maximum observed artifacts
2.5792 GiB. Expected float64 tensor shapes were 1,270,025 x 46, 1,201,422 x 46 and
700,000 x 46. The request/source hashes, final receipt and all saved file hashes
were verified. Evidence:
`experiments/c03/engineering/COMBINED_PREPARATION_SUCCESS.json`.
This establishes the measured combined preparation/reload footprint only; model
fitting, scoring and test evaluation remain unvalidated and unapproved.

The user requested progress indicators for subsequent work. The shared probe
supervisor now prints stage changes and 30-second resource heartbeats to stderr,
keeping final JSON on stdout. It reports elapsed time, worker RSS, available RAM
and artifact size without inventing an ETA or completion percentage. No resource
limits or preparation semantics change, and no real-data rerun is required for
this display change.

## Resilience and isolated workers — 2026-11-08

User directed proceeding and explicitly requested fault tolerance, resume and
checkpoints. Implemented checkpoint-boundary recovery, preserving the historical
fit procedure: complete preparation and complete model components are reusable;
interrupted fits restart only that component in a fresh attempt. No estimator
warm-start or partial-tree/boosting continuation is introduced.

The synthetic runner launches one process per component and loads only its
training tensor plus common validation. OS-managed pipeline/writer/component
locks release when their owning process dies; a still-live component worker
blocks recovery. Legacy sentinel locks are not stolen. Explicit recovery forks
copy verified preparation and completed component bytes, retain incomplete
original attempts, preserve scope, and commit a fork receipt only after complete
verification. Source/environment changes, corrupted completion receipts and
unfinished recovery copies block reuse. Preparation now publishes its input
binding before its final completion receipt, closing the completed-but-unbound
crash window. Explicit resume can finish a missing final selection seal only
after validating the unchanged already-frozen decision object.

Combined preparation probes can explicitly resume a same-revision completed
preparation in a new engineering attempt, skipping resampling and transformation.
An unfinished preparation without its final receipt still requires a restart.
Process-interruption tests do not establish arbitrary power-loss or network-
filesystem durability. The isolated model runner remains synthetic-only; no real
classifier resource setting or fit was approved by this engineering work.

Verification: 93 focused tests and 235 full-suite tests passed; all nine required
repository commands passed. Tests include an actual killed component worker,
active-lock rejection, recovery of a completed prefix and restart of its interrupted
component, five isolated model processes, corruption rejection and receipt-only
seal recovery. Evidence: `experiments/c03/producer/RESILIENCE_PROVENANCE.json`.

### Approved RF sizing measurement — 2026-11-08

One TRAIN-only RF Stage-2 sizing probe: all 1,201,422 already-approved represented-
attack rows, 46 unchanged float64 features, seed 133, historical weighting/tree
settings except an explicit engineering cap of eight trees and one worker.
No validation/test scoring, selection, model-quality metrics or retained model.
This measures full-row fit cost and tree storage only, not certify the
300-tree scientific model or the historical 24-worker configuration.

Reuse of the successful combined probe's Stage-2 diagnostic array requires
an explicit engineering import record binding its original plan/source and
artifact hashes to the new probe, with full byte/input-identity verification.
It must remain engineering-only; do not weaken ordinary same-revision resume or
promote old probe arrays into scientific checkpoints. This avoids repeating the
seven-minute preparation merely to measure an engineering fit.

Approved limits remain 900 seconds including load/verification, >=6 GiB available
RAM and >=20 GiB disk to start, <=4 GiB sampled worker RSS, >=2 GiB available RAM,
<=5 GiB new artifacts and >=10 GiB free disk. One fresh supervised worker, stage
progress and retained failure evidence; no automatic retry or budget increase.
The user explicitly approved implementation and one attempt, including the new
real-data classifier fit and recorded engineering import across code revisions.
Implementation: `engineering/rf_stage2_sizing_probe.py`; import is pinned to the
immutable combined-success record, with original receipt/array verification and
tensor, label, row and weight identity checks. Historical model settings and
helpers must still match. No validation/test arrays are loaded. Completed probe
results can be verified with `--resume` without refitting; incomplete fits retain
their evidence and require a fresh, separately authorized attempt.

Verification: 52 focused tests and 247 full-suite tests passed; all nine required
repository commands passed. The single guarded attempt on 2026-11-08 returned
START_REFUSED with 3.25 GiB available RAM, below the unchanged 6 GiB gate. No worker,
training-array load or classifier fit occurred. RF sizing remains unmeasured.
Evidence: `experiments/c03/engineering/RF_STAGE2_SIZING_PROVENANCE.json`. The
approved command may be run manually when sufficient RAM is available.

Manual follow-up verified on 2026-11-08: all 1,201,422 Stage-2 rows and eight trees
completed in 543.297 seconds, including 531.078 seconds fitting. Peak sampled RSS
was 1.6390 GiB, with minimum available RAM 4.8420 GiB. All 1,620 monitor samples
stayed within the approved limits. Completion/source/import receipts, original
array bytes and diagnostic consistency passed verification; completed-result
resume succeeded without fitting. Tree node/value buffers totaled 50,826,256
bytes (48.47 MiB). No validation/test scoring or retained model. Evidence:
`experiments/c03/engineering/RF_STAGE2_SIZING_SUCCESS.json`.

Linear extrapolation suggests approximately 5.53 hours for 300 trees with one
worker and 1.775 GiB of tree buffers alone. These are estimates, not measured
full-fit performance or memory bounds; inputs, temporary allocations, estimator
overhead, parallel workers and scoring are additional. Stage-1 RF and XGB remain
unqualified.

User approved on 2026-11-08: one comparable engineering fit with eight trees,
two RF workers, seed 133 and the same verified Stage-2 import, unchanged remaining
model settings and unchanged 900-second/RAM/disk limits. Retain timing/storage
diagnostics only; no scoring, retained model or automatic retries. Compare with
the one-worker baseline, then reassess before any further experiment or full fit.
Implementation: `engineering/rf_two_worker_sizing_probe.py`, with a distinct
command and run prefix. The original one-worker code and result remain unchanged.
The comparison pins that result's hash, requires the same imported array identity
and checks every effective model parameter except the approved worker-count
change. Reports include a single-run fit-time ratio and tree-size comparisons;
these do not establish stable speedup or model identity. Completed results can
be verified without fitting; incomplete fits require a fresh authorized attempt.

Two-worker verification: 48 focused tests and 255 full-suite tests passed; all
nine required repository commands passed. The approved attempt on 2026-11-08
returned START_REFUSED with 3.48 GiB available RAM, below the unchanged 6 GiB gate.
No worker, array load or fit occurred. Evidence:
`experiments/c03/engineering/RF_TWO_WORKER_SIZING_PROVENANCE.json`. The approved
two-worker command can be run manually when sufficient RAM is available.

The manual two-worker follow-up completed and was verified on 2026-11-08.
Completion/source/import receipts, original array bytes, effective parameters
and all 736 resource samples passed review. Completed-result resume passed
without fitting. Evidence:
`experiments/c03/engineering/RF_TWO_WORKER_SIZING_SUCCESS.json`.

| Observed Stage-2 eight-tree diagnostic | One worker | Two workers |
| --- | ---: | ---: |
| Fit seconds | 531.078 | 233.781 |
| Total seconds | 543.297 | 245.297 |
| Peak sampled worker RSS (GiB) | 1.6390 | 1.7536 |
| Minimum available RAM (GiB) | 4.8420 | 4.2242 |
| Tree node/value storage (bytes) | 50,826,256 | 50,826,256 |

Fit time fell by 55.98% (2.2717x ratio), with 117.39 MiB additional peak RSS.
All sampled limits held. Inputs and effective model settings matched except
`n_jobs`; each tree's node count and storage matched, without scoring or retained
models. Equal tree sizes do not prove model identity. These are single-run
observations under potentially different background load, not a repeated or
controlled estimate of parallel speedup.

Linear extrapolation gives approximately 2.44 hours for the 300-tree Stage-2 fit
with two workers; it does not establish full-fit memory, checkpoint or scoring
feasibility. Two-worker RF is a candidate resource setting, not a production
approval. Next review should address the remaining Stage-1 RF and XGBoost
readiness and the full execution envelope rather than automatically extending
Stage-2 concurrency tests. Any further real-data probe requires a concrete
bounded scope and user approval. No new probe is authorized by this result.

## Remaining-model readiness review — 2026-11-08

Read-only constructor/source inspection confirms two unmeasured Stage-1 RF
profiles and two XGBoost components. No benchmark arrays were opened and no
estimator was fitted during this review. Current plan hash remains
`a631e5078246c22e39652d266c3ec9e488b730b2e9460e999facc908d769a019`.
Local evidence: `.release-audit/c03/remaining-readiness-review-20261108.json`.

### Approved next batch: both Stage-1 RF profiles

The user approved implementation and one attempt on 2026-11-08 with the settings
and boundaries below. The command is
`python -B -u -m experiments.c03.engineering.rf_stage1_sizing_probe --probe`.

Minimum useful result: measure load/fit cost and tree storage for each of the
two distinct historical Stage-1 RF weighting modes on the full prepared TRAIN
tensor. This closes the missing small-forest Stage-1 measurements; it does not
qualify the complete 300-tree models, checkpoint serialization or scoring.

- Import only the original combined-success record's `stage1.npz`: 1,270,025
  rows, 46 float64 features. Pin the original success/receipt/file hashes and
  verify tensor, binary labels, row IDs and each profile's weight identity.
  This is an explicit engineering import across revisions, not scientific
  checkpoint promotion. No raw TRAIN resampling or validation/test array reads.
- Run `stage1_rf_class_weight_balanced` followed by
  `stage1_rf_inv_family_clipped`, sequentially in fresh supervised processes.
  Both use seed 123, eight trees and two RF workers; all remaining effective
  constructor settings must equal their historical definitions.
- Preserve the first profile's native `class_weight=balanced_subsample` and
  absent sample weights. Preserve the second profile's `class_weight=None`
  and saved inverse-family-clipped sample weights. Do not substitute binary
  class balancing for family weights. Family labels are not in the saved tensor,
  so verify the saved weight bytes against original preparation identities;
  do not claim a fresh family-weight recomputation from those arrays.
- One attempt, **900 seconds total across both profiles**, including input
  verification/loading and result verification. Start each worker only with at
  least 6 GiB available RAM and 20 GiB free disk. Retain the 4 GiB sampled worker
  RSS ceiling, 2 GiB available-RAM floor, 10 GiB free-disk floor and 5 GiB total
  new-artifact cap. Stop/reap on violation or failure; do not automatically
  launch the second profile after a failed first profile.
- Print profile/stage changes and 30-second resource heartbeats. Atomically
  record each completed profile's source/import identity, effective settings,
  timing, tree sizes and resource result. Preserve incomplete attempts. Explicit
  recovery can reuse verified completed diagnostic records; an interrupted fit
  restarts its profile only in a newly authorized attempt, without mid-fit
  continuation or automatic budget reset.
- No retained model, validation/test scoring, selection, GPU work or quality
  metric. Reassess after this single batch before any further data fit.

Acceptance before the attempt: generated-data checks for both actual weighting
paths and fits, exact input/parameter preservation, no scoring/model retention,
per-profile completed-result reuse, corruption rejection and interruption/shared
budget handling, plus the repository's required checks. Result acceptance requires
verified completion receipts and both measured profiles within the sampled limits;
a partial result remains partial and preserves its completed diagnostic evidence.

Implemented in `engineering/rf_stage1_sizing_probe.py`: immutable profile and
batch receipts, fresh sequential workers under a common deadline, per-worker
start gates, whole-batch artifact accounting, progress and process-tree cleanup.
`--resume` verifies completed results without fitting. An explicitly authorized
`--fork` copies verified completed profile diagnostics to a fresh attempt and
restarts only unfinished profiles; source/code drift, live locks and corrupt
completion records block recovery. A failed copy lacks a committed request and
cannot execute. Existing Stage-2 runtime sources and historical evidence remain
unchanged. Focused verification exposed a synthetic timing-test issue: a tiny
fit can report zero seconds on the Windows monotonic clock. The existing
two-worker test now uses a deterministic clock for its ratio assertion; probe
timers are unchanged. The prior two-worker run remains verified historical
evidence under its original test/source fingerprint, not a current-revision
resume target after this test-only change.

Verification: 67 focused tests and 274 full-suite tests passed; all nine required
repository commands passed. The single approved attempt on 2026-11-08 returned
START_REFUSED with 3.92 GiB available RAM, below the unchanged 6 GiB gate. Neither
profile started; no training array was loaded or model fitted. Evidence:
`experiments/c03/engineering/RF_STAGE1_SIZING_PROVENANCE.json`. The approved command
can be run manually when sufficient RAM is available; both Stage-1 measurements
remain outstanding.

The user's manual Stage-1 follow-up completed both profiles in 370.203 seconds
within the shared 900-second budget. Verified on 2026-11-08: source/request and
profile/batch receipts, original import bytes, tensor/label/weight identities,
effective constructor settings, tree-storage consistency and all 1,099 monitor
samples. Completed-batch resume succeeded without fitting. Evidence:
`experiments/c03/engineering/RF_STAGE1_SIZING_SUCCESS.json`.

| Stage-1 eight-tree diagnostic | Native balanced subsample | Inverse-family clipped |
| --- | ---: | ---: |
| Fit seconds | 159.375 | 186.453 |
| Profile wall seconds | 170.937 | 198.406 |
| Peak sampled worker RSS (GiB) | 1.8309 | 1.6410 |
| Minimum available RAM (GiB) | 4.4784 | 4.3121 |
| Tree node/value storage (bytes) | 26,106,880 | 28,009,120 |

Both profiles used all 1,270,025 prepared Stage-1 rows with 46 features, eight
trees, two RF workers and seed 123. Their distinct weighting paths were preserved.
All sampled limits held; no scores, models or array copies were retained. The
last progress line reports the second profile's peak; the batch peak is the
maximum of both profiles, 1.8309 GiB.

The three RF components now have full-row eight-tree measurements. Linear
300-tree fit-time projections are approximately 1.66 hours and 1.94 hours for
the Stage-1 profiles, plus 2.44 hours for shared Stage-2: about 6.04 hours of RF
fitting at two workers. These projections exclude preparation, checkpoint
save/reload, scoring and XGBoost. They do not qualify full-forest memory or
authorize production fitting. Next is bounded synthetic CUDA readiness with
explicit device/CPU-worker/VRAM decisions, before any real-data XGBoost sizing.

### XGBoost remains a separate execution decision

The current installed XGBoost 3.3.0 reports `USE_CUDA=true` with CUDA build version
12.9. A read-only GPU query reports an RTX 5060 Laptop GPU with 8,151 MiB total
VRAM and 7,668 MiB free at inspection; this is a transient observation, not a
reservation or successful CUDA fit. Effective historical models use CUDA/hist,
24 CPU workers, 800 binary rounds (seed 123) and 1,400 multiclass rounds (seed 133).
The Stage-2 objective has five represented classes.

After the RF batch, prepare a separate CUDA readiness proposal with explicit
CPU worker count, boosting-round cap, VRAM start/stop criteria, device verification
and failure behavior. A tiny generated-data CUDA check should precede any
full-row XGBoost sizing; do not silently fall back to CPU. No GPU fit, real-data
XGBoost batch or production worker/device setting is authorized by this review.

### Approved synthetic CUDA readiness batch — 2026-11-08

The user approved implementation and one attempt with the settings below.

Read-only review confirms the installed XGBoost 3.3.0 build reports CUDA 12.9
support, and the NVIDIA GPU/memory query succeeds. Local evidence:
`.release-audit/c03/cuda-readiness-review-20261108.json` and
`.release-audit/c03/cuda-readiness-gpu-20261108.txt`. Installed APIs expose build
metadata, booster configuration and training callbacks. No GPU fit or benchmark
array access occurred during this review.

Minimum useful result: prove that both historical XGBoost objective paths can
complete a tiny fit on the selected GPU under monitored limits, with verified
effective CUDA device/round count and reliable failure cleanup. This is runtime
compatibility evidence only, not real-data capacity or quality evidence.

| Setting | Proposed fixed value |
| --- | --- |
| Data | 4,096 generated rows and 46 float64 CSR features per objective; deterministic generation; no benchmark arrays |
| Components | Binary Stage-1, then five-class Stage-2, in fresh sequential processes |
| Models | Historical constructors/weighting; model seeds 123 and 133; eight boosting rounds each, two CPU workers, GPU 0 (`cuda:0`); other settings unchanged |
| Shared time budget | 300 seconds total, including generation, context initialization, both fits and verification |
| Host RAM start | At least 3 GiB available before each worker |
| Host RAM stop | Worker-tree RSS above 2 GiB or available RAM below 2 GiB |
| GPU start | At least 6 GiB free VRAM on GPU 0; successful device/memory query |
| GPU stop | Total device-used VRAM above 4 GiB, free VRAM below 2 GiB, or missing/invalid/failed monitoring |
| Disk/artifacts | Start with at least 2 GiB free disk; stop below 1 GiB free or above 64 MiB new artifacts |
| Monitoring | Host/process checks about every 0.3 seconds; GPU query about every second with a two-second query timeout |

These are the explicitly approved synthetic-smoke limits. The 3 GiB host start
gate replaces the 6 GiB full-row-probe gate only for this tiny generated-data
surface; it does not weaken any benchmark probe's limits. GPU measurements are
device-wide, including other applications, not claimed process-exclusive VRAM.
All caps are sampled stop conditions rather than hard allocation limits.

Acceptance requires both workers to finish eight boosting rounds and report a
fitted booster configured for CUDA device 0 with the expected objectives and
effective settings. Reject CPU fallback or an unverified device; no automatic
CPU retry. Do not treat a visible GPU or a requested `device` parameter alone as
proof of CUDA execution. Eight multiclass rounds are not described as eight
individual trees. Preserve warnings and device/config diagnostics for review.

Use the existing external artifact root with a distinct synthetic-CUDA scope.
Retain progress, timing, RAM/VRAM traces, source/environment hashes and completion
receipts, but no fitted models, generated arrays, predictions or quality metrics.
No validation/test reader or benchmark import is allowed. Completed component
diagnostics can be reused after verification in an explicitly authorized fresh
recovery attempt; incomplete fits restart, without mid-boosting continuation.
One attempt is approved, with no automatic retry or limit increase.

Before the attempt, verify monitoring parsers/timeouts, shared-budget and stop
handling, CPU-fallback rejection, corruption/recovery and generated-data parameter
preservation using disposable CPU/mocked-GPU checks, plus required repository
checks. After the attempt, reassess the observed result before proposing any
real-data XGBoost sizing. Approval does not extend to the 800/1,400-round models
or resolve the final scientific worker configuration.

Implementation: `engineering/cuda_readiness_probe.py`. The worker checks the
fitted booster device, objective, class/feature counts, CPU thread setting and
histogram tree method after each round and at completion. It records the eight
completed rounds, parameter/input hashes, configuration and warnings without
retaining a model or arrays. The parent monitors process-tree RSS and total GPU
memory, uses the common deadline across fresh sequential workers, and rejects
missing/failed GPU queries or device-identity changes. This approved laptop's
single-NVIDIA-GPU enumeration is required; ambiguous CUDA remapping is refused.
All limits are sampled; a bounded GPU query can delay a host sample.

Component and batch receipts are immutable. `--resume` verifies a completed
batch without GPU access/fitting. An explicitly authorized `--fork` can reuse
verified completed component diagnostics in a fresh attempt on the same GPU
identity/driver and source revision. Active locks, corrupt receipts and incomplete
copy commits block recovery. Existing benchmark probe code/limits stay unchanged.

Verification: 57 focused tests and 312 full-suite tests passed; all nine required
repository commands passed. The single guarded GPU attempt on 2026-11-08 completed
both objectives in 8.141 seconds. Current source/request/completion receipts,
regenerated synthetic input hashes, effective parameters, all eight round
callbacks per component and 22 resource samples passed review. Completed-result
resume succeeded without fitting or GPU queries. Evidence:
`experiments/c03/engineering/CUDA_READINESS_PROVENANCE.json` and
`experiments/c03/engineering/CUDA_READINESS_SUCCESS.json`.

| Generated-data CUDA diagnostic | Binary | Five-class |
| --- | ---: | ---: |
| Fit seconds | 0.3646 | 0.5139 |
| Boosting rounds | 8 | 8 |
| Individual trees | 8 | 40 |
| Peak sampled worker RSS (MiB) | 357.51 | 375.78 |
| Peak sampled whole-device VRAM used (MiB) | 114 | 624 |

Both fitted boosters report `cuda:0` and `grow_gpu_hist`, with the expected
objective, feature/class counts and CPU worker count. No warnings or CPU fallback
were reported. All sampled limits held. Short GPU peaks may be missed; these
readings do not bound full-row GPU memory. No benchmark arrays, predictions,
quality metrics or fitted model artifacts were retained.

This establishes tiny generated-data CUDA runtime compatibility on the inspected
machine. Full-row XGBoost sizing is the next separate approval checkpoint, with
explicit import, host/VRAM, round and time limits. The synthetic host-RAM gate
does not apply to benchmark work; no full-size capacity or runtime estimate is
inferred from this result, and full scientific fitting remains unapproved.

## Approved full-row XGBoost sizing — 2026-11-09

The user approved implementation and one attempt under the fixed scope below.

Read-only review reverified the successful synthetic CUDA batch against current
sources and checked the stored training identities and model definitions. Current
plan hash remains `a631e5078246c22e39652d266c3ec9e488b730b2e9460e999facc908d769a019`.
Evidence: `.release-audit/c03/xgb-full-row-review-20261109.json`. No benchmark array
was opened and no fit occurred during this review. The instantaneous RAM reading
was below the proposed full-row start gate; resources must be checked again at
execution, without relaxing the gate automatically.

Minimum useful result: establish whether full-row input loading, GPU data
construction and eight rounds of each XGBoost objective fit the bounded laptop
envelope. This measures short full-row fits, not the full 800/1,400-round models,
scientific performance, checkpoint serialization or validation/test scoring.

| Setting | Proposed fixed value |
| --- | --- |
| Stage-1 | `stage1_xgb_inv_family_clipped`, 1,270,025 TRAIN rows, seed 123 |
| Stage-2 | `stage2_xgb_shared`, 1,201,422 TRAIN rows, seed 133, five represented classes |
| Inputs | Original combined-preparation diagnostic arrays; 46 unchanged float64 CSR features, labels, row IDs and component-specific weights |
| Fit overrides | Eight boosting rounds each; two CPU workers; explicit CUDA device 0; all other historical constructor settings/weighting unchanged |
| Execution | Stage-1 then Stage-2 in fresh sequential processes; one attempt, 900 seconds total including loading, verification and both fits |
| Host RAM | At least 6 GiB available before each worker; stop above 4 GiB worker-tree RSS or below 2 GiB available |
| GPU VRAM | At least 6 GiB free before each worker; stop above 4 GiB whole-device used or below 2 GiB free |
| Disk/artifacts | At least 20 GiB free to start each worker; stop below 10 GiB free or above 5 GiB total new artifacts |
| Monitoring | Host/process checks about every 0.3 seconds; GPU query about every second with a two-second timeout; stop on unavailable telemetry/device changes |

Pin `COMBINED_PREPARATION_SUCCESS.json` and `CUDA_READINESS_SUCCESS.json` by hash.
For each component, verify original source/request/preparation receipts and the
appropriate TRAIN array bytes before decoding, then verify tensor, labels, row
IDs and the named XGBoost weight identity. Load only that component's training
stage. Preserve the original Stage-1 family-weight array; family labels are not
available in the saved tensor for recomputation. Stage-2 balanced weights can
also be checked against the unchanged historical helper. No raw resampling,
preprocessor/model deserialization or validation/test array access is proposed.
The import remains engineering-only across recorded source revisions.

Require `cuda:0` and the expected GPU histogram updater in fitted configuration,
the correct objective/class/feature/thread settings and eight completed boosting
rounds. Reject CPU fallback, wrong-device execution or incomplete rounds. Retain
per-round/stage progress, fit timing, configuration diagnostics, warnings and
sampled resource traces. Eight five-class rounds produce 40 individual trees;
do not describe them as eight trees. No predictions, quality metrics, retained
model or new prepared-array copies are allowed.

Commit each component's diagnostic completion receipt atomically. Stop after any
failed component rather than automatically starting the next. Explicit recovery
may reuse verified completed diagnostics on the same source/GPU identity and
restart only unfinished work in a newly authorized attempt; no automatic retry,
budget reset or mid-boosting continuation. Preserve all failed attempts. All
limits remain sampled stop conditions, not hard allocation guarantees, and GPU
memory readings cover other applications too.

Before the single attempt, verify the new import and both weighting paths on
generated arrays, GPU-fallback and telemetry rejection with CPU/mocked-GPU tests,
the shared deadline, source/corruption/active-lock rejection, partial recovery,
and no scoring/model/validation/test access; run the required repository checks.
Acceptance requires verified completion of both short full-row fits within the
limits. Partial completion remains partial. Stop and reassess the evidence before
proposing any larger or full scientific run; the synthetic timing is not a
full-row runtime estimate and this batch does not approve production resources.

Implementation: `engineering/xgb_full_row_sizing_probe.py`, a separate engineering
command that leaves the verified synthetic CUDA and RF runtime sources unchanged.
It reuses the pinned CUDA telemetry/configuration checks, uses the full-row
resource limits, and binds each fitted result's input hashes to its verified
component-specific import receipt. GPU histogram execution and eight/40 individual
tree counts are verified alongside eight completed rounds. Existing component
receipt, lock, explicit recovery and fresh-process behavior is preserved.

Verification: 62 focused tests and 336 full-suite tests passed; all nine required
repository commands passed. The single attempt on 2026-11-09 returned START_REFUSED
because the GPU memory query exceeded the approved two-second timeout. The
initial RAM snapshot was also below its gate: 4.54 GiB available versus 6 GiB
required. No worker, training-array load, fit or external run was created. A
subsequent read-only `--inspect` query also timed out; no batch retry, fallback or
limit change occurred. Evidence:
`experiments/c03/engineering/XGB_FULL_ROW_SIZING_PROVENANCE.json`.

The refusal left the full-row measurement outstanding; it did not authorize
disabling monitoring or increasing the query timeout. The user's subsequent
manual run completed both components under the original limits in 27.218 seconds.
Verified on 2026-11-09: current source/request and profile/batch receipts,
original TRAIN array bytes, fitted tensor/label/weight identities, effective
parameters, CUDA device/GPU histogram updater, eight round callbacks per component
and all 74 resource samples. Completed-result resume passed without fitting or
GPU queries. Evidence:
`experiments/c03/engineering/XGB_FULL_ROW_SIZING_SUCCESS.json`.

| Full-row eight-round diagnostic | Stage-1 binary | Stage-2 five-class |
| --- | ---: | ---: |
| TRAIN rows | 1,270,025 | 1,201,422 |
| Fit seconds, including fit-time data construction | 2.6033 | 2.9050 |
| Component wall seconds | 12.578 | 13.313 |
| Individual trees | 8 | 40 |
| Peak sampled worker RSS (GiB) | 1.8885 | 1.8472 |
| Peak sampled whole-device VRAM used (MiB) | 114 | 664 |

Both used 46 unchanged float64 features, original component-specific weights,
seeds 123/133, two CPU workers and `cuda:0` with `grow_gpu_hist`. No warnings or
CPU fallback were reported. Minimum available host RAM across the batch was
4.6261 GiB. All sampled limits held. No validation/test scoring, retained models,
predictions or new array copies were produced. GPU readings can miss short
peaks and are device-wide; they do not bound full-round model memory.

All five model components now have short full-row fit measurements. Full-fit
memory, model save/reload, validation scoring and cumulative storage/runtime
remain separate qualifications. The eight-round XGBoost timings include data
construction and initialization, so no linear full-round forecast is claimed.
Next is an end-to-end producer lifecycle review and a concrete execution or
rehearsal proposal with fit/scoring checkpoints and total resource budgets.
This result does not authorize longer fits, model retention, scoring, selection
or scientific execution; new consequential choices require user approval.

## Lifecycle execution review — 2026-11-09 (bounded rehearsal approved)

The user requested this review after the successful full-row XGBoost batch.
This section proposes the next bounded batch and outlines the later full pilot;
The user subsequently approved implementation, synthetic verification and one
bounded rehearsal as specified below. The later full-pilot envelope remains
unapproved. This does not change the scientific policy.

### What the current implementation still needs

`producer/validation.py::load_component_prepared` loads a training stage and
validation together. `_fit_component` fits, predicts validation, saves the model,
then commits a single model-and-score receipt. Therefore a scoring failure can
lose a successful fit, and validation increases fit-time memory. The sizing
workers did not exercise this path. `producer/recovery.py::fork_verified` reuses
only fully completed components; it cannot recover a saved fit independently.
`producer/isolated.py` remains explicitly synthetic-only and has a shared
15-minute budget. It is not a full scientific execution command.

Proposed successor lifecycle, with new versioned receipts and preserved old runs:

1. Verify the request, source/environment/policy and original preparation
   lineage. Import only explicitly authorized arrays by hash-verified reference.
   Engineering evidence is never relabeled as a scientific checkpoint.
2. Start a fresh TRAIN-only worker for one component. Fit, save to an atomic
   local artifact, then publish a fit receipt binding model bytes, input hashes,
   effective parameters, classes, source/environment and scope. Exit the worker.
3. Start a separate scoring worker. Verify that receipt before deserializing
   the locally generated model; load validation without TRAIN. Score in ordered
   blocks of 8,192 rows using unchanged class alignment and probability
   transforms. Publish a separate score receipt binding the fit receipt and
   validation tensor/truth/row identities. Release the worker before continuing.
4. Reuse a verified fit if scoring fails; reuse completed score receipts too.
   Preserve interrupted attempts and require explicit recovery. Recheck source,
   environment, inputs and all reused artifacts. A corrupt completed receipt
   stops recovery; it is not silently classified as unfinished work.
5. Only a later scientifically authorized run may pass all five completed
   components to selection and sealing. The shared RF Stage-2 remains one fit
   and one score artifact referenced by both RF candidates.

The fit receipt commits saved bytes; a separate fresh-process reload and scoring
receipt proves the later lifecycle stage. Hashes alone do not make arbitrary
joblib files safe: only this producer's locally generated, bound artifacts may
be loaded. Keep legacy receipt readers and evidence intact; no in-place migration.
Use additive modules where possible because historical probe records pin source
bytes. Any necessary change to pinned helpers needs a successor provenance record,
never an edit to historical evidence.

Recovery is at fit/score boundaries, not inside a running estimator or a partial
score array. An interrupted fit restarts that component; interrupted scoring
restarts scoring from its verified model. No RF warm-start, changed seed, partial
ensemble or XGBoost continuation is proposed. A recovery command does not create
extra time authorization: record consumed wall time and enforce the remaining
approved lineage budget, including recovery copying and verification.

### Approved next batch: implement and rehearse this lifecycle

First implement and verify the successor on generated data. Then, if all checks
pass, attempt **one engineering-only lifecycle rehearsal** on this laptop:

- All five components, sequentially, eight RF trees / eight XGBoost rounds,
  two CPU workers each, original seeds 123/133 and weights, GPU 0 for XGBoost.
  Keep the existing sorted component order. No simultaneous model workers.
- Explicitly verified imports of the original Stage-1, Stage-2 and ordered
  700,000-row validation arrays. No new preparation fit or resampling. Retain
  the five diagnostic models and validation probability arrays outside the
  repository solely to verify reload and recovery. No selector, quality metrics,
  selection seal, test observations or scientific checkpoint promotion.
- One shared **15-minute** real-data deadline covering verification, fits,
  save/reload, scoring and final receipts. No automatic retry or budget reset.
  The three earlier RF/XGB probe batches totaled 642.718 seconds of wall time;
  the remaining 257.282 seconds is unmeasured headroom, not a completion promise.
- Host RAM: at least **6 GiB available** before starting a worker; stop above
  **4 GiB worker-tree RSS** or below **2 GiB available**. Same limits for reload
  and scoring, not just fitting.
- GPU: at least **6 GiB free** before XGBoost work; stop above **4 GiB device-wide
  used** or below **2 GiB free**. Preserve 0.3-second host / 1-second GPU polling
  and the 2-second query timeout. Verify training and inference execution paths;
  no unapproved CPU fallback or silent parameter/device substitution.
- Disk: at least **20 GiB free** to start, **10 GiB free** throughout; at most
  **5 GiB newly created artifacts across this attempt and any recovery lineage**,
  including pending files and copied recovery artifacts. Existing referenced
  preparation files are not new bytes; their presence affects free disk.
- Display component/phase, elapsed time, RAM/VRAM, completed component count,
  scoring rows and checkpoint reuse. Continue 30-second heartbeats. Report exact
  boosting-round progress where available; do not invent an RF completion percent.

Synthetic acceptance must cover: separate TRAIN/validation loaders; serialization
and reload score equivalence; ordered block versus whole-array score equivalence
and unchanged selection on fixtures; killed workers after fit commit and during
scoring; recovery without refitting; partial publication; corrupt models/receipts;
source/environment drift; active locks; shared deadlines and artifact accounting;
GPU telemetry/fallback rejection; and engineering scope rejection by selectors
and evaluators. Numerical comparisons must state their tolerances; do not loosen
them or change score transforms to hide a discrepancy. Reconcile a material
numerical difference before the real-data attempt. Run the repository's required
checks after implementation. No real-data fault injection is needed.

Rehearsal acceptance: all five fit and score receipts verify, both workers exit
cleanly per component, all 700,000 ordered score rows have the expected shape,
class order and valid probabilities, and all sampled resource limits hold.
Completion-only resume must verify without fitting or scoring again. A stopped
batch remains partial; preserve evidence and review rather than expand limits.
This rehearsal cannot establish full-model memory or runtime feasibility.

### Later full pilot: resource envelope for review after the rehearsal

Historical proposal: the November 17 execution-authority update at the top of
this document supersedes the approval status and available-RAM gates in this
subsection. Its full-ensemble settings and storage estimates are preserved below.

Scientific settings remain three 300-tree RF fits, 800-round Stage-1 XGBoost and
1,400-round Stage-2 XGBoost. Two CPU workers per estimator would be an explicit
execution-setting proposal for the full run, not an already approved replacement
for archived worker settings. Preserve all other parameters and original data
semantics. New scientific preparation must bind the approved producer revision;
do not promote the diagnostic preparation through a scope rename.

| RF component | Estimated fit hours at two workers | Estimated tree-buffer GiB |
| --- | ---: | ---: |
| Stage-1 native class weighting | 1.66 | 0.912 |
| Stage-1 inverse-family clipping | 1.94 | 0.978 |
| Shared Stage-2 | 2.44 | 1.775 |
| Total | 6.04 | 3.665 |

These are eight-tree measurements multiplied by 300/8, not measured full-fit
bounds. Tree buffers exclude estimator overhead, training arrays, serialization
scratch and scoring memory; summing them describes retained storage scale, not
simultaneous worker RSS. XGBoost full-round time and model size remain unknown;
initialization dominates the short measured fits, so no linear forecast is used.

The existing preparation occupies about 2.579 GiB. Five float64 probability
arrays require 72,800,000 data bytes (three binary attack-score columns and two
five-class matrices). Adding preparation, estimated RF tree buffers and those
scores gives about 6.312 GiB before XGBoost models, metadata, temporary files or
recovery copies. Consequently the probe's 5 GiB artifact cap is unsuitable for
a full prepared-and-retained-model run under this estimate.

A **provisional, not approved** full-producer envelope is 10 hours total,
15 GiB new artifacts across its recovery lineage, 30 GiB disk free at start /
10 GiB throughout, and the same 6/4/2 GiB host and 6/4/2 GiB GPU gates above.
This is a proposed stopping budget, not a prediction that the full run will fit.
Reassess it using measured reload/scoring/storage costs before seeking execution
approval; never raise the memory ceiling automatically. Resume must account for
time already consumed. A power loss may still cost the current multi-hour fit.

The eventual end-to-end sequence is: freeze the approved plan and execution
settings; independently bind scientific preparation; fit/save/reload/score each
component; verify all validation dependencies; select both arms and write the
immutable seal. Separate post-seal test evaluation has its own execution budget
and approval, followed by the paired analysis and article work in the master
plan. Neither the rehearsal nor the provisional producer envelope authorizes
that evaluator, new holdouts/seeds or a campaign expansion.

User decision: approved the checkpoint/lifecycle implementation, synthetic
verification and one bounded real-data rehearsal above. Implementation resumed
after interruption on 2026-11-14. The later full pilot remains unapproved.

The additive implementation is `engineering/lifecycle.py` plus
`engineering/lifecycle_probe.py`, with two matching test files. Previous pinned
producer/probe code and historical evidence are unchanged. Independent phase
receipts live in immutable attempt folders within one run. Explicit `--recover`
creates new folders only for unfinished phases; verified models are referenced
in place, not copied. The original 900-second wall-clock deadline includes
downtime, and every pending/failed attempt counts against the same artifact cap.
After that deadline, preserved checkpoints are evidence; more execution needs a
new decision. Completed `--resume` verifies without fitting, scoring or a GPU query.

XGBoost reload explicitly restores two CPU threads, CUDA 0 and histogram mode;
native model files do not preserve these runtime settings. Scoring explicitly
uses the native DMatrix prediction path, retaining the historical class-column
expansion and probability transforms. Generated-data checks compare it against
the existing estimator prediction path (exact XGB equality; RF absolute tolerance
1e-12, zero relative tolerance). No tolerance is used to change stored scores or
selection thresholds. The generated eight-tree selection fixture can produce
undefined threshold-search metadata under ties; its before/after decisions are
compared exactly with matching NaNs treated as matching undefined values. The
existing seal still rejects NaNs; this rehearsal does not qualify scientific
selection or resolve that historical fallback case.

Verification and attempt on 2026-11-14: 52 new lifecycle cases passed within
the 388-test full suite; all nine required repository commands returned exit 0. Actual
generated-data worker termination verified recovery after fit commit and during
scoring. The single real-data attempt was refused before worker creation:
2.2660 GiB available host RAM versus the unchanged 6 GiB start gate. GPU monitoring
worked, with 7.4111 GiB free. No benchmark array load, fit, score or external run
was created. Evidence: `experiments/c03/engineering/LIFECYCLE_PROVENANCE.json`.
There is no real-data lifecycle success claim and no automatic retry.
The repository scan still reports findings outside this batch; none of the seven
affected files was flagged and the counts did not change in the closure scan.
An exit-zero scan is not a statement that the whole workspace is publication-clean.

From the repository root, inspect before the manual follow-up:

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.lifecycle_probe --inspect
```

When the approved start gates pass, the bounded command is:

```powershell
.\.venv\Scripts\python.exe -B -u -m experiments.c03.engineering.lifecycle_probe --probe
```

For a completed external run, `--resume RUN_DIRECTORY` verifies its receipts
without new computation. For an interrupted run with time remaining,
`--recover RUN_DIRECTORY` explicitly reuses verified fit/score checkpoints and
starts only unfinished phases in new folders. It never restarts the 15-minute
clock. No automatic transition to a full scientific run follows completion.
