# Published-method implementation preflight

Updated 2026-11-20. The native EFC fixed-case pilot is complete after source,
synthetic-equivalence and full-row preparation checks. DOC's paper-rule tabular
implementation passed synthetic and full-row readiness checks; its scientific
pilot was interrupted and requires the approved successor recovery below.
A policy-matched comparison and the full
campaign are not frozen here.

## EFC: native fidelity pilot complete

The relevant multiclass publication is Souza et al., *A novel open set
Energy-based Flow Classifier for Network Intrusion Detection*, Computers &
Security, 104569 (2025), DOI 10.1016/j.cose.2025.104569. The available
[author preprint v3](https://arxiv.org/html/2109.11224v3), sections 3.3–3.4,
describes one energy model per training class, minimum-energy class selection,
and a class-specific training-energy 95th-percentile cutoff. This differs from
selecting a rejector under a shared validation rejection budget.

The [author-maintained package](https://github.com/EnergyBasedFlowClassifier/EFC-package)
was inspected at commit `2b935be347abf7daf4420989ef391436db418eac`
(2026-04-25), declared package version 0.1.0. `setup.py` declares “new BSD”;
the checked LICENSE contains three BSD-style conditions. The original notice
is retained in the ignored source checkout. No upstream source is vendored into
the public package. [Static audit receipt](../experiments/literature/EFC_SOURCE_AUDIT.json)
pins individual source, license and downloaded paper hashes.

Direct inspection of `_energyclassifier.py`, `_base.py` and `_base_fast.pyx`
establishes the following implementation details:

- Numerical features pass through MaxAbsScaler and 30 quantile bins. Encoded
  categorical columns can bypass that transformation. The preprocessor fits all
  supplied training rows before per-class model fitting; the caller controls
  whether any held-out family is present in those rows.
- Default pseudocount weight is 0.5. The cutoff is the sorted own-class training
  energy at index `floor(n * 0.95)`. Multiclass prediction accepts equality at
  the cutoff and resolves energy ties by the first sorted class label.
- Prediction returns a label and optionally its minimum energy, not a normalized
  probability vector. VERA's probability-based rejectors cannot be applied
  directly without defining an additional method. Numeric labels provide the
  explicit unknown sentinel -1.
- The package leaves discretizer defaults to scikit-learn. Locally, version
  1.9.0 defaults to a 200,000-row subsample, no random seed and the
  `averaged_inverted_cdf` quantile method. A reproducible preprocessing contract
  must resolve this before fitting; silently inheriting defaults is insufficient.
- The compiled pair-frequency table occupies `8 * (p*q)^2` bytes per class;
  the coupling matrix occupies `8 * (p*(q-1))^2` bytes. Retained training arrays,
  inversion temporaries and parallel class fits add memory. These formulas are
  source-derived estimates, not a measured full-row runtime or memory result.
- Cython build compatibility, numerical finiteness, save/reload equivalence and
  batchwise scoring still require a bounded synthetic check. The upstream
  tests include an implementation comparison but have not been run here.

The user adopted native multiclass EFC first as a fidelity baseline on
2026-11-18, with any common-policy adaptation labeled and specified separately.
The source-audit receipt predates that decision and remains unchanged. Model placement,
input representation, discretization/version contract and any validation-tuned
cutoffs must be explicit before a benchmark pilot. The earlier RF/XGB fits and
results are not evidence of EFC readiness.

Primary API reference:
[EFC documentation](https://efc-package.readthedocs.io/en/latest/generated/efc.EnergyBasedFlowClassifier.html).
The source commit, rather than mutable online documentation, identifies the
implementation inspected here.

### Synthetic compatibility result

The unmodified package compiled in a separate local environment and completed a
600-row, eight-feature, three-class generated-data fit (seed 123, one worker,
native defaults). All model parameters and energies were finite; whole-batch and
13-row-batch predictions/energies matched exactly, as did a locally saved and
reloaded model. The fit and these checks took 0.188 seconds; environment setup,
dependencies, compilation and checks took 36.259 seconds total. Peak process-tree
RSS was 324,780,032 bytes. Available RAM remained telemetry only; the 15-minute,
4 GiB RSS and disk/artifact guards held.

[Compact result](../experiments/literature/EFC_SYNTHETIC_SUCCESS.json) records
the exact versions, source/worker/model hashes and scope. No benchmark results
or runtime extrapolation follow from this small sample. In particular, it is
below the discretizer's 200,000-row sampling boundary and does not settle the
full-row preprocessing contract. The preceding source-audit receipt remains
historical and correctly records that no code had been executed at audit time.

The user explicitly adopted all supplied TRAIN rows with linear quantiles and
no subsampling. This matches
[scikit-learn 1.1's implementation](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.1.3/sklearn/preprocessing/_discretization.py),
but the authors' exact original environment remains unverified. The explicit
constructor passed a 200,017-row generated-data check against NumPy linear
percentiles, including RNG independence, batch equivalence and fixed bin edges
on query extremes. This constructor is not yet integrated into an EFC benchmark fit.

An initial attempt stopped before environment creation because Windows newline
translation changed the generated worker hash. Writing exact worker bytes fixed
the issue; both attempt records are retained. Completed install/build phases
have receipts, and the successful run's source, worker, model and phase hashes
were checked after completion. This local diagnostic is not yet a packaged
benchmark runner or evidence of interrupted-fit recovery.

A second synthetic check at 84 features also passed finite-energy and exact
batch/reload checks: 600 generated rows, three classes, 14.234 seconds for the
fit/checks, 30.022 seconds including environment/build, peak RSS 580,767,744 bytes.
[Width-check receipt](../experiments/literature/EFC_WIDTH84_SUCCESS.json).
The feature width matches CICIDS metadata; this is not a full-row runtime estimate.

### Adopted first pilot and TRAIN-input readiness

The user selected CICIoT2023, Botnet held out, seed 123, with the same underlying
numeric predictors and EFC's own TRAIN-fitted preprocessing. This compares
complete methods, including preprocessing. It does not isolate the classifier
or reproduce VERA's validation-budget policy.

The [bounded input extractor](../experiments/literature/efc_input_probe.py)
recovered the exact 1,270,025 known/benign C03 TRAIN rows from 239 verified source
parts, in the original sampled order. All 46 numeric columns were finite. No
imputation, classifier fit, validation or test observation reads occurred.
The class support is 68,603 benign; 5,275 BruteForce; 666,284 DDoS; 180,290 DoS;
204,628 Other; 144,945 Scan/Recon. Botnet is excluded before EFC preprocessing.

The audit completed in 32.031 seconds, peak process-tree RSS 374,505,472 bytes,
and 479,301,104 bytes of new artifacts. Its original 15-minute deadline and
4 GiB RSS/disk/artifact guards held; available RAM was telemetry. Per-part
checkpoints preserve raw values, labels and their positions. Incomplete parts
restart; completed parts verify and reuse. Completion was independently checked.
[Compact receipt](../experiments/literature/EFC_TRAIN_INPUT_SUCCESS.json) binds
source, code, test, identity and artifact hashes.

### Native integration and full-row preprocessing

The [adapter](../experiments/literature/efc_adapter.py) makes a hash-verified
source-copy change limited to three explicit discretizer keywords; an AST check
verifies that no other algorithm statement changes. The upstream license is
preserved. [Derived-source record](../experiments/literature/EFC_ADAPTER_SOURCE.json).

On 600 generated rows with 46 predictors and three classes, separate
preprocessing/per-class fits reproduced the patched native implementation
exactly: discretized rows, class parameters, cutoffs, predictions, energies,
batching and reloads. This includes constant and repeated-value predictors.
The calculation took 12.985 seconds, the complete isolated check 38.275 seconds,
peak RSS 432,205,824 bytes.
[Equivalence receipt](../experiments/literature/EFC_NATIVE_EQUIVALENCE_SUCCESS.json).

The [TRAIN runner](../experiments/literature/efc_train_probe.py) then fitted
native preprocessing on every selected known/benign row. It saved the raw and
discretized arrays, labels and preprocessor, and checked the reloaded transform
against every training row. This completed in 18.818 seconds, peak RSS
2,050,854,912 bytes, with 936,034,080 bytes of artifacts. The observed maximum
alphabet size is 30; constant/narrow bins follow native scikit-learn handling.
[Preprocessing receipt](../experiments/literature/EFC_NATIVE_PREPARATION_SUCCESS.json).

All six class fits completed and passed exact model-reload checks. Each had its
own immutable 15-minute envelope and checkpoint. The largest, DDoS (666,284
rows), took 313.650 seconds and peaked at 1,196,462,080 bytes RSS. The
[aggregate fit receipt](../experiments/literature/EFC_NATIVE_FIT_SUCCESS.json)
binds all six model/completion hashes, cutoffs and common preprocessing.

### Native fixed-case result — 2026-11-19

The [sealed evaluator](../experiments/literature/efc_fidelity.py) completed on
all 1,190,899 fixed test rows. Ordered population and truth hashes exactly match
the C03 pilot. Models, preprocessing and native prediction policy were sealed
before test reads; no validation tuning or refitting occurred. The native rule
uses minimum class energy and rejects only when that winner's energy exceeds
its own TRAIN-derived cutoff. Equality is accepted; ties use sorted class order.

| Native operating-point metric | Value | Numerator / denominator where applicable |
| --- | ---: | --- |
| Seven-label macro-F1 | 0.6610152264 | All six known classes plus Unknown |
| Accuracy | 0.7851261946 | All test rows |
| Unknown detection rate | 61.2676% | 108,144 / 176,511 |
| False rejection, all known traffic | 2.6126% | 26,502 / 1,014,388 |
| False rejection, known attacks | 2.7203% | 25,993 / 955,530 |
| Benign assigned a known attack class | 34.2095% | 20,135 / 58,858 |
| Benign rejected as Unknown | 0.8648% | 509 / 58,858 |
| Overall rejection | 11.3062% | 134,646 / 1,190,899 |

This is a **standalone native operating point**, not a policy-matched ranking
against VERA. In particular, it does not enforce VERA's validation benign-FPR
or rejection-budget constraints. Its preprocessing also excludes Botnet before
fitting, whereas the historical C03 preparation has the documented unsupervised
exposure caveat. No Stage-1 probability/AUROC is fabricated. No uncertainty or
cross-seed generalization is inferred from this one case.

Evaluation took 76.181 seconds, peak RSS 885,710,848 bytes, new artifacts
12,003,493 bytes. All 65 part checkpoints, native model/source hashes and final
metrics verified. Independent scikit-learn confusion, macro-F1 and accuracy
calculations matched the saved arrays. The
[compact result receipt](../experiments/literature/EFC_NATIVE_FIDELITY_SUCCESS.json)
has SHA-256 `ea6b8a0f1a26ac7c7d4b9046780deacfb4b9aeec10c64c9c8fd7cb03d5f1d283`.
Raw data, models and row-level scores remain external local artifacts.

Before execution, all 501 tests and all nine repository-required commands passed.
The release scanner still reports existing content findings; its zero exit code
does not mean the workspace is ready to publish. The native EFC feasibility
pilot is complete; common-policy adaptation and the full campaign remain gated.

## OpenMax: second-method candidate, not selected

Bendale and Boult's [CVPR 2016 paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)
introduces OpenMax for neural open-set recognition. The
[authors' reference repository](https://github.com/abhijitbendale/OSDN) was
inspected at commit `c13b7697c52f6748dc29f4387deced4daa6d6239`.
[Static source receipt](../experiments/literature/OPENMAX_SOURCE_AUDIT.json)
records hashes; retrieved files were checked against Git blob identities.

The inspected code assumes Python 2, Caffe/AlexNet activations, 1,000 classes and
ten image crops. It fits upper-tail Weibull models through libMR, defaults to
tail size 20, alpha rank 10 and the combined `eucos` distance, and produces an
additional Unknown output. Applying it to IDS requires an explicitly specified
neural representation and training protocol. Feeding it RF probabilities would
not reproduce the reference method.

The repository's root copyright notice and bundled libMR notice differ. The
latter is a historical restricted-use source agreement; current libMR provenance,
license and numerical equivalence were not checked in that initial receipt.
Follow-up found the [Vastlab libMR README](https://github.com/Vastlab/libMR/blob/master/README.rst)
states that BSD-3 supersedes the old license, while its
[LICENSE.TXT](https://github.com/Vastlab/libMR/blob/master/LICENSE.TXT)
still contains both the revised notice and older restricted-use paragraphs.
This source inconsistency remains unresolved; it is not an automatic rejection
of OpenMax. Neither library was built or executed. No redistribution or
compatibility conclusion is asserted.

Before selecting this candidate, resolve the maintained dependency, tabular
network architecture, training-only activation statistics, distance scaling,
alpha rank for the actual number of known classes, and validation selection
policy. The source audit does not satisfy the mandatory second-method smoke test.

## DOC and ODIN: additional source checks — 2026-11-19

The [DOC author code](https://github.com/leishu02/EMNLP2017_DOC/tree/2b870170ab20cdc9d6f0ec85631a9ddd199a2b18)
is MIT-licensed Python 2.7/Keras 2.1.2 text-CNN code. It uses independent sigmoid
outputs, binary cross entropy and class cutoffs from mirrored positive TRAIN
scores. Its example differs materially from the
[paper](https://aclanthology.org/D17-1314.pdf): the paper uses alpha 3 and rejects
only when every class score is below its own threshold; the example uses scale 1,
checks only the argmax class and rejects equality. The example also omits the
paper's pretrained embeddings. These are unresolved fidelity choices, not
interchangeable implementations. [Source receipt](../experiments/literature/DOC_SOURCE_AUDIT.json).

The [ODIN author source](https://github.com/facebookresearch/odin/tree/64e97962ccaed1fe979f43a089c0feb4d8b002fd)
combines temperature scaling with gradient-based input perturbations. Its
implementation assumes image channels and their standard deviations, old
PyTorch APIs and a separately trained neural network. The source license is
CC BY-NC 4.0. Tabular scaling, discrete-input perturbations, backbone and
selection policy remain unresolved.
[Source receipt](../experiments/literature/ODIN_SOURCE_AUDIT.json).

For both checks, retrieved source bytes matched the pinned Git blob identities.
Neither implementation was installed or executed. No method has been selected
on the basis of benchmark outcomes. A tabular neural backbone would be an
explicit adaptation for any of these candidates, requiring a reviewed contract
before its benchmark pilot.

## Broader candidate screen after user deferred selection

On 2026-11-19 the user chose further source research before selecting method two.
The following are static findings, not measured performance or runtime results.
All retrieved text files were checked against their pinned Git blob identities;
no candidate code or serialized models were executed. Each receipt lists the
files actually inspected, so this is not a claim of whole-project verification.

| Candidate | Relevance | Main unresolved requirement | Source record |
| --- | --- | --- | --- |
| OpenNet | Published open-set representation method with an author-supplied fully connected model | Python 2 / TensorFlow 1.3 port, exact batch-normalization and two-optimizer semantics, scaling and loss variant | [Screen](../experiments/literature/OPENNET_SOURCE_SCREEN.json) |
| EVM | Non-neural, distance-based open-set classifier | Reference allocates a full square distance matrix and uses libMR; full-row execution is infeasible within current limits | [Screen](../experiments/literature/EVM_SOURCE_SCREEN.json) |
| ZDA-SHAP-UQ | Published IDS method with CICIoT2023 and RF/XGB support | Attack-only scope, pre-split feature filtering, taxonomy/protocol adaptation and absent license file | [Screen](../experiments/literature/ZDA_SHAP_UQ_SOURCE_SCREEN.json) |
| GONS | Non-neural IDS method with released tabular code | Class-incremental protocol, calibration interpretation and full explicit-map memory | [Screen](../experiments/literature/GONS_SOURCE_SCREEN.json) |
| FOSS | Published isolation-based unknown-attack discovery method | Required core module missing from pinned repository; different features and incremental discovery task | [Screen](../experiments/literature/FOSS_SOURCE_SCREEN.json) |

**OpenNet:** the [paper](https://cs.fit.edu/~pkc/papers/sdm20.pdf) evaluates both
convolutional and fully connected networks. The
[author code](https://github.com/shrtCKT/opennet/tree/3fe1f589926e5507777cdff17228f8c21032434c)
provides OpenNetFlat and an Apache-2.0 license. Its native global cutoff uses
the upper 1% of TRAIN distances. Training skips sampled minibatches missing a
class; the ii+ce variant performs two sequential optimizer updates. A port must
preserve these details and verify inference batch behavior, rather than silently
replace them with familiar framework defaults. No variant is selected.

A further static concern needs resolution before an OpenNet port: OpenNetFlat
passes `self.is_training` to dropout, but its batch-normalization calls do not
pass that flag. The
[TensorFlow 1.3 implementation](https://github.com/tensorflow/tensorflow/blob/v1.3.0/tensorflow/contrib/layers/python/layers/layers.py)
defaults batch normalization to training mode and requires explicit execution
of moving-statistic update operations. The inspected OpenNet file supplies no
corresponding update dependency. This suggests batch-dependent inference in the
standalone reference path; it is a source-based inference, not a reproduced
runtime failure. Preserving reference behavior and correcting it are different
fidelity surfaces. Its scaling helper also assumes inputs already lie in
[0,1], so passing raw traffic counters directly is not a defined reproduction.

**EVM:** the [author implementation](https://github.com/EMRResearch/ExtremeValueMachine/tree/dc282598b80beaeb1db2abd39e812cc77f0e95b1)
explicitly notes minor differences from the [published method](https://arxiv.org/abs/1506.06112v4).
For 1,270,025 rows, one float64 square matrix alone is 12,903,708,005,000 bytes
(about 11.74 TiB). This is an analytical allocation estimate, not a failed run.
Subsampling or a blockwise equivalent implementation cannot be silently adopted.

**ZDA-SHAP-UQ:** the [author publication page](https://www.husseinfawaz.com/blog/proj-zero-day)
places the method after attack detection. The
[code](https://github.com/hussein-fawaz/ZDA-SHAP-UQ/tree/88fa3c196a62f4621cb2f3081fa4fc4dc9979631)
drops benign, removes nonfinite rows/duplicates and filters features before
splitting. Its main detector fits a GMM on validation-known SHAP reconstruction
error and epistemic uncertainty, selecting by BIC and rejecting below the fifth
density percentile. These semantics differ from our complete-IDS fixed population.
No LICENSE file is present in the inspected complete tree; redistribution terms
need resolution before reuse. The known-test reports in training wrappers do not
by themselves show test-based fitting: inspected fit calls receive TRAIN only.

**GONS:** the [conference accepted-paper list and program](https://ispec2026.github.io/)
confirm the title/authors; final proceedings DOI was not verified. Its
[MIT-licensed release](https://github.com/troisang1/GONS/tree/b270c7940073d129f048a67c620e4b68f0d5cd49)
uses random Fourier features, null-space projection, prototypes and per-session
calibration. The fixed configuration has 512 map components; its inspected fit
path materializes float64 features, requiring 5,202,022,400 bytes for our training
row count before other arrays. That 4.845 GiB analytical lower bound exceeds
the worker ceiling. A fixed LOAO use and any memory-preserving adaptation need
review; published class-incremental results are not a matched comparison.

**FOSS:** the [published IDS method](https://ttfish.cc/content/Papers/TON24.pdf)
has an isolation-tree design and discovers new attack clusters. However, the
[pinned repository](https://github.com/Secbrain/FOSS/tree/77d9bcdbe784cf20778273cb1096b1d7d5339b8f)
has no module matching the entry point's `from FOSS import *`, which must supply
`FOSS` and `Testingpro`. The entry also expects external 12-feature matrices;
the extractor starts from separated PCAP flows. No LICENSE file was found.
This is a source-completeness obstacle, not evidence of poor detection quality.

### Follow-up: calibration and inference contracts

The [follow-up receipt](../experiments/literature/SECOND_METHOD_FOLLOWUP_20261119.json)
records a deeper static review of GONS's base runner, calibration and inference,
and OpenNet's scaling and batching. Candidate source files matched their pinned
Git blobs; the official TensorFlow tag source has a separately recorded SHA-256.
No candidate implementation was installed or executed.

GONS's base runner supplies only base-class TRAIN and calibration rows to its
initial fit. Under the fixed configuration, calibration first searches 64
candidate quantiles for a threshold with at most 5% impostor acceptance and at
least 90% positive coverage. The 85th positive-score percentile is the fallback
when no candidate qualifies. Impostors here are distances to other known-class
prototypes, not held-out-family examples. Buffers retain the last 512 scores per
positive class and the last 512 impostor scores overall; class balancing takes
equal-length tails of the positive buffers. Calibration row order therefore
belongs in a reproducible contract. Known-class assignment also includes the
fixed configuration's support-sensitive NCDR refinement, independently of the
novelty threshold. A plain nearest-prototype implementation would omit it.

OpenNet's `latent` and `predict_prob` split input into `batch_size` chunks. With
the batch-normalization concern above, changing chunk composition or the final
short chunk could affect outputs. A reference-preserving port and a corrected
running-statistics version would need distinct labels and evidence. The source
does not supply a TRAIN-fitted numeric scaling contract: its default helper
assumes [0,1] inputs, while the separate `unit_scale` helper recalculates extrema
on each supplied array.

OpenNet also increments its iteration counter only after a batch containing
every class is accepted. For illustration, uniform replacement draws of 128
rows with the existing six-class EFC TRAIN counts have an analytically computed
41.266% probability of containing all classes. Thus 5,000 accepted batches
would require about 12,117 draws in expectation. This is an inclusion-exclusion
calculation, not a benchmark, runtime forecast or adopted pilot configuration.
Replacing that sampler with stratified batches would change the method.

The review now supports a decision between concrete routes: a clearly labeled
DOC paper-rule adaptation to tabular inputs; an OpenNet port that explicitly
resolves its inference semantics; or a GONS base-session adaptation with a
memory plan and frozen calibration order. DOC appears the smallest next
implementation contract, based on source complexity rather than test outcomes.
That recommendation concerns engineering scope, not scientific performance.
The user subsequently selected DOC paper-rule tabular drafting and then approved
the [design](RECKONING_DOC_TABULAR_CONTRACT_DRAFT.md) for implementation and
synthetic verification on 2026-11-19. Benchmark fitting remains separately gated.
ZDA-SHAP-UQ remains an attack-stage option; EVM and
FOSS retain the resource/source obstacles above. No full campaign is frozen.

### DOC implementation and synthetic verification — 2026-11-19

Implemented the [numerical core](../experiments/literature/doc_core.py) and
[synthetic training/recovery harness](../experiments/literature/doc_training.py).
The core provides streaming TRAIN standardization, mirrored positive-score
cutoff moments and the paper's all-class rejection/global-argmax rule. The
46→250→6 network uses the accepted unweighted objective and optimizer. A separate
real-data adapter and seal/evaluation workflow remain to be implemented; the
current harness is limited to generated arrays of at most 4,096 rows.

The isolated environment uses PyTorch 2.10.0+cu128, Python 3.12.10 and
NumPy 2.4.4, with [exact environment pins](../experiments/literature/doc-requirements.txt).
The CUDA wheel is from the official PyTorch index. The main project dependency
files/environment were not extended with a neural framework. CPU and GPU are
separate synthetic checks; CPU execution is not a fallback for the CUDA check.

All **20 tests passed on CPU and on GPU 0**, including stochastic-update recovery,
orphan payload/committed checkpoint recovery, multi-block validation/cutoff
recovery, changed input/payload rejection, deadline preservation and exclusive
writer ownership. Resumed and uninterrupted states matched exactly within each
tested device/environment. Early-stopping ties retained the earliest model;
selected-model cutoffs, sigmoid output reloads and batch-size decision parity
passed. Streaming scaling is numerically tested independently; a durable
real-data preprocessing pipeline is not yet verified.

The shared bounded check completed in 41.915 seconds. CPU peak worker RSS was
630,304,768 bytes; GPU peak worker RSS was 1,302,827,008 bytes and whole-device
used VRAM peaked at 751,828,992 bytes. No benchmark observations were read.
Synthetic shortened epoch/checkpoint/block schedules are engineering fixtures,
not changes to the prospective benchmark design. The
[compact receipt](../experiments/literature/DOC_SYNTHETIC_SUCCESS.json) binds the
code/tests, environment lock, logs and resource measurements; its SHA-256 is
`7a4cdcbf4ac14f9c181334d84aefb21432be7341db69ead21516a5978a3f2dbf`.
These checks establish synthetic behavior, not full-row feasibility or DOC
scientific performance. Original source-audit receipts remain unchanged.

After implementation, all nine repository-required commands completed with
exit code zero. The main suite passed 511 tests with one expected module skip
because PyTorch is isolated; that module was exercised successfully in both
dedicated CPU and GPU checks. CLI workflows, metric recomputation, manifests,
links and smoke checks passed. Release-scan exit zero is not a clean-release
claim; its content findings remain separately recorded in local audit evidence.

### DOC readiness batch — completed with checkpoint recovery

The [preparation adapter](../experiments/literature/doc_preparation.py) imported
all 1,270,025 known TRAIN rows from the verified EFC input extraction and
recovered C03's exact validation identities. Excluding Botnet left **599,278
known-validation rows**, independently verified against source labels and hashes.
The accepted standardizer was fitted on TRAIN only; raw imports, scaling moments,
float32 arrays and row bindings are checkpointed externally.

The [fit worker](../experiments/literature/doc_readiness_fit.py) completed **one
diagnostic epoch**, 9,923 optimizer updates, on all known TRAIN rows with batch
128, seed 123, the accepted network/loss/Adam settings, GPU 0 and two CPU threads.
Known-validation BCE was 0.7350341406267903, an engineering diagnostic only.
Reloaded-model sigmoid outputs matched exactly on 8,192 known-validation rows.
No DOC cutoffs, test reads or open-set performance metrics were produced. The
diagnostic model is not a completed scientific fit and will not initialize it.

The [supervisor](../experiments/literature/doc_readiness.py) used one shared
900-second deadline for preparation/fit/validation/reload, 4 GiB
worker-tree RSS, 5 GiB new artifacts, at least 20 GiB disk at start/10 GiB
throughout, GPU free at least 6 GiB at start/2 GiB throughout and whole-device
GPU used at most 4 GiB. Host available RAM remained telemetry only. Its initial
live GPU-telemetry query timed out after two seconds during CPU preparation;
the supervisor stopped that worker. Recovery reused verified import/scaling
checkpoints with the **same source hashes and original deadline**. It completed
in 215.568 seconds from the original start. No limit or population was changed.

Successful recovery-stage peaks were 1,696,813,056 bytes worker RSS,
281,018,368 bytes sampled whole-device GPU use and 2,240,912,773 bytes of new
artifacts. The initial telemetry exception occurred before its attempt summary
was written, so these records are not a complete resource trace of that aborted
attempt. The interruption and evidence limit are explicit in the
[compact receipt](../experiments/literature/DOC_READINESS_SUCCESS.json), SHA-256
`cd5d8f10abfe456f1e51db216285abadefc06f05caa60aaa241a52a74b2315d3`.
Read-only verification rehashed preparation outputs, sources, receipts and all
committed model payloads. Synthetic preparation tests passed five cases; the
actual one-epoch adapter passed six recovery/integrity cases on CPU and GPU.

Known-validation class support in the frozen order is
31,978 / 1,976 / 338,108 / 98,942 / 75,193 / 53,081. Its selected identity digest is
`279d2ecec0a2dac62ae0c5cb7e4686dfb7e279d2de74815d7f62cc9810131a7a`.
The full scientific fit and test evaluation remain gated. All nine required
repository commands completed with exit code zero after the readiness adapter
was added: 516 tests passed, with two expected PyTorch module skips in the main
environment; both modules passed separately in the isolated environment.
Release-scan exit zero is not a clean-release claim; findings remain in local
audit evidence.

### Approved DOC scientific pilot — interrupted; successor recovery approved

On 2026-11-20 the user approved implementation, synthetic verification and one
bounded scientific pilot. The batch uses this verified preparation with a **fresh** model
initialization, the accepted 100-epoch cap and patience-5 known-validation
selection. Keep the exact architecture, unweighted objective, optimizer, batch
size, seed and class order. After natural schedule completion, reload the best
epoch, derive paper-rule cutoffs from all known TRAIN rows, seal the model,
preprocessing, cutoffs and inference settings, then evaluate once on the existing
1,190,899-row C03/EFC test population. No common-policy tuning or further
holdouts/seeds are included. Native-method comparison limitations remain.

Approved maximum: **120 minutes for scientific fitting and 15 minutes for
cutoff fitting/sealing/test evaluation**, within 135 minutes overall. Keep the
4 GiB worker/whole-device-used ceilings, GPU-free/disk floors, 5 GiB new-artifact
cap and host-RAM telemetry policy. The measured fit-worker phase, including
startup, validation, checkpointing and reload, took about 67 seconds for one
epoch. Multiplying by 100 gives roughly 112 minutes: a conservative planning
extrapolation, not a verified full-fit duration or guaranteed bound. Early
stopping may finish sooner. A budget-stopped incomplete fit is not promoted to
scientific test evaluation.

Before launch, implement and synthetically verify an efficient checkpoint
layout: store each epoch's row order once; reference immutable model/optimizer
payloads from lightweight validation-block records instead of repeating the
entire row order and model per block. Keep update-level recovery, RNG state,
exclusive ownership, source/data/environment bindings and the original
deadlines. Catch telemetry failures into an explicit stopped-attempt receipt,
while still terminating the worker and preserving the two-second live-query
limit. The current readiness checkpoint layout must not be scaled blindly to
100 epochs under the artifact cap.

The successor runner is implemented in
[doc_pilot.py](../experiments/literature/doc_pilot.py), with separate fit,
binding and post-seal evaluation modules. Generated-data checks passed eight
cases on CPU and nine on GPU, including exact agreement with the prior
synthetic training schedule, interrupted-update recovery, payload/order/input
tamper rejection, cutoff replay, and an integrated fit/seal/score/reload case.
Four main-environment tests passed for phase deadlines, telemetry timeout
cleanup and receipts, rejection of unsealed export, and exact test export and
recovery on generated Parquet data. All nine required repository commands
completed with exit code zero; 520 tests passed with three expected isolated
PyTorch module skips. Release-scan findings remain separately recorded.
The bounded scientific pilot started on 2026-11-20. When the chat resumed later,
its supervisor and worker were absent, and the original fit deadline had expired.
The verified checkpoint records 38 completed epochs and 534,528 rows of epoch 39
(381,250 optimizer updates). All source bindings, committed journal payloads and
model references passed integrity checks. No completed-fit receipt, policy seal
or test evaluation exists. The cause of process exit is not established, and no
terminal resource summary was retained for that attempt.

The user approved a separately recorded successor recovery: import this verified
checkpoint without changing its optimizer/RNG state, epoch count, selected model
or patience state; keep the original 100-epoch total cap; grant a new 120-minute
fit window and 15-minute post-fit window with unchanged resource limits. The
original request and deadline remain immutable. Recovery implementation and
synthetic verification precede launch; the next supervisor will run independently
of the chat's command session. No scientific performance result is claimed yet.
Historical completed readiness source bytes remain unchanged.
