# DOC paper-rule tabular pilot — review draft

Status: **DESIGN APPROVED FOR IMPLEMENTATION AND SYNTHETIC VERIFICATION ONLY**.
Date: 2026-11-19. Following review, the user approved the design and synthetic
verification. The tabular design choices below are adopted for that scope;
future-tense proposal wording records the reviewed specification. Benchmark
fitting remains gated by a concrete readiness plan. This document does not
freeze the full campaign or authorize inspecting new test outcomes. A subsequent
2026-11-19 decision authorizes one bounded engineering readiness batch with full
known TRAIN/validation preparation, one diagnostic epoch and reload verification;
it does not authorize the scientific early-stopped fit or test evaluation.

## Purpose and fidelity boundary

Develop a second, methodologically distinct literature pilot alongside native
EFC: a neural one-versus-rest classifier with DOC's paper-defined rejection.
Proposed label: `DOC-paper-rule-tabular-v1`. It is a tabular adaptation of DOC,
not a reproduction of the published text-CNN experiments or their reported scores.

Primary sources: [Shu, Xu and Liu, EMNLP 2017](https://aclanthology.org/D17-1314.pdf),
especially equations 2–3 and section 2.3; and
[author code at commit 2b870170ab20cdc9d6f0ec85631a9ddd199a2b18](https://github.com/leishu02/EMNLP2017_DOC/tree/2b870170ab20cdc9d6f0ec85631a9ddd199a2b18).
The [existing source audit](../experiments/literature/DOC_SOURCE_AUDIT.json)
records verified source hashes and the MIT license. Its candidate-only status is
historical; the current decision authorizes this draft, not a benchmark.

## Core rule proposed for adoption

For K known classes, train K sigmoid outputs with one-hot binary cross entropy.
For each class i, obtain its own-class scores p_ij from all its TRAIN examples
with the selected model in evaluation mode. Mirror each score to 2−p_ij.
The mirrored distribution has mean 1; using population variance gives

    sigma_i = sqrt(mean_j((1 - p_ij)^2))
    t_i = max(0.5, 1 - 3 * sigma_i)
    reject(x) iff p_i(x) < t_i for every i
    otherwise predict argmax_i p_i(x)

Equality is accepted. The argmax runs over all classes, not only those passing
their cutoff. No softmax normalization, held-out-family threshold fitting, or
validation tuning of alpha is proposed. The Gaussian construction does not
establish a distribution-free rejection-rate guarantee.

The author example instead uses alpha=1, tests only the winning class and
requires strict acceptance above its cutoff. We explicitly choose the paper
rule for this draft. The mirrored population-standard-deviation estimator is
consistent with the example's Gaussian maximum-likelihood fit.

Proposed tie extension: smallest index in the frozen class order. Empty class
support, nonfinite scores or scores outside [0,1] fail the run; no substitute
cutoffs. Threshold moments use float64 accumulation in a fixed row/block order.
The score of each true class is included even when the model misclassifies that
TRAIN row. Do not filter to correctly classified examples.
Inference sigmoids are computed in float32 and then promoted to float64 for
cutoff accumulation/comparison; the promotion does not recover lost precision.

## Proposed pilot population and preprocessing

| Item | Proposed contract |
| --- | --- |
| Dataset and holdout | Existing CICIoT2023 Botnet case, seed 123; one pilot only |
| TRAIN population | Exact 1,270,025 known/benign identities from the verified EFC numeric-input receipt |
| Class order | Benign, BruteForce, DDoS, DoS, Other, Scan/Recon; Unknown is output index 6 |
| TRAIN support | 68,603; 5,275; 666,284; 180,290; 204,628; 144,945 |
| Predictors | Same ordered 46 raw numeric columns; no EFC bins or VERA-transformed tensors |
| Scaling | Per-column mean and population variance fitted on these TRAIN rows only; divide by standard deviation, use scale 1 for zero variance; no clipping |
| Missing/nonfinite values | Fail and report; no invented imputation. Existing TRAIN audit found none; validation/test still need verification |
| Numeric representation | Fit scaling moments in float64, transform in bounded blocks, cast to float32 with finite checks |
| Validation | C03's fixed 700,000-row validation sample, filtered to the six known classes; exclude Botnet before model selection |
| Test | Exact sealed C03/EFC ordered population of 1,190,899 rows; same truth mapping; no fresh test sampling |

TRAIN identities, feature order and source hashes come from the
[verified input receipt](../experiments/literature/EFC_TRAIN_INPUT_SUCCESS.json).
Reusing these inputs is a proposal for DOC; EFC's fitted transformations/models
are not imported. The exact known-validation count, support and identity hash
must be extracted and verified before fitting; they are not assumed here.

This uses an existing validation partition rather than creating a new internal
20% split. All known TRAIN rows remain available for gradients and cutoff fitting.
Test identities are anchored by the
[EFC evaluation receipt](../experiments/literature/EFC_NATIVE_FIDELITY_SUCCESS.json).
The DOC worker receives no test arrays or labels before its model and cutoffs
are sealed. It must not use prior EFC/C03 test outcomes to tune this design.

## Proposed network and training specification

The text embedding/convolutional representation is replaced by a dense network.
The following numerical settings are explicitly tabular engineering choices;
they are not claimed as the authors' exact training environment.

| Setting | Proposal |
| --- | --- |
| Network | 46 → Dense(250) → ReLU → Dropout(0.2) → Dense(6) logits |
| Output | Six independent sigmoids; no batch normalization or output dropout |
| Initialization | Xavier-uniform weights, zero biases |
| Objective | Sum of six binary cross-entropies per row, then mean across the batch; stable logits implementation |
| Weighting/sampling | Unweighted loss; each TRAIN row once per epoch, shuffled without replacement; no oversampling |
| Optimizer | Adam, learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0; no schedule or gradient accumulation |
| Batch size | 128; final short batch included |
| Model seed | 123; distinct from the fixed partition/sample identity |
| Epoch cap | 100, with early stopping on known-validation mean BCE |
| Selection | Strictly lower validation loss improves; minimum delta 0; patience 5 completed non-improving epochs; ties retain earliest model |
| Validation cadence | Once after every complete epoch, evaluation mode, fixed ordered blocks of 8,192 rows |
| Final model | Reload best completed epoch; no refit on TRAIN+validation |
| Cutoff fit | Selected model's own-class scores on every known TRAIN row, evaluation mode, ordered blocks of 8,192 |
| Precision/runtime | Float32 neural operations, float64 aggregate statistics; AMP and TF32 off; deterministic operations requested, fail if unavailable |

Unweighted training is a deliberate first-pilot proposal despite class imbalance.
Changing class weights, batch size, architecture, epoch cap or early-stopping
policy would create a revised contract; none is to be selected after test scoring.
For validation loss, aggregate sums and row counts rather than averaging batch
means equally, which would over-weight the final short batch.

Proposed implementation runtime: a dedicated local PyTorch environment, GPU 0,
two CPU compute threads and a synchronous input reader. Exact candidate package,
CUDA and driver versions must be pinned before the synthetic readiness check
and retained with its result if it passes.
Metadata inspection on 2026-11-19 found no torch, TensorFlow or Keras installed
in the project environment. No dependency was installed during this draft.
No CPU fallback or dependency change to the main project environment is implied.

## Recovery, progress and bounded execution proposal

Use the repository's existing immutable-plan, atomic-publication and exclusive
writer-lock conventions. Every stage binds source revision, environment,
configuration, ordered row identities and predecessor artifact hashes. Verify
hashes before reuse and retain the original stage deadline on recovery.

Checkpoint preprocessing and transformed input parts independently. During fit,
save the last complete optimizer update every 250 updates and at epoch end:
model and optimizer state, epoch/order/cursor, Python/NumPy/CPU/CUDA RNG states,
early-stopping state and best-model reference. A checkpoint must reference all
its durable payloads before being considered complete. An interrupted update
is replayed from the preceding committed checkpoint. Persist the epoch's row
order; never reshuffle the remainder on resume. Checkpoint validation and cutoff
accumulators by completed block, bound to an immutable model hash, so replay
cannot double-count a block. Test scoring has separate per-part checkpoints.

Progress should report stage, completed rows/updates, epoch, validation loss,
best epoch, elapsed time and measured host/GPU memory. Show an ETA only after
enough measurements, and label it an estimate. Never report a partial fit as a
scientific completion merely because its time budget expired.

Proposed first readiness batch after design approval: synthetic semantic and
resume checks, then a separately approved bounded real-data readiness run.
Retain 15-minute stage budgets, 4 GiB worker RSS, GPU 0 with at least 6 GiB free
at start, at most 4 GiB whole-device used and at least 2 GiB free throughout;
20 GiB disk at start/10 GiB throughout and 5 GiB new artifacts. Available host
RAM remains telemetry only, as already directed by the user. These are proposed
DOC limits, not authority to launch a job from this draft. Full-fit duration and
an executable run manifest must be frozen from readiness evidence before the
scientific fit. Neither runtime nor memory sufficiency is yet measured.

## Verification and reporting requirements

- Check streaming threshold moments against explicitly mirrored arrays,
  including zero variance, the 0.5 floor and empty-class rejection.
- Distinguish the paper rule with scores (0.90, 0.70) and cutoffs (0.95, 0.60):
  accept and return class 0. Also test equality, all-fail and tie cases.
- Verify unweighted BCE reduction, TRAIN-only scaling, exclusion of Botnet from
  gradients/validation selection/cutoff fitting, and exact population bindings.
- On generated data, interrupt before/after checkpoint publication and within
  an epoch; compare resumed updates/model selection/cutoffs to uninterrupted
  execution in the same pinned environment. Reject changed inputs and sources.
- Verify fixed-block repeatability, reload parity and alternative-block numeric
  differences. Require identical decisions on the acceptance fixture; investigate
  any discrepancy before a benchmark run. Do not promise cross-device bit identity.
- Seal the selected model, cutoffs, feature order, class order and inference
  settings before test resolution; independently reconcile confusion and metrics.

Report seven-label macro-F1, accuracy, Unknown detection, known and known-attack
false rejection, benign-to-known-attack rate, benign rejection and overall
rejection, with their counts. Save per-class thresholds and TRAIN support.
No synthetic Stage-1 AUROC is assigned to this single-stage method. Native DOC
cutoffs do not enforce VERA's validation FPR/rejection budgets; a common-policy
adaptation remains a separate future contract. One case establishes feasibility,
not robustness, ranking, full P2 completion or campaign-wide conclusion stability.

## Review disposition

The user accepted the population/scaling, dense unweighted network and known-only
validation training schedule, with implementation and synthetic verification
authorized on 2026-11-19. Benchmark execution still needs its own concrete
resource/readiness manifest. The full campaign and manuscript remain unfrozen.
