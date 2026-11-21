# Reckoning handoff — 2026-11-07

## Current checkpoint — 2026-11-20

LATEST ~17:05UTC resumption: user continued after interruption; quota window
1795212263,9%used. DOC supervisor/worker absent; session50651 unknown. Last
checkpoint phasefit,epoch index38 (=39inprogress),cursor534528,updates381250;
38completedepochs,best_epoch index37 (=38),bestVALBCE.4778781863738218,bad_epochs0.
Original fitdeadline1795179460.310 expired. No fit-complete/seal/test. Exitcause
not established; no terminal attempt resource summary. All33sourcehashes,journal
chain and ALL committed modelreferences verified. Lastcommit004376 SHA
bb9391c412475755980a35c660d91fef453b03667dd2b2c36275047d9b673d4c;
requestSHA ed82ca9f91fc4b89d475f1672938b7bd6142aef5deedc4234249ddd9ae30b598.
User explicitly APPROVED successor recovery contract importing this checkpoint,
preserving100TOTALepochs/patience with NEW120minfit+15minpost limits and unchanged
RAM/GPU/disk/artifactlimits. Implement/syntheticverify beforelaunch. Launch next
long supervisor detached/hidden with logs so tool-session teardown need not kill
it. Preserve original request/deadline/source; no further DOCfit started yet.

CICIDS materialization COMPLETE31.5181788063s,peakRSS189419520B,
peakartifacts243971253B; run contiguous-seed123-20261120t170524030304z.
All20parts independentcopies;2099879rawpositions unique acrosssplits;currentraw
hashes match. train1223347,val432067,test444465; nofit/scoring. Public
MATERIALIZATION_SUCCESS.json SHA266b0c46ac5062e0c773e6176a6ad64b72ac00b0f369737c0e0dea296aa88eea;
completionSHA fcf39b5f3e55061a00430f6bb4ee94bbd85e8c236c10202c6f2a073a599b9064.
Materializer source/test nowfrozen. Required9checks session34413running, logs
.release-audit/literature/cicids-materialize-required-check-*-20261120.txt and
matching summaryJSON. No benchmarkjobsactive. These entries supersede earlier
running/pending descriptions below.

User resumed and explicitly APPROVED the bounded DOC scientific pilot: fresh
seed123 model, up to100epochs/patience5,120min fitting then15min TRAIN cutoffs,
seal and the existing1190899-row test population;135min overall. Retain4GiB
worker/GPU-used ceilings,GPU-free/disk floors,5GiB new artifacts;host available
RAM telemetry only. Implement/synthetically verify compact checkpointing first.
New successor files doc_pilot{,_io,_fit,_test}.py; tests/test_doc_pilot{,_fit}.py.
Do not modify sources bound by previous completed readiness/synthetic runs.
Implementation underway; no scientific benchmark fit launched yet. Mainenv
supervision/deadline tests3passed17.56s. First isolated synthetic attempt failed
at collection because shared artifact helper imported SciPy (absent there);
successor now uses dependency-light path checks and lazy mainenv-only imports.
Final synthetic run doc-synthetic-2ae6649844b9 COMPLETE24.647s:8CPU tests plus
1GPU-onlyskip;9GPU tests; peakworker1339289600B/GPU617611264B. RequestSHA
aa80643b3c9bcabeebb9ee501e1818a36bc9e05f86305dea6d863b64f37084cb.
Final mainenv focused4tests passed2.35s. Definition33sourcehashes verified.
Required9checks COMPLETE all exit0;520tests passed227.08s/3expected Torch skips;
isolated newtests8CPU/9GPU passed, earlier unaffected Torch coverage retained.
Logs doc-pilot-required-check-*-20261120.txt
and summary .release-audit/literature/doc-pilot-required-checks-20261120.json.
New quota window reset1795189168,32%used onlookup; preserve exhaustion/reset rule.
LAUNCHED .venv/Scripts/python.exe -B -u -m experiments.literature.doc_pilot --run;
session50651, external run doc-pilot-20261120t105740310195z. Initial fitting stage.
Next: monitor session50651/progress and resource receipts. Review any stop before
recovery; original fit/postfit deadlines cannot be extended. Full scientific
pilot source bytes now frozen for this run. User permits bounded jobs to finish
if quota ends. No other benchmark jobs active.
This paragraph supersedes the earlier awaiting-approval/stop state below.

User ALSO APPROVED one bounded CICIDS seed123 materialization after DOC finishes:
12/4/4units, train1223347/val432067/test444465, independent copies of20 verified
parts plus raw-row identities;900s,4GiBworker,5GiB artifacts,disk20GiBstart/10floor,
hostavailableRAMtelemetry,noGPU/model/scoring. Implemented successor
experiments/cicids_partition/materialize.py and tests/test_cicids_materialize.py.
Focused11tests passed1.54s (interrupted copy/publication recovery, corruption,
native raw-label offsets, support and deadline). Metadata-only definition verified
20units/14sourcebindings/exactcounts. No split files materialized yet. Launch after
DOC completion with .venv/Scripts/python.exe -B -u -m
experiments.cicids_partition.materialize --materialize; then verify/record output
and run required repository checks for this added module. Never edit DOC-bound
source while running. Latest DOC epoch32,best32;worker~1.57GiB;session50651active.

User resumed again after cutoff. New quota window reset epoch1795136096, 0% used
at this resumption; retain stop-on-exhaustion/reset and allow bounded jobs to finish.
All six native EFC classes COMPLETE and read-only verified. Aggregate receipt
experiments/literature/EFC_NATIVE_FIT_SUCCESS.json SHA256
9611b0c1b9778226dee6592cce05334837c4ff9091f02007a5f15a180b57fca8
binds common preprocessing, six model/completion hashes and native cutoffs.
Largest class DDoS666284 rows took313.650s, peakRSS1,196,462,080B.
All nine required checks completed exit0; 501tests passed232.58s. Evidence:
.release-audit/literature/fidelity-checks-20261118.json and per-command logs.
Existing release scan findings remain; exit0 is not a clean-release claim.
Native fidelity evaluation COMPLETE: efc-native-fidelity-20261119t053854440790z,
76.181s, peakRSS885,710,848B, artifacts12,003,493B. Exact C03 ordered test/truth
hashes match across1,190,899rows/65parts. Native TRAIN cutoffs retained.
MacroF1 .6610152263803241; UDR108144/176511; knownFUR26502/1014388;
benign-to-known-attack20135/58858; benignUnknown509/58858. Native policy is
NOT policy-matched to VERA; no controlled-method ranking/generalization claim.
All hashes/part arrays reconciled and independent sklearn confusion/F1/accuracy
matched. Compact EFC_NATIVE_FIDELITY_SUCCESS.json SHA256
ea6b8a0f1a26ac7c7d4b9046780deacfb4b9aeec10c64c9c8fd7cb03d5f1d283.
Console .release-audit/literature/efc-native-fidelity-20261119.txt; native evaluator
and its tests are now bound; do not change without a successor surface.
Literature report reconciled with result; check_links and git diff --check passed.
Static DOC/ODIN audits added, each source byte checked against pinned Git blob.
No upstream candidate code installed/executed. DOC paper/code differ on alpha3
versus1, all-class versus winner-only rejection and equality. Scientific choice
asked via async: DOC paper rules + draft tabular contract, OpenMax, or more audit.
User answered: continue source research before selecting method two. No DOC
implementation or tabular contract adopted. Expanded static screens saved as
OPENNET/EVM/ZDA_SHAP_UQ/GONS/FOSS_SOURCE_SCREEN.json under experiments/literature.
Each inspected text matches pinned Git blob. No candidate installed/executed.
OpenNet has native flat model but Python2/TF1.3 numerical port and inference
batch-normalization questions; no batch_norm is_training flag/update dependency
in inspected flat source, TF1.3 defaultTrue: source inference only, not runtime proof.
EVM full1270025-square float64 matrix analytically11.736TiB; GONS512component
float64 map4.845GiB before other arrays. No resource benchmark was run for either.
ZDA-SHAP-UQ is attack-only, pre-split filtering and no LICENSE in pinned tree.
FOSS lacks imported FOSS core module/Testingpro in pinned tree, expects12feature
PCAP-derived inputs; no LICENSE found. GONS accepted-paper/program confirmed
at official ISPEC2026 site; proceedings DOI not verified. No candidate selected.

CICIDS raw-label reconciliation COMPLETE, run raw-label-reconciliation-20261119,
10.527s,peakRSS144826368B. All20unit counts/row boundaries/drop totals reproduced
from5raw sources using current native mapper; before/after source hashes and
independent receipt/count reconciliation passed. No predictor values parsed or
splits written. Private checkpointed bounded runner:
.release-audit/cicids-partition/raw_support_audit_20261119.py, source hash inreceipt.
Public RAW_LABEL_RECONCILIATION_SUCCESS.json under experiments/cicids_partition.
Does not prove historical raw-byte identity or processed-predictor equivalence.
Follow-up static source review completed for GONS calibration/config/inference/
base runners and OpenNet scaling/chunking. Receipt
experiments/literature/SECOND_METHOD_FOLLOWUP_20261119.json SHA256
d60ad219e15c3516efcce5e6629c9718f2868b2e1f2cb6aae7501d7210173cfa.
Candidate files match pinned Git blobs; official TF1.3 source has SHA256 only.
GONS fixed config uses constrained grid first, 85th percentile fallback, last512
ordered calibration buffers and separate NCDR class refinement. OpenNet graph
suggests batch-stat inference; no runtime proof. Native iterations count accepted
all-class batches: analytical all-class probability .412659 at128 rows for current
known support, about12117 draws for5000 accepted batches, not runtime measurement.
User selected DOC paper-rule tabular DRAFT route on2026-11-19; not benchmark
execution. Draft docs/RECKONING_DOC_TABULAR_CONTRACT_DRAFT.md now specifies
paper alpha3/all-class strict rejection/global argmax/equality acceptance,
mirrored positive TRAIN population SD, plus proposed tabular choices: EFC raw
TRAIN identities/46columns, TRAIN standardization,46-250-6 network with hidden
dropout.2, unweighted BCE, Adam.001,batch128,seed123,100epoch cap/patience5,
existing known-only C03 validation for selection, best model then TRAIN cutoffs.
User APPROVED design and synthetic verification; benchmark fitting still gated.
doc_core.py/doc_training.py implement numerical rules and synthetic-only network/
recovery (4096row cap). Isolated .release-audit/doc-env installed torch2.10.0+cu128,
numpy2.4.4,pytest8.4.2,psutil7.2.2; exact18package pins in
experiments/literature/doc-requirements.txt. Main project dependencies unchanged.
CPU20tests PASS20.79s; CUDA20tests PASS15.33s; bounded run
doc-synthetic-98528e215fbc COMPLETE41.914934s. Peak CPU worker630304768B;
CUDA worker1302827008B,whole-deviceGPU751828992B. No benchmark reads.
Optimizer/RNG/order/cursor, validation/cutoff block recovery, orphan/commit
publication failures, input/payload drift, deadline, lock, early stopping and
reload parity tested; resumed states exactly match same-device uninterrupted.
Receipt DOC_SYNTHETIC_SUCCESS.json SHA256
7a4cdcbf4ac14f9c181334d84aefb21432be7341db69ead21516a5978a3f2dbf.
Sources/request/log/environment bindings independently verified. Real-data
preparation and sealed evaluator remain unimplemented, no full-row feasibility.
Next proposal in literature report: one900s full-row TRAIN/knownVAL preparation
plus1diagnostic epoch/validation/reload, unchanged4GiBworker/VRAM and disk caps,
availableRAMtelemetryonly. ExpectedknownVAL599278 fromC03plan, mustreverify.
No cutoffs/testscoring; scientificfit wouldstartfresh. Requires userdecision.
User APPROVED bounded real-data DOC readiness: prepare TRAIN/knownVAL plus
1epoch/validation/reload,900s shared,4GiBworker/wholeGPUused,existingGPUfree/disk
and5GiBartifact limits,hostavailableRAMtelemetryonly. No cutoffs/test reads.
Implemented doc_runtime.py/doc_preparation.py/doc_readiness_fit.py/doc_readiness.py.
Preparation5generatedtests PASS; actual adapter6recoverytests PASS onCPU andCUDA,
bounded run doc-synthetic-c525c90ef5b1 COMPLETE21.095s. Initial helper startup
GPU query timed out before workers; successor helper uses existing10sstartup
policy, retains2s live queries. Existing synthetic/core evidence unchanged.
Approved batch executed: doc-readiness-20261119t200421626706z COMPLETE215.5676s
including one telemetry-interrupted preparation attempt and same-deadline recovery.
Initial nvidia-smi livequery timedout2s; workerstopped, no source/limit changes,
raw/scale checkpoints reused. Initial attempt lacked final resource summary;
do not call recovery metrics complete trace of abortedattempt.
FullTRAIN1270025,knownVAL599278; val support31978/1976/338108/98942/75193/53081.
Oneepoch9923updates; valBCE.7350341406267903 engineeringonly;8192row reload exact.
SuccessfulrecoverypeakRSS1696813056B, sampledGPU281018368B, artifacts2240912773B.
Receipt DOC_READINESS_SUCCESS.json SHA256
cd5d8f10abfe456f1e51db216285abadefc06f05caa60aaa241a52a74b2315d3.
Source/prepared/modeljournal/receipts independentlyverified. Newreadinesssource
bytes nowboundby completedrun; use successors for changes. No cutoffs/test reads.
Required9checks COMPLETE, all exit0; session30086 finished. Main suite516passed,
2expectedPyTorchmodule skips in352.82s; isolated CPU/GPU checks cover these.
Logs
doc-readiness-required-check-{1..9}-20261119.txt, summary
.release-audit/literature/doc-readiness-required-checks-20261119.json.
Next proposalinliteraturereport: fresh scientificDOCfit100epochs/patience5,
120minfit+15mincutoffs/sealedtest, sameRAM/GPU/disk/artifactlimits. Mustfirst
implementefficient checkpoints (epochorderonce, lightweightvalidationrecords)
and explicitstopped telemetryreceipts; currentreadycheckpointduplicationwould
exceedartifactcapover100epochs. Notauthorized untiluserdecision.
Earlier candidate-only receipts remain historical evidence, not authorization.
Draft checks: inline stdlib arithmetic verified mirrored population-SD identity
on3fixtures and4 paper-rule decisions (winner-fails/nonwinner-passes, equality,
all-fail,tie). New draft/report relative links verified; scripts/check_links.py
and git diff --check passed. No product code changed or full suite rerun.
Design question answered APPROVE. Required nine-command verification COMPLETE:
.release-audit/literature/doc_required_checks_20261119.py;
logs doc-required-check-{1..9}-20261119.txt and doc-required-checks-20261119.json.
All9commands exit0. Full suite511passed/1expectedskip in272.84s; skipped DOC
PyTorch module independently passed in dedicated env on CPU and GPU (20each).
CLI workflows, ProtocolA recomputation, manifests, links, smoke and scan passed.
Latest scan counts unchanged:206 localpaths,22emails,36privatenetwork,5filenames.
CICIDS predictor reconstruction COMPLETE: predictor-reconciliation-20261119,
41.53145098686218s, peakRSS455258112B,20units/2099879rows/84predictors+3targets.
All parsed numeric/text values match historical prepared parts in order exactly,
without tolerance. Only dtype differences:two port buckets and binary target;
no reconstructed numeric nonfinite cells. Private bounded runner/checkpoints:
.release-audit/cicids-partition/predictor_reconciliation_20261119.py.
Receipt experiments/cicids_partition/PREDICTOR_RECONCILIATION_SUCCESS.json SHA256
2e8e9f22fcd65f0b477c6e1881e2841d2fdc65b0fae41cb392ea1b34a657ba7e.
Independent current raw/reference/metadata/runner hashes and all unit receipts
verified. No new splits or models. Exact values do not establish historical raw
byte identity, unique provenance of duplicate rows, or independent units.
Next proposed checkpoint:seed123 materialization from verified unit contents,
separate hash-checked copies of227120777bytes plus raw-position identities,
fixed approved assignment and original deadline/checkpoint integrity. Not run.
Common-policy adaptation and full campaign remain separate decisions.
No jobs active; latest quota100%, reset epoch1795136096. Stop rule reached;
no new work started. Next resumption: obtain decision on concrete DOC scientific
pilot proposal before implementing/launching it. No approval question sent yet.
Post-screen release scan exit0; counts unchanged:206 absolute_local_path,
22 email_address,36 private_network,5 sensitive_filename. Evidence:
.release-audit/literature/post-screen-release-scan-20261119.json.
Both updated reports' local links and candidate-screen JSON checked; tracked
check_links and git diff --check passed. Historical sections below describe
earlier checkpoints and do not override this current state.

## Active EFC continuation — 2026-11-18

User resumed after cutoff; renewed quota window allowed, 1% used, reset epoch
1795038858. Keep quota-exhaustion/reset stop rule; bounded jobs may finish.
Decisions now explicitly adopted: native EFC fidelity baseline first, adaptations
later; all supplied TRAIN rows with linear quantiles/no subsampling; first pilot
CICIoT2023 Botnet/seed123; same underlying numeric predictors with EFC's own
TRAIN-fitted preprocessing, not VERA-transformed tensors. This is a comparison
of complete methods, not classifier-only variation. No full campaign freeze.

Numerical constructor experiments/literature/efc_discretization.py and two tests
in tests/test_efc_discretization.py implemented. Tests passed2 in17.24s, including
200017 generated rows checked against full-sample NumPy linear percentiles,
independence from global RNG, exact batch transformation, no refit on query
extremes. Initial fixture generated sub-1e-8 bins (correctly removed by sklearn);
changed fixture scale, not algorithm. Logs efc-discretization-{fixed-}20261118.txt
under .release-audit/literature/. Constructor is not yet connected to EFC fit.

Additional synthetic84-feature probe COMPLETE, 600 rows/3classes/seed123, native
defaults, fit/checks14.234s, total30.022s, peakRSS580,767,744B. Exact batch/reload
and finite checks passed. Run efc-width84-20261118t064043992200z; compact receipt
experiments/literature/EFC_WIDTH84_SUCCESS.json. Source/worker/model/phase hashes
verified. No full-row runtime extrapolation; no benchmark values read by probe.

Initial action in this continuation: bounded TRAIN input extraction/validity audit
using the completed C03 stage1 sample identities and its46 numeric columns.
Read exact selected raw rows, exclude Botnet before EFC preprocessing, verify
source hashes/labels/ordered identities; count nonfinite values before deciding
any imputation. Checkpoint external part artifacts; 900s,4GiB RSS, retained disk
and5GiB artifact guards, available RAM telemetry only. No classifiers/val/test
reads in this preparation batch. Never edit C03/replay source bytes bound by
completed runs. Existing C03 Stage1 array has1,270,025 known/benign rows and46
numeric predictors, zero categorical columns. It uses scaling, not PCA.

UPDATE: input extraction is implemented and COMPLETE, run
efc-train-inputs-20261118t170122755261z. All1,270,025 selected known/benign rows,
46 numeric columns,239parts; zero nonfinite cells. Class support in0..5 order:
68603,5275,666284,180290,204628,144945. Wall32.031s, peakRSS374,505,472B,
artifacts479,301,104B. Read-only verify_completed passed. Compact receipt
experiments/literature/EFC_TRAIN_INPUT_SUCCESS.json SHA256
c4e4d96fc9f0ed9e65c29700d6ea6410c0d210f29fe22b1742d2d725bedf7021.
Source experiments/literature/efc_input_probe.py and tests/test_efc_input_probe.py
are now bound by the completed run; do not modify them. Focused14tests passed
in1.76s (12input +2discretization), log efc-input-focused-final-20261118.txt.
Full nine-check loop COMPLETE, all exit0; 476 tests passed246.15s. Logs
input-check-{1..9}-20261118.txt and input-checks-20261118.json under
.release-audit/literature/. Scan counts unchanged206/22/36/5; not a clean-release claim.

EFC upstream currently only tested unmodified. For approved linear/all-row mode,
prefer an explicit hash-verified source-copy patch limited to KBinsDiscretizer
constructor settings; preserve native energy/class cutoff semantics and license.
Freeze original and derived source bytes and test constructor integration before
benchmark fits. Input validity is now verified; full-class resource costs remain
to be measured.

Next actual action: implement a hash-verified source-copy patch confined to the
native KBinsDiscretizer call (all rows, linear quantiles), then test native fit
against a checkpointed decomposition (same preprocessor, BaseEFC per-class fits,
same class ordering/cutoffs/energies). Source SHA for upstream classifier:
b85f95b8cd4a0e502913ac1a271f897d78167b0d0514b2ce5bb143b7daa60e4d.
Upstream .fit does MaxAbsScaler + quantile ordinal discretizer in a
ColumnTransformer, then computes max_bin globally and fits independent BaseEFCs
per sorted class. Private isolated envs exist in the two synthetic run dirs;
do not mutate those completed environments. A new source-copy/env is needed.
Propose bounded full-row preprocessing first, then smallest class (BruteForce5275)
native-fit sizing, preserving completed class models; no fit resumes internally.
UPDATE: hash-verified source copy and checkpoint decomposition implemented in
experiments/literature/efc_adapter.py; source record EFC_ADAPTER_SOURCE.json.
Only the three explicit binning keywords change upstream source. AST comparison
checks that the rest of the algorithm is unchanged. Original license retained.
Derived classifier SHA67cb85ffe27b846a488b0d57624f3e03ff452f2f0181e564a6702c30599a8582.
Synthetic native-equivalence run efc-native-equivalence-20261118t170722155558z
passed: 600rows46features3classes; exact discretization, each class's native
parameters/cutoffs, prediction energies, batching, per-class/preprocessor/model
reload. 12.985s calculation,38.275s total,peakRSS432,205,824B. Receipt
experiments/literature/EFC_NATIVE_EQUIVALENCE_SUCCESS.json pins installed native
Python/Cython binary bytes. Six focused adapter tests passed1.34s.

Native TRAIN runner experiments/literature/efc_train_probe.py implements separate
preprocessing and class-fit jobs,15min original deadlines,4GiBRSS,5GiB artifacts,
no available-RAM gates. Verified completed supervisor checkpoints can recover
an interrupted final publication; incomplete fits restart. It imports only the
hash-checked native installation from the equivalence run, leaving primary env
and completed environments unchanged. Native adapter/discretizer/TRAIN runner
and tests/test_efc_train_probe.py are NOW BOUND; do not edit without successor.
Focused train+adapter16 tests passed3.27s, compileall and22 combined focused/release
checks passed2.19s. Logs efc-train-focused-20261118.txt and
efc-train-and-release-20261118.txt; no broad rerun after these additive tests.

Full-row native preprocessing COMPLETE: efc-native-prepare-20261118t171325640901z,
all1,270,025known/benign rows46features,max_bin30; exact reload transform checked
on EVERY TRAIN row. Wall18.818s,peakRSS2,050,854,912B,minavailable1,395,896,320B,
artifacts936,034,080B. Receipt EFC_NATIVE_PREPARATION_SUCCESS.json under
experiments/literature/. Read-only completion/artifact verification passed.

Native classes1and0 COMPLETE: BruteForce5275rows7.456s peak652,234,752B;
Benign68603rows29.056s peak724,582,400B. Receipts EFC_CLASS1_SUCCESS.json and
EFC_CLASS0_SUCCESS.json under experiments/literature/. Each verified and saved.
Next four classes5,3,4,2 launched sequentially in exec session71615, each its
own900s guard. Classes5,3,4 now COMPLETE; class2(DDoS666284rows) last observed
3.1min at~0.84GiB RSS, still running. Logs efc-native-class{k}-20261118.txt under
.release-audit/literature/. Exact run paths are first log line. Source/code bound
by completed runs; do not edit adapter, discretizer, train runner or pinned tests.
All six class fits are needed before assembling the native multiclass pilot;
no unsupported performance conclusions from single-class readiness.

Native fidelity evaluator implemented in experiments/literature/efc_fidelity.py
and tests/test_efc_fidelity.py;9 focused tests passed1.94s. NOT RUN ON REAL TEST.
Full nine-check loop now running (session reported by latest tool), logs
.release-audit/literature/fidelity-check-{1..9}-20261118.txt and ledger
fidelity-checks-20261118.json. Expected full suite501tests; do not claim outcome
until actual result. Native evaluator prepares an immutable seal before any test
path resolution, verifies six models/preprocessor/native binary, scores test
parts with recoverable receipts, checks exact population/truth hashes against
the existing C03 test, and independently reconciles all stored metrics from arrays.
Native Unknown -1 maps to canonical6. No validation reads/tuning/refit; 7-label
macro-F1/confusion and same exact rate denominators as C03, no fabricated Stage1
AUROC. Native policy is not policy-matched to C03; no controlled-method ranking.

NEXT: when all six fits complete, write experiments/literature/EFC_NATIVE_FIT_SUCCESS.json
with statusCOMPLETE, prepared_run_id/prepared_completion_sha256, and classes map
keys'0'..'5' each {run_id,completion_sha256,model_sha256,cutoff}. Derive paths from
per-class log first line; verify each via train.verify_completed, stage/class and
common preprocessing. Include other resource/support provenance as useful.
Evaluator definition requires this aggregate fit record; do not invent it early.
After successful final checks, run python -B -u -m experiments.literature.efc_fidelity
--evaluate under its900s/4GiB guard; save console to ignored literature log.
Native operating-point test evaluation is part of the user-adopted first fidelity
pilot. No common-policy adaptation or full campaign is authorized by it.

No benchmark validation/test scores inspected for EFC. Dataset/input choices
are settled; no additional routine permission is required. Any newly discovered
scientific choice still requires user input.

## Current batch — CICIDS contiguous assignment metadata audit, 2026-11-18

User explicitly adopted first-valid seeded shuffles of the existing twenty
contiguous units, 12/4/4 train/validation/test units, minimum 200 rows per retained
family in every split, at most 10,000 attempts per seed. Approved first step is
metadata-only audit of partition seeds 123–127; no materialization or fitting.
Duplicates/exhaustion are reported without redraw/fallback. RNG implementation:
NumPy RandomState(seed).permutation(20), sorted units within assigned buckets.
Implementation: experiments/cicids_partition/seeded_assignments.py and
tests/test_cicids_seeded_assignments.py. Atomic per-seed checkpoints and partial
final-publication recovery retain the original deadline and immutable results.
Available RAM is telemetry only; retain 900 s, 4 GiB RSS, disk/artifact limits.
Focused checks: 23 passed in 1.32 s (17 new metadata/recovery tests plus six
release-content checks); .release-audit/cicids-partition/contiguous-focused-20261118.txt.
Real metadata audit COMPLETE, run contiguous-seeds-20261118t062656935054z,
0.147181 s, peak RSS 59,392,000 bytes. All five distinct and support-valid;
first valid draws for seeds 123–127: 168, 237, 107, 263, 244. Read-only completion
verification and independent Python-integer support reconciliation passed.
Compact evidence experiments/cicids_partition/CONTIGUOUS_ASSIGNMENT_SUCCESS.json;
details docs/RECKONING_CICIDS_PARTITION_PREFLIGHT.md. Source/test bytes now frozen
by receipt; do not mutate. No raw observations read, datasets written or models fit.
Full verification: 462 tests passed in 205.93 s; required checks 3–9 all exit0,
ledger final-checks-20261118.json under .release-audit/cicids-partition/.
Locked sync passed in replay-checks ledger; compileall repeated successfully
after the new tests. Scanner has prior content findings; no clean-release claim.

EFC static source audit complete: author package commit
2b935be347abf7daf4420989ef391436db418eac, package0.1.0, paper arXiv2109.11224v3.
Source receipt experiments/literature/EFC_SOURCE_AUDIT.json; no code execution
at audit time. User subsequently explicitly chose native fidelity baseline first,
separately labeled common-policy adaptation later. No benchmark fitting yet.
Synthetic compatibility COMPLETE: efc-synthetic-20261118t063726546228z;
600 generated rows, eight features, three classes, seed123, one worker, native
parameters. Finite parameters/energies and exact batch/save-reload equality passed.
Fit/checks0.188 s; including isolated install/build36.259 s, peakRSS324,780,032 B.
Receipt experiments/literature/EFC_SYNTHETIC_SUCCESS.json. Source/model/worker/
phase hashes verified. Ignored diagnostic script .release-audit/literature/
efc_compatibility.py, log efc-compatibility-second-20261118.txt in that directory.
First attempt failed before environment creation on Windows newline/hash drift;
corrected to write_bytes, failed evidence retained. No benchmark observations read.
This is a local diagnostic, not yet a packaged fault-tolerant benchmark runner.

Subsequently resolved: user adopted all TRAIN rows + linear quantiles and no
subsampling, explicitly overriding modern defaults. Native energy/cutoff rule
is also adopted. Current execution status is at the top of this note.
OpenMax source audit complete as CANDIDATE ONLY, not selected: Python2/image-network
assumptions need an explicit tabular architecture and comparison contract.
Receipt experiments/literature/OPENMAX_SOURCE_AUDIT.json. Upstream code not executed.
Newer Vastlab/libMR README claims BSD3 supersedes earlier licensing, but its
LICENSE.TXT still contains old restricted-use paragraphs; source/legal packaging
provenance remains unresolved, not an automatic algorithm rejection.
Read docs/RECKONING_LITERATURE_METHOD_PREFLIGHT.md for scope and sources.
Post-documentation checks: check_links passed; six release-content tests passed
in0.62 s. No experiment remains running after these completed diagnostics.

## Completed historical assignment replay, 2026-11-18

User resumed with "got cut off, continue". New quota window was available
(4% used, reset epoch 1795000236). Preserve the existing quota stop condition;
bounded jobs may finish after the agent session as previously authorized.

Recovered exact materializer bytes from Git bff66918cbf28d6a1d70e17bcfaeb51a7f11057b
to ignored `.release-audit/cicids-partition/materializer-bff6691.py`, SHA-256
88c3091f14997f7cfc059d415284b00fcaf49b131ae859a7a1155b7136646e3d.
The bounded wrapper calls only build_assignment with the original seed 123,
minimum support 200, and chunk size 200000. Destructive main/materialize entry
points are disabled; no split materialization or model fitting is permitted.
This is reproduction of the existing recipe, not approval of a new partition
distribution or repeated-seed scientific campaign.

Implementation: experiments/cicids_partition/replay.py and
tests/test_cicids_partition_replay.py. Focused verification: 12 passed in 2.78 s,
including exact source hash, destructive-entrypoint rejection, deterministic
assignment/group integrity on generated data, discrepancy reporting, checkpoint
reuse/corruption, stop conditions, original deadline and real orphan cleanup.
Log: `.release-audit/cicids-partition/replay-focused-20261118.txt`.

Real read-only replay COMPLETE in 15.856 s, peak RSS 1,007,697,920 bytes;
all 13 historical aggregate/support/repair/feature-order checks matched.
Read-only completion verification passed; compact evidence REPLAY_SUCCESS.json
under experiments/cicids_partition/. Console:
`.release-audit/cicids-partition/replay-run-20261118.txt`.
Envelope: 900 seconds shared, worker RSS <=4 GiB, disk >=20 GiB start/10 GiB
throughout, new artifacts <=5 GiB; no available-RAM gates or GPU work. Input
parts are hashed before/after assignment. Output is only external audit arrays,
new group-to-split checksum, and comparisons with saved aggregates/support/
repair log/feature order. Matching aggregate evidence alone cannot prove equality
to an unavailable historical row-assignment checksum, nor audit hash collisions.
Do not edit completed-run-bound replay source/tests.
Broader checks initially found one failure (444 passed): the archival wrapper
conflicted with the public dynamic-legacy-loading ban. The release test now
permits only this exact hash-pinned internal audit wrapper; no directory waiver
and no public workflow exception. Focused release checks passed after the fix.
Original failure retained in .release-audit/cicids-partition/replay-check-3-20261118.txt.

## Current batch — 2026-11-18 post-seal continuation

User said Proceed after verified producer completion. Continue the fixed C03
pilot with a separate prospective contract for selection/test evaluation. No
producer-bound source, parameters, input bytes or historical evidence is edited.
New code is confined to `experiments/c03/postseal/` and
`tests/test_c03_postseal.py`. The extension freezes source and test-manifest
identities before validation selection, writes both arms in one immutable seal,
then uses separate workers for selected-component test inference and paired
analysis. Test input uses ordered 8,192-row batches, unchanged preprocessing,
and exact source-row/label pairing. Full model counts and CPU/GPU settings stay
fixed. No refit, calibration, threshold tuning on test, or new inferential claims.

Operational envelope: one hour shared across selection, test scoring and analysis;
4 GiB worker RSS, 5 GiB new artifacts, original GPU/disk limits, available RAM
telemetry only. Recovery keeps the original deadline and completed score receipts;
interrupted scoring restarts only that component. Partial selection/evaluation
publication can finish its final receipt after content verification, without
overwriting existing decisions or results. The producer's completed ten-hour
budget is not reused or extended.

Verification: 22 focused checks passed in 32.90 s; all 433 repository tests passed
in 224.73 s, all nine required commands exited zero. Ledger:
`.release-audit/c03/postseal-checks-20261118.json`. Release-scan content findings
are unchanged; the fifth filename finding is `postseal/supervisor.py`, a reviewed
process-supervision source-name match rather than private material.

Real post-seal run `postseal-20261118t055250590478z` COMPLETE in 330.005 seconds.
Both arms sealed before test access; selected RF Stage-1 and shared Stage-2 scored
all 1,190,899 ordered test rows; 16 paired policies/32 arm outcomes computed.
Read-only `supervisor.verify_completed` passed. Independently reconciled all 32
confusion matrices, accuracy/macro-F1 and rate numerators/denominators.
Evidence: `.release-audit/c03/postseal-run-20261118.txt`,
`.release-audit/c03/postseal-verified-20261118.json`, compact immutable
`experiments/c03/postseal/PILOT_SUCCESS.json`, and readable
`docs/RECKONING_C03_FIXED_PILOT_RESULTS.md`.

Both arms selected native-weight RF and Stage-1 threshold 0.9166666666666666.
At primary 3% known-validation budget, visible-minus-blind macro-F1 is +0.141488
(confidence), +0.138680 (margin), +0.153580 (entropy), +0.159330 (family conditional).
All 16 predeclared cells are feasible in both arms and have positive macro-F1
deltas; no independent-replication or population causal inference is implied.
Unknown support is 176,511; known-all denominator 1,014,388. False-unknown costs
are retained alongside detection gains. No uncertainty/margin was added post hoc.
Peak worker RSS 2.870 GiB, artifacts 57.12 MiB. Available RAM transiently fell to
4,096 bytes; available-RAM gate remains removed as authorized, other guards held.

Next: remaining P2 CICIDS partition/method feasibility and P1 audit completion.
Do not expand C03 seeds/holdouts, choose new grouping rules/methods, or freeze the
full campaign without resolving those scientific specifications. Producer and
post-seal source bytes are now bound by completed evidence; use additive modules
or explicit successor scopes for future changes.

## Active autonomous session — 2026-11-17

- User withdrew both available-RAM gates (6 GiB start and 2 GiB runtime floor)
  and granted autonomy to run necessary experiments/scripts to advance the plan.
  Continue logging available RAM; retain the 4 GiB worker RSS ceiling and GPU,
  disk, time and artifact supervision. Preserve prior limits as historical evidence.
- Stop active agent work when the CURRENT Codex five-hour quota is exhausted or
  resets: reset epoch 1794965344 (2026-11-18 01:29:04 UTC). Initial usedPercent 6.
  Use `mcp__codex_app__get_usage_limits` at milestones; do not roll into a new window.
  User explicitly allows already-started jobs to finish under their own fixed
  resource/time budgets and write checkpoints after the session ends.
- Operational implementation, probes and necessary fixed-protocol experiments
  no longer require repetitive per-batch permission. Ask first for genuinely
  unresolved scientific choices; do not change scientific settings, labels,
  thresholds or historical evidence by inference. No optional dataset-role or
  full-campaign scientific freeze is inferred from execution autonomy.
- First action: remove available-RAM gates in the successor lifecycle runner,
  verify, execute/review the real-data lifecycle rehearsal, then advance the fixed
  C03 pilot. Prior lifecycle sources were hash-verified and copied into ignored
  `.release-audit/c03/lifecycle-before-20261117/` before editing. New policy module:
  `experiments/c03/engineering/resource_policy.py`. No prior probe runtime changed.
- Lifecycle rehearsal COMPLETE: `lifecycle-20261117T204538381720Z`, 515.889 s,
  all ten fit/score phases, 700,000 validation rows per component. Peak worker
  1,984,114,688 bytes; whole-device GPU 979,369,984 bytes. Read-only `--resume`
  verification passed. Additive compact receipt:
  `experiments/c03/engineering/LIFECYCLE_RAM_POLICY_SUCCESS.json`.
- Startup-only GPU query timeout increased from 2 to 10 seconds after a measured
  cold query took 2.625 s (subsequent queries 0.078/0.079 s). Live query timeout
  remains 2 seconds. Focused lifecycle suite: 53 passed in 49.30 s; evidence
  `.release-audit/c03/lifecycle-startup-policy-tests-20261117.txt`.
- New fixed scientific producer: `experiments/c03/pilot/{contract,preparation,models,runner}.py`.
  Fresh preparation, full historical ensemble counts, two CPU workers/GPU 0,
  ten-hour original deadline, 15 GiB new artifacts, 30 GiB disk start. Other
  retained guards unchanged. No selector/test execution. `--recover RUN` preserves
  completed stages, `--resume RUN` verifies completion without execution.
- Focused pilot tests: 21 passed in 20.84 s, including full-count generated models,
  fresh-preparation equivalence, actual isolated fit/score workers, all retained
  resource stop conditions, corruption rejection and fit reuse. Log:
  `.release-audit/c03/pilot-focused-tests-20261117.txt`.
- All nine required repository commands exited 0. Full suite: 410 passed in
  166.07 s. Check ledger: `.release-audit/c03/pilot-checks-20261117.json`.
  Release scan still reports 206 path, 22 email, 36 private-network and four
  filename findings, unchanged from the prior scan and outside this batch's
  changed files; this is not a clean public-release certification.
- First launch at 21:04:15 UTC stopped before run creation: uppercase timestamp
  characters failed the existing lowercase identifier validator. Corrected only
  timestamp naming and added an actual run-creation regression test. Focused
  pilot suite: 22 passed in 22.33 s, evidence
  `.release-audit/c03/pilot-startup-fix-tests-20261117.txt`; earlier unaffected
  repository check evidence retained. Both attempts' logs preserved.
- Corrected scientific producer launched detached/hidden at 2026-11-17
  21:05:42 UTC, launcher PID 51192. Process identity and stdout/stderr logs:
  `.release-audit/c03/pilot-full-producer-process-v2-20261117.json` and
  `.release-audit/c03/pilot-full-producer-v2-20261117.{stdout,stderr}.txt`.
  External run: `pilot-producer-20261117t210546664156z` in the approved sibling
  `VERA-IDS-local-runs/c03/`. Original deadline: 2026-11-18T07:05:46.940572+00:00
  (2026-11-18 10:05:46 Asia/Amman). Fresh preparation is active; no model fit
  checkpoint yet. At last quota check the current window was 96% used.
  Current plan SHA: `6b54cfd10448ff4f4cf9cc601e3269debe14f95e78999b0378ffba955d057bdd`.
  Additive fix receipt: `experiments/c03/pilot/STARTUP_FIX_PROVENANCE.json`,
  linked to original `experiments/c03/pilot/PROVENANCE.json`.
  Do not modify the producer plan's bound source/test files while it runs.
  It may finish after the active agent quota ends; its own original ten-hour
  deadline and resource guards remain in force. Next action: confirm fresh
  preparation identity checks and monitor the first full RF fit.

## Scope, decisions and source

- Continue the complete Reckoning programme; ask before unresolved scientific or
  consequential scientific choices. Full campaign remains unfrozen; fixed C03
  scientific fitting is authorized by the November 17 continuation above.
  Source HEAD: `ca13f1097f851ce6693658ccc0c6dd35480fabd6`.
- Canonical scope: `docs/RECKONING_MASTER_CAMPAIGN_PLAN.md` and
  [decision log](https://docs.google.com/document/d/1BNNXb3zCEvhKLsmnn8K-4UOR9GaZrqhYQ5SEcnZ9Amk/edit).
- Shared chat could not be fetched. Original producer contract remains missing;
  user approved its reconstruction, then Batch 1 and sibling output location
  `VERA-IDS-local-runs/c03/`, then Batches 2 and 3 and continuation after interruption.
- Current contract: `docs/RECKONING_C03_SCORE_PRODUCER_CONTRACT_DRAFT.md`.
  Batches 1–3 are approved for synthetic verification; full-fit resources/execution
  were separate gates historically; November 17 autonomy supersedes these
  operational gates for the fixed pilot.

## Preserved evidence and organization

- User approved relocating 21 loose C03 files. Tools/policy/tests now under
  `experiments/c03/`; historical notes in its `docs/`; private manifests/logs
  and exact original backups in ignored `.release-audit/c03/`.
- Scientific policy SHA: `bf56f938d1fb193d2b383deeeec29f9d84759a4be829f6a0011f1bc20fe5e099`.
- Original input snapshot: `13e667a86a6872ff6192d1662085383e2b526752ceb783f8fc372b51487de9f4`.
  Original manifest unchanged; relocation snapshot:
  `39be8e71ddd5af58a312ef97da2cad994eacd49e5d2d97491465f3ae012630ad`.
  `experiments/c03/RELOCATION.json` records migration and code hashes; 369 data
  files were byte-verified unchanged. Later producer/engineering subfolders
  have separate additive provenance and do not rewrite that baseline.
- Protected research, historical releases and scientific stored values unchanged.
  No commits or external document edits performed. All current additions untracked.

## Engineering memory probe (completed 2026-11-06)

Original worker recovered with matching SHA
`c8f2145e6ca1327f33d2ccd5053f1072bd672184a8ef9df164a7dbfd7cb7756a`.
A Windows check proved launcher PID != worker PID; added a separate supervisor
that monitors/stops descendants. Original worker bytes preserved.

Command: `.venv/Scripts/python.exe -B -u experiments/c03/engineering/guarded_memory_probe.py --probe`.
PASS: 30k/80k TRAIN-only rows, 8 trees each; fits 1.59/6.14 s; tree storage
1.75/3.58 MiB; peak summed tree RSS 0.599 GiB; minimum available physical RAM
3.788 GiB; total 14.52 s. No validation/test reads or persistent models.
Limits: 3 GiB minimum start, stop below 1.5 GiB available or above 3 GiB tree RSS;
900-second ceiling. Five supervisor tests PASS, including real child cleanup.
Log: `.release-audit/c03/memory-probe-2026-11-06.txt`.
Code/log hashes: `experiments/c03/engineering/PROVENANCE.json`.
This does NOT certify full-size RF/XGB feasibility. Bare python is 3.11;
use the existing `.venv` Python 3.12.10 environment.

## Batch 1 (completed 2026-11-06)

Producer plan, sidecar row identities, historical-sampler differential tests,
five-component map, shared RF2, immutable checkpoint/resume contracts.
36 focused tests; 150 total; all nine required repository commands PASS.
Evidence: `.release-audit/c03/producer-checks.json`, associated log files,
`experiments/c03/producer/PROVENANCE.json`.
External run `c03-foundation-20261106` contains only `plan.json` and `run.json`.
Its plan hash is `6bfc0d9adf98435cfab940979f9b7f8e556ea881b28c32216ffd770fd88b8c55`.
Preserve it: subsequent code changes intentionally invalidate current resume.

## Batch 2 (completed 2026-11-07)

- Added `producer/artifacts.py`, `sources.py`, `models.py`, `validation.py`,
  `sealing.py`; updated plan/checkpoint handling for explicit synthetic mode.
- Actual historical train-only preprocessing, deterministic train/val sampling,
  strict supervised LOAO, shared CSR tensor/row/weight hashes, five sequential
  model fits, native XGB/local RF serialization, class-aligned validation scores,
  verified reload/resume, existing selector, atomic both-arm selection and final seal.
- Execution restricted to generated external fixtures: <=10k source rows per
  split, <=8 trees, one worker, CPU XGB. No benchmark fitting command or evaluator.
  Tests use 1,750 rows/split and three trees/model; poisoned test Parquet is unused.
- Seal explicitly says synthetic-only, evaluation unauthorized, no evaluator
  revision. It checks current code and all input/model/score dependencies;
  incomplete stages fail closed without auto-retry. Hashes are not signatures.
- New `tests/test_c03_producer_validation.py`: 22 cases. Combined with foundation:
  58 PASS (10.97 s). Tests include model reload equality, shared RF2 object,
  hidden-U perturbation preserving models/K scores/blind selection, class order,
  insufficient support, source/tensor/score/model/seal drift and interrupted seal.
- Successor code/test record: `experiments/c03/producer/BATCH2_PROVENANCE.json`.
  Current no-fit plan: `.release-audit/c03/batch2-plan.json`, SHA
  `473c6a464ecd08b7108ada84c423ea83d094fd6fb4dc81b7e6c7716aa9bc1f71`.
- Required check evidence: `.release-audit/c03/batch2-check-*.txt` and
  `batch2-checks.json`: all nine commands PASS, exit 0. Full pytest: 172 PASS
  (25.79 s). Final code hashes and `git diff --check` verified.

## Batch 3 (completed 2026-11-07)

- Added `producer/evaluator.py` and `tests/test_c03_producer_evaluator.py`;
  extended `sealing.py` to bind a prospective synthetic test inventory and
  evaluator revision. Old seals cannot be upgraded; benchmark mode is absent.
- Separate-process evaluator verifies the seal before test-path resolution,
  checks model/source bytes and shared row identity, preserves exact rejection
  boundaries, and writes 16 paired descriptive results with a final digest receipt.
  Fixed seven-class metrics include explicit rate denominators, defined Stage-1
  attack-score AUCs, undefined cases, feasibility transitions and signed deltas.
- Combined focused suite: 69 PASS (36.91 s), command:
  `.venv/Scripts/python.exe -B -m pytest tests/test_c03_producer_evaluator.py tests/test_c03_producer_validation.py tests/test_c03_producer_foundation.py -p no:cacheprovider -q -x`.
  A 21-row synthetic test fixture exercises the full separate-process path.
- Full pytest: 183 PASS (43.03 s). All nine required repository commands PASS,
  exit 0; evidence: `.release-audit/c03/batch3-checks.json` and `batch3-check-*.txt`.
  Additional producer compile check and `git diff --check` passed.
- Additive record: `experiments/c03/producer/BATCH3_PROVENANCE.json`; earlier
  provenance is preserved. No-fit plan: `.release-audit/c03/batch3-plan.json`, SHA
  `185025edbd9a1f154c3bf820fa3e01ef722c1ea2bc04bd6c526c392883dfd70a`.

## Next executable action and limits

Batches 1–3 provide a verified synthetic fit-to-sealed-evaluation path. The
2026-11-07 execution review is now in the existing reconstructed contract:
source/plan hashes and pinned environment match Batch 3; all 183-test/nine-command
evidence remains applicable. No implementation change or observation read/fit.
Effective constructors were inspected: RF 24 workers; XGB CUDA/24 workers.
Laptop: 15.43 GiB OS-visible RAM, only 2.53 GiB available at inspection; GPU
8,151 MiB total/7,590 MiB free; output volume approximately 463.66 GiB free.
Evidence: `.release-audit/c03/execution-review-20261107.json` and hardware snapshot.

User approved the local TRAIN-preparation readiness batch. Concrete scope in
contract: guarded fresh
worker, unchanged historical TRAIN preprocessing/sampling/tensors, no validation/
test reads or classifier fitting, 900 seconds, start >=6 GiB free RAM, stop at
>4 GiB worker-tree RSS or <2 GiB free RAM; 5 GiB artifact limit, 20 GiB disk start
and 10 GiB disk floor. These are approved sampled limits, not hard OS bounds.
Implemented `experiments/c03/engineering/preparation_probe.py` and
`tests/test_c03_preparation_probe.py`. Focused 17 tests PASS (4.57 s): exact sampler
comparison, historical preprocessing on fixtures without validation/test reads or
classifier construction, source/count drift, all resource thresholds, real child
cleanup, monitor error/interruption, and completion only after a valid result.
All nine required commands PASS, exit 0, with evidence at
`.release-audit/c03/preparation-checks.json` and `preparation-check-*.txt`.
Full pytest: 200 PASS (39.58 s); additional probe compile and diff checks PASS.
Batch 3 plan/source hashes verified unchanged. Additive implementation/result
record: `experiments/c03/engineering/PREPARATION_PROVENANCE.json`.

Approved attempt command:
`.venv/Scripts/python.exe -B -u -m experiments.c03.engineering.preparation_probe --probe`.
Result: exit 3, START_REFUSED, 4,414,869,504 bytes (4.11 GiB) available RAM <6 GiB.
Evidence: `.release-audit/c03/preparation-attempt-20261107.txt`.
No worker launched, no external run created, no benchmark observations opened.
Implementation is complete; real TRAIN preparation feasibility remains unmeasured.
User-requested recheck on 2026-11-08 also returned exit 3, START_REFUSED:
6,039,601,152 bytes (5.62 GiB) available RAM <6 GiB. No worker launched or
benchmark observations opened. Evidence:
`.release-audit/c03/preparation-attempt-20261108.txt`.
User requested standalone CMD/PowerShell commands if still blocked; provided the
same guarded `--probe` entrypoint. Await manual output or a request to recheck;
the original resource limits and approval remain in force.
Manual run subsequently started with 6.24 GiB available RAM and stopped after
354.06 s at the 4 GiB worker-tree RSS limit (sampled peak 4.1196 GiB). Minimum
available system RAM was 3.2236 GiB, above the 2 GiB floor. Run ID:
`prep-readiness-20261107T214712540659Z` (2026-11-08 in Asia/Amman).
Verified request source record matches current code and no completion receipt
exists. Stages show TRAIN verification and preprocessing completed, sampling/
support checks completed, then stop during `stage1_tensor`; no Stage-2 tensor
or classifier fit. Evidence hashes and monitor samples:
`.release-audit/c03/preparation-manual-review-20261108.json`.
This is observed failure of the current preparation path within the 4 GiB worker
envelope, not a machine-wide memory-floor failure or full-size feasibility proof.
User then directed proceeding with the allocation review. Implemented in the
engineering probe only: unchanged historical transform over 8,192-row blocks,
two-pass count/fill into exact-size CSR arrays, no full selected feature-frame
copy and no retained list of block tensors. Both final training tensors remain
live. Protected transformer, producer sources, rows, dtypes, masks, seeds, weights
and resource limits unchanged. Before-change code/test copies preserved in
`.release-audit/c03/preparation-before-memory-change/`.
Focused 26 tests PASS (6.29 s), including byte-for-byte data/indices/indptr
equivalence on numeric/categorical/nonfinite inputs and both prepared tensor,
row-identity and label hashes. Added stage metadata events after tensor assembly.
All nine required repository commands PASS, exit 0; full pytest 209 PASS (27.67 s).
Evidence: `.release-audit/c03/preparation-memory-checks.json` and associated logs.
Original producer plan hash unchanged; before-change backups match predecessor
provenance. Additional probe compile and diff checks PASS.
Revised probe attempt returned exit 3, START_REFUSED, with 4,586,942,464 bytes
(4.27 GiB) available RAM. No worker or benchmark observation read occurred.
Evidence: `.release-audit/c03/preparation-memory-attempt-20261108.txt`.
Additive provenance: `experiments/c03/engineering/PREPARATION_MEMORY_PROVENANCE.json`.
Manual revised run `prep-readiness-20261107T220732281954Z` completed successfully
(2026-11-08 Asia/Amman). Verified request source hashes, completion receipt,
supervisor/result hashes and preprocessor artifact bytes. Peak sampled worker-tree
RSS 2.8859 GiB; minimum available RAM 4.0977 GiB; wall 360.89 s; worker exit 0.
All approved limits retained. Verified 3,676,338 TRAIN source rows; 1,500,000 sampled
rows; Stage-1 1,270,025 x 46 and Stage-2 1,201,422 x 46, both float64. Both final
tensors remained live. Sampling accounted for 334.875 s; Stage-1/2 tensor and
metadata work took 10.031/9.203 s. No validation/test read or classifier fit.
Evidence: `.release-audit/c03/preparation-memory-success-review-20261108.json`;
immutable successor result `experiments/c03/engineering/PREPARATION_MEMORY_SUCCESS.json`.
Earlier refusals and failed-run records remain unchanged. Existing 209-test/nine-
command evidence still applies: implementation source hashes match tested code.
TRAIN preparation feasibility is now established for this measured envelope;
these probe artifacts are not reusable scientific checkpoints.
User directed proceeding. Integrated the unchanged bounded transform into shared
`producer/transforms.py`; `producer/validation.py` now uses it for both TRAIN
tensors and validation; the engineering probe imports the same implementation.
Synthetic-only gates remain intact. Before-change copies and plan preserved at
`.release-audit/c03/bounded-integration-before/`; old plan/seal hashes are historical.
New differential test checks all three prepared tensors against the original
full-frame path. Combined probe/producer/evaluator/foundation suite: 96 PASS
(16.81 s). Full pytest 210 PASS (23.58 s); all nine required commands PASS,
exit 0. Evidence: `.release-audit/c03/bounded-integration-checks.json` and logs.
Extracted function body is AST-identical to the successful probe; extra compile
and diff checks PASS. Successor source/plan record:
`experiments/c03/producer/BOUNDED_TRANSFORM_PROVENANCE.json`; no-fit plan saved at
`.release-audit/c03/bounded-integration-plan.json`. No new benchmark reads/fits.
Next scope is drafted in the existing contract: combined real TRAIN/validation
preparation, serialization and reload only, same 900-second/6-GiB-start/4-GiB-RSS/
2-GiB-free/5-GiB-artifacts limits; no classifier, selection, test or GPU work.
Integration is complete. User approved the combined real TRAIN/validation
preparation-and-reload probe, with the existing limits and no classifier/selection/
test/GPU work. Implemented `engineering/combined_preparation_probe.py`, dedicated
engineering checkpoint/receipt scope, shared preparation with stage events,
release-before-reload identity verification, and scope-bound supervision.
Synthetic fitting/selection/evaluation gates remain closed to engineering plans.
Atomic-publication disk accounting now deduplicates hard links and tolerates
disappearing temporary names. Pre-change sources/plan are preserved under
`.release-audit/c03/combined-preparation-before/`.
Focused 108 tests PASS (21.28 s); full pytest 222 PASS (27.77 s). All nine required
commands PASS, exit 0; evidence `.release-audit/c03/combined-preparation-checks.json`
and associated logs. Additional compile, CLI help and diff checks PASS.
Approved attempt on 2026-11-08 returned exit 3, START_REFUSED: 4,957,786,112 bytes
(4.62 GiB) available RAM <6 GiB. No worker, external run or benchmark read occurred.
Evidence: `.release-audit/c03/combined-preparation-attempt-20261108.txt`.
Successor record: `experiments/c03/engineering/COMBINED_PREPARATION_PROVENANCE.json`;
engineering plan: `.release-audit/c03/combined-preparation-plan.json`.
Manual combined run `combined-prep-20261107T223843911512Z` completed on 2026-11-08
Asia/Amman. Verified current request/plan source hashes, completion receipt and
every saved artifact hash. Reload identity checks passed. Peak worker-tree RSS
3.7078 GiB; minimum available RAM 3.2346 GiB; wall 437.062 s; artifacts 2.5792 GiB.
Tensor shapes 1,270,025 x 46, 1,201,422 x 46 and 700,000 x 46, all float64.
Evidence `.release-audit/c03/combined-preparation-success-review-20261108.json`;
immutable result `experiments/c03/engineering/COMBINED_PREPARATION_SUCCESS.json`.
Combined preparation/reload feasibility is established in this measured envelope;
no classifier fit/selection/test access occurred and arrays remain engineering-only.

User requests progress indicators for subsequent work. Added shared supervisor
stage/resource output on stderr at stage changes and 30-second intervals; final
JSON remains stdout. Displays elapsed/budget, worker RSS, available RAM and artifact
size, with no guessed ETA/percentage. Handles incomplete stage-log writes.
Pre-display-change sources saved at `.release-audit/c03/progress-before/`.
Focused supervisor/combined tests: 40 PASS (7.03 s); full suite 224 PASS (30.36 s).
All nine required commands PASS, exit 0; evidence `.release-audit/c03/progress-checks.json`
and logs. Additional compile/diff checks PASS. Successor display provenance:
`experiments/c03/engineering/PROGRESS_PROVENANCE.json`.
Progress display is complete. User then requested proceeding with fault tolerance,
resume and checkpoints. Added `producer/locking.py`, `recovery.py`, `isolated.py`:
OS-managed writer/pipeline/component locks, one synthetic worker per model, only
that component's TRAIN tensor plus validation loaded, explicit recovery forks
that copy verified completed work and preserve failures, and missing-seal receipt
recovery without reselection. Preparation's final receipt now commits after its
input binding. Combined probes support explicit same-revision completed-preparation
resume in a fresh engineering attempt. Incomplete preparation restarts; no mid-fit
continuation, silent scope promotion, deletion or automatic retry.
Before-change source/plan copies: `.release-audit/c03/resilience-before/`.
Focused 93 tests PASS (37.63 s), including actual killed component worker, live-lock
recovery rejection, prefix reuse, corruption rejection and five isolated workers.
Full pytest 235 PASS (53.18 s); all nine required commands PASS, exit 0. Evidence:
`.release-audit/c03/resilience-checks.json` and logs. Compile, CLI-help and diff
checks PASS. Successor record `experiments/c03/producer/RESILIENCE_PROVENANCE.json`;
no-fit plan `.release-audit/c03/resilience-plan.json`.
New code invalidates older resume fingerprints;
historical evidence is preserved under its original revisions.
User approved the next resource measurement: one full-row Stage-2 RF
engineering sizing fit (8 trees, 1 worker, seed 133, no scoring/retained model),
same resource limits, explicit hash-verified engineering import of the successful
probe's Stage-2 array. Ordinary resume protections stay intact. Implemented
`engineering/rf_stage2_sizing_probe.py` and dedicated synthetic checks for source
drift, corrupted import/identities, no-scoring eight-tree fit, start refusal,
scope-specific completion and completed-result reuse. Focused 52 tests passed
(9.11 s). Full suite: 247 passed (46.09 s); all nine required repository commands
passed, exit 0. Evidence: `.release-audit/c03/rf-sizing-checks.json` and logs.
Extra compile, CLI help and diff checks passed. The single approved guarded
attempt returned START_REFUSED (exit 3): 3,487,301,632 bytes / 3.25 GiB available
RAM, below the unchanged 6 GiB start gate. No worker, training-array load or fit
occurred. Evidence: `.release-audit/c03/rf-sizing-attempt-20261108.json`.
Successor record: `experiments/c03/engineering/RF_STAGE2_SIZING_PROVENANCE.json`;
metadata-only plan: `.release-audit/c03/rf-sizing-plan.json`.
User's manual follow-up `rf-stage2-sizing-20261108T113317773613Z` completed.
Verified on 2026-11-08: current source/request identity, completion receipts,
original import receipts and Stage-2 array bytes, diagnostic counts and all 1,620
monitor samples. Completed-result `--resume` passed without refitting.
Wall 543.297 s; fit 531.078 s; peak RSS 1.6390 GiB; minimum available RAM 4.8420 GiB.
Eight trees contain 50,826,256 bytes of node/value storage (48.47 MiB). All sampled
resource limits held. No scoring or model retention. Evidence:
`experiments/c03/engineering/RF_STAGE2_SIZING_SUCCESS.json` and
`.release-audit/c03/rf-sizing-success-review-20261108.json`.
Linear estimates only: 300 trees at one worker ~5.53 h and ~1.775 GiB tree buffers;
these do not establish total full-fit memory or runtime. User approved one
comparable eight-tree, two-worker engineering probe under the same limits.
Added `engineering/rf_two_worker_sizing_probe.py` and its dedicated test module;
original one-worker sources/evidence remain unchanged. The comparison pins the
baseline result and requires the same input identity and effective parameters
except `n_jobs=2`. Focused 48 tests passed (7.85 s): actual tiny one/two-worker fits,
no scoring/serialization, baseline/import drift, pre-start gate, request drift,
corrupt/incomplete recovery and one-worker-result rejection. Full suite: 255 passed
(60.84 s); all nine required commands passed, exit 0. Compile, CLI help and diff
checks passed. Evidence: `.release-audit/c03/rf-two-worker-checks.json` and logs.
The approved attempt returned START_REFUSED, exit 3: 3,734,265,856 bytes / 3.48 GiB
available RAM below the unchanged 6 GiB gate. No worker, array load or fit occurred.
Attempt evidence: `.release-audit/c03/rf-two-worker-attempt-20261108.json`;
successor record: `experiments/c03/engineering/RF_TWO_WORKER_SIZING_PROVENANCE.json`.
User's follow-up `rf-two-worker-sizing-20261108T115235607909Z` completed. Verified
on 2026-11-08: current source/request/completion receipts, original import bytes,
effective parameters (only n_jobs differs), diagnostic counts and 736 monitor
samples. Completed-result resume passed without fitting. Wall 245.297 s; fit
233.781 s; peak RSS 1.7536 GiB; minimum available RAM 4.2242 GiB. All sampled
limits held. Same eight tree sizes (50,826,256 bytes total), without scoring or
retained model. Evidence: `experiments/c03/engineering/RF_TWO_WORKER_SIZING_SUCCESS.json`
and `.release-audit/c03/rf-two-worker-success-review-20261108.json`.
Comparison: 2.2717x fit-speed ratio, 55.98% less fit time, +117.39 MiB peak RSS.
This is a single-run comparison, not a repeated speedup estimate or proof of
model identity. A 300-tree fit extrapolates to ~2.44 h with two workers; full-fit
memory/checkpoint/scoring feasibility remains unmeasured. No code changed;
existing 255-test evidence remains applicable. Next action: define bounded
readiness checks for remaining Stage-1 RF and XGBoost components and obtain
approval before new benchmark reads/fits. Two-worker RF is a resource candidate,
not an approved production setting. No further Stage-2 concurrency probe proposed.
User requested proceeding. Completed read-only remaining-model review; no data
arrays opened or fits run. Current plan/source identity unchanged. Evidence:
`.release-audit/c03/remaining-readiness-review-20261108.json` (constructor parameters,
installed XGBoost build and instantaneous host resources). The contract now has
a concrete proposed Stage-1 RF batch: both historical weighting profiles in
fresh sequential processes, full 1,270,025-row/46-feature Stage-1 import,
8 trees/2 workers/seed 123 each, one shared 900-second attempt, unchanged host
resource limits, per-profile diagnostic completion receipts/reuse, no scoring,
models or GPU. The inverse-family weights must be verified from the original
identity receipt; saved arrays do not contain family labels for recomputation.
User approved the Stage-1 import/two-fit scope. Implemented the separate
`engineering/rf_stage1_sizing_probe.py` and `tests/test_c03_rf_stage1_sizing_probe.py`:
two fresh sequential workers, shared deadline, per-worker start gates, completed
profile receipts and explicit verified recovery forks. Existing Stage-2 runtime
files remain unchanged. Focused checks exposed a pre-existing tiny-fit
timing-test failure (zero elapsed baseline on Windows); fixed only
`tests/test_c03_rf_two_worker_sizing_probe.py` with a deterministic clock for its
ratio assertion. Original test bytes are retained under
`.release-audit/c03/stage1-before/`. Old two-worker evidence retains its original
fingerprint; test-only drift blocks current-revision resume of that old run.
Focused 67 tests passed (22.29 s), including both actual synthetic weighting
fits and disposable process failure/deadline/recovery cases. Evidence:
`.release-audit/c03/rf-stage1-focused-fixed.txt`; original observed failure log
`.release-audit/c03/rf-stage1-focused.txt`. Full suite: 274 passed (75.83 s); all
nine required repository commands passed, exit 0. Extra compile/CLI-help/diff
checks passed. Evidence: `.release-audit/c03/rf-stage1-checks.json` and logs.
The approved attempt returned START_REFUSED (exit 3): 4,207,300,608 bytes / 3.92 GiB
available RAM, below the 6 GiB start gate. No worker, array load or fit occurred.
Attempt: `.release-audit/c03/rf-stage1-attempt-20261108.json`; successor record:
`experiments/c03/engineering/RF_STAGE1_SIZING_PROVENANCE.json`.
User's follow-up `rf-stage1-sizing-20261108T121533598855Z` completed both profiles
in 370.203 s. Verified on 2026-11-08: current source/request/profile/batch receipts,
original import bytes, tensor/label/weight identities, exact effective settings,
tree sizes and 1,099 monitor samples across the shared budget. Completed-batch
resume passed without fitting. Balanced-subsample profile: fit 159.375 s, peak
1.8309 GiB; inverse-family profile: fit 186.453 s, peak 1.6410 GiB. Batch minimum
available RAM 4.3121 GiB. No scoring, retained models or array copies in the run.
The last progress line shows only the second profile's peak, not the batch peak.
Evidence: `experiments/c03/engineering/RF_STAGE1_SIZING_SUCCESS.json` and
`.release-audit/c03/rf-stage1-success-review-20261108.json`.
All three RF components now have full-row eight-tree measurements. Linear
300-tree fitting estimates at two workers: 1.66 h balanced Stage-1, 1.94 h
inverse-family Stage-1, 2.44 h shared Stage-2; ~6.04 h total, excluding prep,
save/reload, scoring and XGBoost. These are estimates, not full-fit time/memory
qualification or production approval. No executable code changed; previous
274-test evidence remains applicable. Next action: prepare a bounded synthetic
CUDA readiness proposal with CPU-worker and VRAM limits, seek approval before
GPU execution, then consider real-data XGBoost sizing based on that result.
Installed CUDA support and a visible GPU are not evidence of a successful fit.
User requested proceeding. Completed read-only CUDA readiness review and added
a concrete proposal to the existing contract; no GPU fit or benchmark arrays
accessed. Evidence: `.release-audit/c03/cuda-readiness-review-20261108.json` and
`cuda-readiness-gpu-20261108.txt`. Proposed single synthetic batch: binary then
five-class, 4,096 generated rows/46 CSR features each, 8 boosting rounds, 2 CPU
workers, CUDA device 0, seeds 123/133, 300 seconds total. Proposed host limits:
3 GiB available to start, 2 GiB worker RSS cap, 2 GiB available floor. GPU: 6 GiB
free to start, stop above 4 GiB total device-used VRAM or below 2 GiB free;
query failure also stops. Disk: 2 GiB start/1 GiB floor/64 MiB new artifacts.
GPU telemetry is device-wide. No CPU fallback accepted, no data/model retention
or scores, explicit component-diagnostic recovery only. User approved this batch.
Implemented `engineering/cuda_readiness_probe.py` and
`tests/test_c03_cuda_readiness_probe.py`: deterministic generated inputs, round
callbacks checking fitted CUDA device/objectives, bounded whole-device telemetry,
shared deadline, fresh workers, immutable diagnostic receipts and explicit
recovery. No benchmark readers, models or generated arrays are retained. Existing
benchmark runtime files/limits remain unchanged. First 38 focused CPU/mocked-GPU
checks passed (18.61 s), including real CPU objective fits, device/fallback
rejection, GPU monitoring failure/timeout stops, partial recovery and live-lock/
corruption rejection. Final source additionally refreshes host memory after GPU
queries. Final focused 57 tests passed (31.60 s); full suite 312 passed (99.04 s).
All nine required repository commands passed, exit 0; compile/CLI-help/diff checks
passed. Evidence: `.release-audit/c03/cuda-readiness-focused.txt`,
`cuda-readiness-checks.json` and associated logs.
The single approved GPU attempt completed both objectives in 8.141 s:
`cuda-readiness-20261108T202038883116Z`. Binary fit 0.3646 s, eight trees;
five-class fit 0.5139 s, eight rounds/40 trees. Verified fitted `cuda:0` and
`grow_gpu_hist` for both, all eight round callbacks, no warnings/fallback,
generated input identities, parameters, completion receipts and 22 monitor
samples. Completed-result resume passed without fitting or GPU queries.
Peak worker RSS 375.78 MiB; peak sampled whole-device VRAM use 624 MiB.
GPU figures are sampled/device-wide and can miss short peaks. No benchmark
arrays, predictions, quality metrics or retained models were produced.
Evidence: `experiments/c03/engineering/CUDA_READINESS_PROVENANCE.json`,
`CUDA_READINESS_SUCCESS.json`, `.release-audit/c03/cuda-readiness-attempt-20261108.json`
and `cuda-readiness-success-review-20261108.json`.
CUDA runtime compatibility for these tiny objectives is established on this
machine. Next action: prepare and obtain approval for bounded full-row XGBoost
engineering sizing with explicit import/RAM/VRAM/time/round limits. Do not carry
the synthetic 3-GiB host start gate into benchmark work or extrapolate full-run
capacity/runtime from the tiny fixture. Full scientific execution is unapproved.
On 2026-11-09 the user requested continuation. Read-only full-row XGBoost review
reverified current CUDA receipts/source identity and original training metadata;
no benchmark arrays or fits. Evidence:
`.release-audit/c03/xgb-full-row-review-20261109.json`. Added concrete proposal to
the existing contract: Stage-1 1,270,025 rows then Stage-2 1,201,422 rows, 46 float64
features, original component weights, seeds 123/133, 8 rounds/2 CPU workers/CUDA 0,
fresh sequential workers and one 900-second budget. Host limits return to
6 GiB start/4 GiB worker/2 GiB available floor; GPU 6 GiB free start/4 GiB total
used cap/2 GiB free floor; disk 20 GiB start/10 GiB floor/5 GiB artifacts. Explicit
verified engineering imports, component diagnostic recovery, no scoring or
retained models. The user approved implementation and one attempt, then confirmed
continuation. Implemented `engineering/xgb_full_row_sizing_probe.py` and
`tests/test_c03_xgb_full_row_sizing_probe.py`; previous CUDA/RF runtime sources
remain unchanged. Per-component imports bind all input identities to original
evidence; Stage-2 weights are additionally recomputed by the historical helper.
Fitted device/GPU updater/round/tree counts are checked, with shared-budget
process supervision and explicit verified recovery. Initial 24 focused tests
passed (28.77 s), using generated arrays and mocked GPU workers only. Final focused
62 tests passed (41.80 s); full suite 336 passed (101.38 s); all nine required
repository commands passed, exit 0. Extra compile/CLI-help/diff checks passed.
Evidence: `.release-audit/c03/xgb-full-row-focused-20261109.txt`,
`xgb-full-row-checks-20261109.json` and logs.
The single approved attempt returned START_REFUSED (exit 3): `nvidia-smi` exceeded
the fixed two-second query timeout. The initial host snapshot also had only
4,877,365,248 bytes / 4.54 GiB available RAM, below the 6 GiB start gate. No worker,
training-array load or classifier fit occurred; no external run was created.
A read-only `--inspect` follow-up also timed out (exit 1); no probe retry or limit
change. Evidence: `.release-audit/c03/xgb-full-row-attempt-20261109.json` and
`xgb-full-row-after-refusal-inspect-20261109-errors.txt`. Successor record:
`experiments/c03/engineering/XGB_FULL_ROW_SIZING_PROVENANCE.json`.
The user's manual follow-up `xgb-full-row-sizing-20261109T143827448818Z` completed
both components in 27.218 s. Verified on 2026-11-09: current source/request and
profile/batch receipts, original TRAIN array bytes and fitted input identities,
effective settings, CUDA 0/GPU histogram updater, eight callbacks/rounds per
component and 74 resource samples. Completed-result resume passed without fitting
or GPU queries. Stage-1: 1,270,025 rows, fit 2.6033 s, peak worker 1.8885 GiB;
Stage-2: 1,201,422 rows, fit 2.9050 s, peak worker 1.8472 GiB. Peak sampled
whole-device VRAM 664 MiB; minimum available host RAM 4.6261 GiB. Eight binary
trees and 40 five-class trees; no warnings, CPU fallback, scoring, models or new
array artifacts. Evidence: `experiments/c03/engineering/XGB_FULL_ROW_SIZING_SUCCESS.json`
and `.release-audit/c03/xgb-full-row-success-review-20261109.json`. Original refused
attempt evidence remains intact. No executable code changed; prior 336-test
evidence remains applicable.
All five model components now have short full-row fit measurements. They do not
qualify full 300-tree RF or 800/1,400-round XGBoost fitting, model save/reload or
validation scoring. XGBoost times include data construction/initialization; do
not scale them naively to full-round forecasts.

2026-11-09 lifecycle review completed (read-only code review and compact-evidence
arithmetic; no model/data execution). Current validation loader retains TRAIN and
validation together; `_fit_component` scores before saving and commits a single
model/score receipt. Recovery cannot reuse a fit independently of scoring.
The contract's new "Lifecycle execution review" section proposes versioned
fit/save and separate reload/score receipts, fresh workers, component-boundary
recovery, generated-data verification, then one five-component eight-tree/round
engineering rehearsal under the same 15-minute / host 6-4-2 GiB / GPU 6-4-2 GiB /
5 GiB new-artifact caps. This newly retains models and scores 700,000 validation
rows. The user subsequently approved implementation, synthetic verification and
one bounded rehearsal. No selection, test access or scientific promotion.
Budget must include recovery consumption, not reset on fork. Source-pinned
historical runtime/evidence should stay intact through additive implementation.
RF full-fit estimate is 6.0376 h; estimated RF buffers 3.6651 GiB, plus existing
preparation and score data gives 6.3121 GiB before XGB/overhead/copies. Provisional
later full-producer budget (10 h / 15 GiB new artifacts) is explicitly unapproved
and must be reassessed after lifecycle evidence. The review itself changed no
executable code. Implementation started with additive `engineering/lifecycle.py`,
`engineering/lifecycle_probe.py` and two matching test files; legacy pinned sources
are unchanged. Interrupted before focused verification completed, resumed on
2026-11-14. Implementation verified: 52 new lifecycle cases passed within the
388-test full suite (226.19 s); all nine required commands returned exit 0. Extra AST,
CLI-help, prior-source-binding and diff checks passed. Tests include separate
phase loaders, actual killed RF workers after fit commit/during scoring, model
reuse, corruption/locks, orphan-owner cleanup, no CPU fallback, all sampled stop
conditions, unchanged generated selections and no test-reader access. The tiny
selection fixture can produce NaN threshold-search metadata under ties; matching
undefined values were compared explicitly, without changing the existing seal's
NaN rejection or scientific policy. Recovery uses new attempt folders
in the same run (no copied models) and preserves the original wall-clock deadline,
including downtime; time or artifact budgets never reset.

The one authorized lifecycle attempt returned START_REFUSED: only 2,433,146,880
bytes / 2.2660 GiB available RAM versus the unchanged 6 GiB start gate. GPU query
worked (driver 610.88; 7.4111 GiB free). No worker, benchmark array load, fit,
scoring or external lifecycle run was created. No automatic retry or limit change.
Evidence: `.release-audit/c03/lifecycle-checks-20261114.json`, associated nine logs,
`lifecycle-attempt-console-20261114.txt`, and
`experiments/c03/engineering/LIFECYCLE_PROVENANCE.json` (SHA-256
`77deb86bf93350286332feb0838fc4e9fe8331c5c48a92e240f29b598d35d877`).
Next executable action: user may inspect resources with
`.venv/Scripts/python.exe -B -u -m experiments.c03.engineering.lifecycle_probe --inspect`
and run the same `--probe` command when the start gates pass. Real GPU
reload/scoring and full-row lifecycle measurements remain unverified. Verify
any returned result before considering the unapproved full-producer envelope.
Closure scan also returned exit 0 but is not a clean-release certificate: it
reported 206 local-path, 22 email, 36 private-network and four filename findings
outside this batch's seven affected files. Counts were unchanged between the
pre-attempt and closure scans; no affected file was flagged. Evidence:
`.release-audit/c03/lifecycle-scan-review-20261114.json`. Existing unrelated
findings were not modified. Final source/provenance and Markdown checks passed.
Review checks passed: `uv run python scripts/check_links.py` (tracked links),
`git diff --check`, explicit UTF-8/trailing-whitespace checks on both edited
untracked Markdown files, and metadata-only `producer.plan.build_plan()` to
confirm archived 300/800/1,400 estimator counts. Full tests were not rerun for
these documentation-only edits; the preceding 336-test result remains applicable.
Source revision remains
`ca13f1097f851ce6693658ccc0c6dd35480fabd6` plus uncommitted C03 additions.
Full preprocessing/model resources, production process isolation and benchmark
execution approval remain outstanding. Do not launch a larger sizing probe or
benchmark run without resolving those choices.

Source-path disjointness and numerical support do not establish capture/flow
independence or absence of proxies. One-cell evidence remains conditional on the
prepared corpus. No source-independent generalization or new scientific result
has been produced. Main package scientific parameters, policy, seeds, splits,
thresholds and stored result artifacts remain unchanged.

## Quota-window handoff — 2026-11-18 local

The agent reached the end of the current quota window while the detached full
producer was still preparing TRAIN data. No scientific model/score completion
is claimed. Verify PID identity and the v2 console logs recorded above, then
inspect the run's summaries/producer-complete.json. Let the bounded job finish;
do not launch a duplicate or edit its bound source files. Its deadline is
2026-11-18 07:05:46 UTC. After producer completion, independently verify every
checkpoint; implement/freeze the validation selector and separate post-seal
evaluator against the existing approved policy before any test-data reads.
A producer success alone does not authorize interpreting engineering scores
as scientific outcomes or expanding the full campaign.

## Verified producer completion — 2026-11-18

User requested a progress check. Read-only runner.verify_completed(contract.open_store(run))
passed against current source and all 11 checkpoints, exit 0. Run
pilot-producer-20261117t210546664156z completed in 15,977.139 seconds
(4 h 26 m 17 s), at 04:32:04 Asia/Amman. Fresh preparation plus all five
full fits and independent validation score phases completed. Peak worker RSS
3.6918 GiB, peak whole-device GPU used 0.9492 GiB, artifacts 6.4344 GiB.
No selection or test evaluation performed. Next: prospectively bind and verify
the scientific selection/seal and separate evaluator; then apply approved
blind/visible policies. No background producer remains running. This progress
check does not restart the previous quota-bounded autonomous session.


## Quota boundary and next preflight — 2026-11-18

Current renewed quota reached 100% (reset epoch 1794997909) after completing
and verifying the C03 post-seal pilot. Stop autonomous work here. No jobs remain
running. No additional partitions, classifier fits or method experiments started.
The user-facing results report is docs/RECKONING_C03_FIXED_PILOT_RESULTS.md.

Read-only next-task findings:
- Local processed_V5/B_day_file/CICIDS2017 is the older invalid WHOLE-FILE
  surface, with zero selected holdouts. Do not confuse it with the recovered
  contiguous surface in the protected Codes snapshot.
- Protected recovered metadata: Codes/ids_eval_framework/outputs/02_prepared_data/
  processed_V5_cicids17_recovery/B_day_file/CICIDS2017. Twenty contiguous units
  from five raw day files, four segments each; exhaustive chosen assignment
  12 train / 4 validation / 4 test, six eligible holdouts. Current packaged
  _search_protocol_b_exhaustive uses no RNG; changing seed does not regenerate
  distinct partitions. A new sampling/selection rule needs scientific decision.
- Original group-safe generator recovered READ-ONLY from Git commit
  bff66918cbf28d6a1d70e17bcfaeb51a7f11057b, path
  scripts/15_materialize_cicids2017_group_safe_surface.py. Its precursor is
  ee3db0ea6a8d6bdd9b7a7f1a62b443bf81e2f495,
  scripts/14_feasibility_cicids2017_group_safe_split.py. Neither restored nor run.
- Generator uses the Protocol-A processed union, pandas 64-bit feature hashes,
  seed-mixed 60/20/20 buckets, and deterministic whole-single-family-group
  support repairs to >=200 rows. Expected 2,099,879 rows / 1,735,180 groups;
  84 features, 14 family-conflict groups, historical 109 moved rows/groups.
  Historical summaries are in outputs/12_jisa_finalization/03_group_safe_split_feasibility
  and 04_group_safe_surface. This construction does not preserve raw day/file
  separation. Hash equality is not itself a new collision-audited identity proof.
- Local processed_V5/A_stratified/CICIDS2017 exists; metadata/part existence only
  checked, no current row-level regeneration or source-content verification yet.
- Next useful bounded action: recover exact generator bytes into ignored evidence,
  verify source/environment and existing recipe, then reproduce only its seed-123
  assignment/support fingerprints under a fixed resource/time envelope before
  proposing repeated-seed execution. No new partition distribution has been chosen.
- The other P2 mandatory methods pilot (EFC plus a second faithful literature
  implementation) and full P1 eligibility/audit completion remain outstanding.
  No campaign freeze, extra holdouts/seeds, inference policy or manuscript claims
  were approved by this pilot outcome.
