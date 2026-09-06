# JISA manuscript rewrite blueprint

Status: editorial architecture frozen after journal-final experiments.

## Provisional title

**How Evaluation Conditions Change Security Conclusions in Machine-Learning Intrusion Detection**

Alternative conservative title:

**Evaluating the Validity of Security Claims in Machine-Learning Intrusion Detection**

The first title is preferred because the paper is now an empirical study of conclusion sensitivity across evaluation conditions rather than a framework-introduction paper.

## Central proposition

Strong performance under one valid IDS evaluation condition does not determine behavior under another condition that measures a different security property.

## Primary research question

To what extent do conclusions drawn from strong closed-set IDS evaluation remain informative when the same system is evaluated end to end and under supported attack-family exclusion?

Secondary questions are treated in prose rather than as a long numbered list:

- whether held-out-family behavior is related to known-family recall;
- whether held-out failures are diffuse or concentrated into known-label destinations;
- whether selective rejection consistently recovers held-out-family recognition and at what known-traffic cost;
- whether those conclusions persist across retraining seeds and validation-to-test operating-point transfer.

## Editorial identity

The manuscript is an empirical security-evaluation study. The validity-aware framework is the experimental method that creates comparable evidence surfaces; it is not the paper's protagonist and is not presented as a new IDS architecture.

Random Forest and XGBoost are controlled evaluation vehicles. No state-of-the-art detector, zero-day detector, or universal open-set method is claimed.

## Section architecture

### 1. Introduction

Paragraph-level jobs:

1. Establish that benchmark IDS scores answer questions conditional on an evaluation surface.
2. Explain why stronger security interpretations can fail when system composition, family exclusion, or rejection are introduced.
3. Identify the methodological problem: evaluation conditions change what can be inferred, and those changes are rarely measured directly within one controlled study.
4. Position recent benchmark-validity and open-set IDS work, explicitly acknowledging prior validity-focused frameworks rather than claiming an empty field.
5. State the primary research question and the empirical design used to answer it.
6. Preview the main findings without turning the introduction into a metric dump.
7. State the contribution in one compact paragraph: a controlled empirical comparison of evidence surfaces, supported by audit, support checks, repeated seeds, failure-destination analysis, rejection-cost analysis, and reproducible artifacts.

Do not use a bullet-list contribution section unless JISA formatting requires it.

### 2. Related work

#### 2.1 Benchmark and split validity in IDS

Cover benchmark provenance, leakage/redundancy, realistic-traffic differences, cross-dataset limitations, split sensitivity, and recent validity-oriented IDS evaluation work.

The closest recent validity-focused work must be discussed explicitly. Do not claim that no prior validity framework exists. The distinction to preserve is that this study connects partition sensitivity to end-to-end system behavior, held-out-family recognition, residual destinations, rejection cost, and uncertainty within one controlled evidence chain.

#### 2.2 Held-out-family and open-set IDS evaluation

Separate the development of better unknown-attack detectors from the present question of what a held-out-family experiment supports. Explain validation exposure: the held-out family is absent from training but present as Unknown in validation and test, so this is validation-selected held-out-family evaluation, not blind zero-day detection.

#### 2.3 Selective rejection, uncertainty, and operating cost

Cover confidence, margin, entropy, selective classification, calibration, and conformal-inspired family-conditional thresholds. State that family-conditional thresholds here are empirical validation quantiles and do not carry formal conformal coverage guarantees.

#### 2.4 Research gap

Gap statement: the underexamined issue is how conclusions about the *same IDS study* change across admissible evaluation conditions, and whether those changes are larger than ordinary retraining or operating-point variation.

### 3. Methodology

#### 3.1 Study design and evidence surfaces

Introduce Figure 1. Define the three evidence conditions succinctly:

- supported-family closed-set evaluation;
- support-audited held-out-family evaluation;
- validation-selected rejection replay.

Explain that evidence surfaces answer different questions and are not ranked as progressively more "realistic" in a universal sense.

#### 3.2 Datasets, taxonomy, and preprocessing

Primary datasets: CICIDS2017 and CICIoT2023. NSL-KDD and UNSW-NB15 remain external stress analyses and should be shortened or moved to Supporting Information unless needed by reviewers.

State family mappings and exclusion of CICIDS residual Other only as needed for reproducibility. Avoid implementation variable names and file-level repo narration.

#### 3.3 Split validity, support, and CICIDS2017 sensitivity design

This subsection is critical.

Report:

- the original whole-file CICIDS2017 Protocol-B split lacked sufficient family support;
- contiguous within-day units restored support but exact postprocessed-feature duplication connected all provenance units;
- provenance separation and complete exact-representation isolation therefore could not both be satisfied;
- the exact-representation-grouped, support-aware partition is used as a *sensitivity analysis*, not as a uniquely correct split;
- exact feature equality does not prove the same physical flow, and near duplicates were not analyzed.

The primary scientific question of this sensitivity block is whether held-out-family conclusions survive when exact representation sharing is removed while preserving the processed population and support requirements.

#### 3.4 Controlled IDS and closed-set evaluation

Describe Stage 1 benign-vs-attack and Stage 2 attack-family classification. Explain composed system predictions and supported-label macro-F1. Keep RF/XGB parameter detail compact and move full grids to Supporting Information/repository.

Direct multiclass results are contextual closed-set evidence, not the central comparison.

#### 3.5 Held-out-family evaluation and validation-only model selection

Define six support-admissible holdouts per primary dataset.

For CICIoT2023 repeated-seed scientific selection, candidate choice is validation-only:

1. maximum validation Stage-2 macro-F1;
2. maximum validation Stage-1 AUROC;
3. deterministic lexical tie-break.

The native test-ranked summary is never used for journal claims.

For matched known-vs-held-out analysis, fix RF class-weight-balanced across families so the evaluation condition changes while the model profile does not.

#### 3.6 Failure destinations

Define UDR first, then residual non-Unknown destinations conditional on held-out observations that were not predicted Unknown.

Report dominant destination share and normalized destination entropy/concentration. These are behavioral diagnostics and do not imply semantic or causal similarity.

#### 3.7 Rejection trade-off analysis

Methods:

- maximum confidence;
- top-two margin;
- normalized entropy;
- family-conditional validation quantiles.

Use common validation-time rejection budgets 1%, 3%, 5%, 10%; 3% is primary because it matches the accepted Protocol-B validation constraint.

Selection is validation-only. The 3% budget is a development-time feasibility constraint, not a guaranteed test rejection rate.

#### 3.8 Uncertainty and repeated seeds

Use 1,000 nonparametric bootstrap resamples of fixed test prediction-label states for CIs and paired rejector differences. Apply BH and Bonferroni corrections across the primary paired family. State explicitly that these intervals are conditional on the selected model and observed test sample.

Repeated seeds 123-127 quantify retraining variability separately.

### 4. Results

Results should follow the scientific story rather than the execution order.

#### 4.1 Closed-set performance establishes a strong supported-family baseline

Purpose: show that the evaluation systems are not trivially weak. Keep this subsection short.

Use corrected supported-label final-system macro-F1 and direct-multiclass context. Do not make model-ranking claims the center of the paper.

#### 4.2 Held-out-family conclusions depend strongly on the CICIDS evaluation surface

Claim C1. Figure 3.

Report the key paradox:

- mean UDR changes only modestly between recovered and grouped surfaces;
- mean absolute family-level UDR shift is approximately 0.444;
- maximum family-level shift approximately 0.890;
- family rank order changes substantially;
- RF weighting-lane perturbation on the grouped surface is orders of magnitude smaller than the surface shift.

Do not state that duplicate removal caused the observed changes.

#### 4.3 Strong known-family recognition does not imply strong held-out-family recognition

Claims C2 and C3. Figure 2.

CICIoT2023 is the clearest demonstration: several families with approximately perfect known recall have very low held-out UDR, while BruteForce has lower known recall yet higher held-out UDR under the fixed RF profile.

Report the descriptive family-level Spearman values only as descriptive summaries because n=6 per dataset.

Use the five-seed CICIoT results to show that candidate identity and macro-F1 are stable despite weak held-out recognition for several families.

#### 4.4 Held-out failures are concentrated into stable known-label destinations

Claim C4. Figure 4.

Report dominant destination share conditional on non-Unknown failure. Highlight a small number of representative cases, not every cell in prose.

For CICIoT2023, emphasize that the modal destination is stable across all five seeds for every holdout.

#### 4.5 Rejection recovery is family-dependent and cost-sensitive

Claims C5-C7. Figure 5.

Primary 3% validation budget:

- max-confidence feasible for 12/12 cases;
- margin 11/12;
- entropy 10/12;
- family-conditional 7/12;
- family-conditional improves UDR in 5 of 7 feasible cases but worsens 2;
- margin and entropy more often reduce UDR relative to max-confidence;
- positive and negative paired differences largely remain significant after BH/Bonferroni correction.

Treat CICIDS Web/App family-conditional as an explicit generalization failure example: the operating point satisfies validation constraints but test rejection rises sharply. Do not remove or reselect this point after seeing test outcomes.

### 5. Discussion

#### 5.1 What closed-set evidence actually supports

Closed-set performance establishes learnability and discrimination among represented labels. It does not automatically support conclusions about family exclusion, rejection, or end-to-end system reliability.

#### 5.2 Evaluation surface as a source of conclusion instability

Use CICIDS sensitivity result to argue that family-specific security conclusions can be much more sensitive to partition policy than an aggregate mean suggests.

Distinguish partition sensitivity from retraining instability. The grouped surface result must remain explicitly non-causal.

#### 5.3 Why known-class ease and unknown-family recognition diverge

Discuss the empirical mismatch without speculating about semantic similarity unless directly supported. An attack family can be highly separable when represented in training but lie confidently inside another trained region when absent.

#### 5.4 Structured failure matters operationally

A low UDR alone does not reveal whether failures are diffuse or repeatedly mapped to one known label. Destination concentration therefore adds security-relevant diagnostic information, but does not by itself identify causal feature overlap.

#### 5.5 Rejection is an operating policy, not a universal fix

Discuss feasibility, UDR recovery, false-Unknown cost, test coverage, and out-of-sample constraint stability jointly.

The Web/App case is useful because it shows why validation-feasible rejection cost must itself be evaluated out of sample.

#### 5.6 Threats to validity

Must include:

- validation exposure to the held-out family;
- CICIDS grouped partition sacrifices source-day independence;
- exact postprocessed-feature equality is not identity of physical flows;
- near duplicates not analyzed;
- six coarse holdout families per primary dataset;
- primary known-vs-held-out comparison fixes one RF profile;
- bootstrap inference conditional on fixed predictions;
- five seeds quantify only retraining variability on one partition;
- benchmark results do not establish deployment performance or true zero-day detection;
- dataset taxonomy affects what counts as an unknown family.

### 6. Conclusion

One compact conclusion. State that the principal result is empirical: security conclusions about held-out-family behavior, failure destinations, and rejection cost can change substantially even when supported-family performance is strong or aggregate metrics appear stable.

Do not end by reciting every framework component.

## Figure map

- Figure 1: evaluation design and evidence conditions.
- Figure 2: known-family recall versus held-out-family UDR under fixed RF profile.
- Figure 3: CICIDS recovered versus exact-representation-grouped UDR sensitivity.
- Figure 4: residual destination matrix.
- Figure 5: rejector UDR change versus test rejection-cost change relative to max-confidence.

## Main-table map

Keep the main article to approximately four or five dense tables. Move parameter grids, full support tables, complete seed tables, external stress details, reference-profile replay, and full bootstrap outputs to Supporting Information/repository.

Suggested main tables:

1. Dataset/evaluation-condition summary and validity boundaries.
2. Closed-set component/system context.
3. Held-out-family primary results with support denominators and repeated-seed summary where applicable.
4. Failure destination/concentration summary.
5. Rejection feasibility and validation-to-test cost summary.

## Mandatory removals from the old manuscript

The following old claims/numbers are prohibited from journal-final primary reporting:

- CICIDS historical held-out UDR range 0.001-0.991 as the primary Protocol-B result;
- CICIoT historical test-ranked UDR range 0.007-0.283 as the primary Protocol-B result;
- any statement implying the recovered CICIDS contiguous-within-day split is independent after the exact-feature audit;
- any statement treating the native `best_per_holdout` output as validation-only model selection;
- any claim that family-conditional rejection is formally conformal;
- any claim that a validation rejection budget is guaranteed on test;
- any zero-day or state-of-the-art claim;
- any claim that exact duplicate representations are proven duplicate physical flows.

The old manuscript abstract and result paragraphs containing these values must be rewritten rather than patched locally.

## Abstract strategy

Write the abstract last. It should contain:

1. problem and research question;
2. controlled evaluation design;
3. two or three strongest results only;
4. main interpretation;
5. no framework inventory and no reference-profile digression.

The strongest abstract candidates are:

- CICIDS mean UDR nearly stable while mean absolute family-level surface shift is ~0.444;
- CICIoT strong known recall can coexist with near-zero held-out UDR and stable five-seed model selection;
- held-out failures concentrate into stable known destinations;
- rejection gains are heterogeneous and validation-feasible operating costs can fail to generalize to test.

## Supporting-information policy

Move the following out of the main narrative unless a reviewer specifically requires them:

- exhaustive candidate grids and implementation parameters;
- full external stress tables;
- reference-profile replay details;
- every bootstrap contrast;
- full seed-by-seed operating points;
- calibration-detail tables not directly used by the main claims;
- repository execution maps.

These remain available as reproducibility evidence without competing with the article's scientific story.
