# JISA Q1 Revision Plan

Status: OWNER-APPROVED, PHASE 0 ACTIVE
Date frozen: 2026-09-07
Target journal: Journal of Information Security and Applications (JISA)
Branch: `jisa-q1-revision`

## Purpose

This branch is a new, explicitly approved experiment/revision surface. It does not
rewrite or invalidate the historical public-release evidence on `main`. Existing
claim-bearing outputs remain immutable unless a new JISA result is generated under
this branch and its provenance is recorded separately.

The article contribution is evaluation methodology: determine how the security
conclusion changes as evidence moves from conventional closed-set component
performance to composed-system performance, supported family exclusion,
validation-visible versus blind rejection, cost-constrained selective rejection,
and residual failure destinations.

Competitive model performance is retained as a validity check: the observed
evaluation effects must not be attributable merely to obviously weak classifiers.

## Owner-approved decisions

1. **A1 — blind rejection analysis:** add a separate blind operating-point lane.
   The held-out family remains excluded from training, and held-out-family
   validation observations are excluded from rejector threshold/policy selection.
   The existing validation-visible Protocol B remains preserved and explicitly
   labelled as such.
2. **B1 — evidence hierarchy:** CICIoT2023 is the canonical unsupported-family
   surface. CICIDS2017 remains a primary benchmark but its Protocol-B role is
   partition-validity/sensitivity evidence because source/provenance independence
   and exact processed-identity isolation cannot both be satisfied on the recorded
   construction.
3. **C2 — reference profiles remain in the main paper:** rebuild the comparison so
   published source metrics, reconstructed/framework-compatible profiles, and
   VERA-IDS results are separate provenance categories. No profile is called an
   exact replication unless independently verified.
4. **D2 — controlled direct comparison:** add matched RF and XGB direct-multiclass
   comparators and an FPR-matched operating-point comparison, while retaining the
   broader validation-selected competitive direct baseline.
5. **Local evidence available:** the owner retains raw datasets, prepared datasets,
   full output directories, and historical local run artifacts. Local evidence may
   be audited/replayed but row-level data, predictions, models, and local paths must
   not be committed.

## Hard scientific boundaries

- `Unknown` in the existing Protocol B is a simulated held-out-family condition,
  not proof of live zero-day detection.
- Existing validation-visible and new blind rejection results must never be merged
  into one metric surface.
- Test labels may not select models, thresholds, rejectors, calibration parameters,
  operating budgets, or feasibility rules.
- New experiment outputs must live under a JISA-specific local namespace and may
  not overwrite historical outputs.
- Published-literature metrics are contextual unless task, taxonomy, split,
  sampling, averaging rule, and operating point are demonstrably matched.
- A direct-model FPR advantage/disadvantage is interpreted architecturally only
  under a matched operating-point construction.
- CICIDS2017 family-level Protocol-B results are partition-sensitive evidence, not
  a clean gold-standard estimate.

## Planned evidence sequence

### Phase 0 — evidence/provenance audit (active)

No new model training and no manuscript rewriting.

Required outputs:

- inventory of available raw/prepared/local result artifacts;
- hash/schema audit of tracked compact summaries;
- map of reusable validation/test score artifacts;
- determination of whether source/unit identifiers can support cluster-aware
  resampling;
- exact mapping of current manuscript claims to canonical result surfaces;
- list of claims requiring correction, rerun, or relabelling;
- frozen experiment specifications for Phases 1–5.

Stop if provenance cannot distinguish historical, corrected, seed-reliability,
reference-profile, and open-set result surfaces.

### Phase 1 — controlled direct comparison

Planned surfaces:

- RF two-stage vs RF direct multiclass;
- XGB two-stage vs XGB direct multiclass;
- ordinary direct multiclass argmax;
- direct multiclass under a validation-selected benign/attack gate matched to the
  two-stage Stage-1 operating constraint;
- existing broader competitive direct baseline as a separate selection surface;
- five-seed summary where the local evidence permits a complete rerun.

Exact model-search budget and threshold-search rules are not frozen until Phase 0
confirms the historical search spaces and reusable artifacts.

### Phase 2 — reference-profile provenance repair

For every cited profile, record separately:

- metric exactly reported by the source paper;
- source task/taxonomy/split/sampling/averaging definition;
- parameters reconstructed or inspired from the paper;
- VERA-IDS Protocol-A result;
- VERA-IDS Protocol-B result where valid;
- comparability boundary.

### Phase 3 — blind rejection

Primary dataset: CICIoT2023.

Conceptual rules already frozen:

- LOAO training is unchanged;
- held-out-family validation observations are unavailable to rejection-policy
  selection;
- only represented-family and benign validation traffic may define thresholds;
- primary rejection budget is 3%; sensitivity budgets are 1%, 5%, and 10%;
- the same known-traffic feasibility constraints used by the corresponding
  validation-visible analysis are preserved unless Phase 0 uncovers a historical
  inconsistency that must be resolved before execution;
- seeds 123–127 are targeted;
- test remains evaluation-only.

### Phase 4 — partition-validity analysis

CICIDS2017 is presented explicitly as a comparison between admissible but
incompatible dependence assumptions. Aggregate and family-level shifts are both
reported; no family shift is attributed causally to duplicate removal alone.

### Phase 5 — statistical refresh

Recover the strongest defensible sampling unit from local provenance metadata.
Where enough independent units exist, use cluster-aware resampling. Where they do
not, identify row-level intervals explicitly as conditional row-level uncertainty.
Formal paired method tests should use a predeclared paired randomization/permutation
procedure; bootstrap remains an effect-size/interval tool. Multiple-testing families
must be explicitly declared before test results are inspected.

## Manuscript structure after evidence freeze

1. Introduction: problem -> evidence gap -> research questions -> contributions ->
   design summary.
2. Related Work: benchmark validity; strong closed-set IDS; unsupported/open-set
   evaluation; calibration/selective rejection/evaluation integrity; explicit gap.
3. Methodology: design; datasets/taxonomy/preprocessing; partition/support audit;
   Protocol A; controlled direct comparison; validation-visible Protocol B; blind
   rejection; rejectors/calibration; metrics/operating constraints; statistics;
   reproducibility.
4. Results: closed-set evidence; controlled/literature comparison; held-out
   stability; blind-vs-visible rejection; separability/cost/rejectors; residual
   destinations; partition/external sensitivity.
5. Discussion: interpretation, deployment/evaluation implications, limitations and
   threats to validity.
6. Conclusion: concise findings, significance, and future directions.

## Phase gates

The manuscript Results/Discussion are not substantively rewritten until all new
claim-bearing experiments are frozen and every retained numeric claim has one
machine-readable provenance source.
