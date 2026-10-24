# Reckoning P1: COMNET claim-to-evidence audit

**Status:** Historical evidence-index candidate, **not** a frozen experimental protocol and **not** a declaration that data/source validity gates have passed. Prepared 23 October 2026 from the submitted Computer Networks manuscript, submitted Supporting Information, and release files at Git commit `4efe48cb1d990000040978d685dae33051b3df43`.

## Companion records

- `docs/RECKONING_COMNET_CLAIM_EVIDENCE_MATRIX.csv`: **45 claim-evidence records**, including three supporting observations not classified as stability families. Each has exact paper location, metric denominator semantics, observed values or availability status, source keys, uncertainty scope, limitation and next action.
- `docs/RECKONING_COMNET_EVIDENCE_SOURCE_MANIFEST.json`: exact source paths and verified **Git blob hashes** for 20 existing source files. Git blob SHA is **not** the file SHA-256 and should not be misrepresented as one.
- Submitted manuscript: https://docs.google.com/document/d/1VEUmVIpl4jpFkfrW7kKTYLprZS3dJRZTp7QJAJmTRp0/edit
- Submitted Supporting Information: https://docs.google.com/document/d/1fGslH_tcMogpvAw2IJ9YN8M7hzQBQZbiAIiCq0X2kvk/edit
- Primary provenance: `release/v2026.10-comnet/PROVENANCE.md`.

The CSV `source_keys` are foreign keys into `evidence_paths` of the JSON manifest. The manuscript and SI were read separately; they are **not** Git blobs in this manifest. Some main experimental CSVs are stored in the earlier **immutable** `release/v2026.10/raw/` path and are explicitly referenced by the COMNET provenance document; using them as retained historical evidence is not fabricating a new COMNET artifact.

## Reuse decisions, by scientific claim

| Family | COMNET paper | Existing scientifically relevant evidence | Necessary addition | Initial disposition |
|---|---|---|---|---|
| **C01** Direct/staged ordering | §4.2, Figure 2; SI S2 | Five-seed RF/XGB direct vs staged final-system macro-F1, ordinary argmax and validation-FPR-aligned policies. CICIoT RF observed reversal: direct-minus-staged approximately **+0.021 to −0.013**. | Match original per-seed comparison anchors/test identity and assess uncertainty; no automatic rerun. | **Descriptive reuse** |
| **C02** Rejector ordering | §4.3 Figure 3; §4.4 Table 4; SI S3–S4 | Five-seed CICIoT blind per-family method differences and separately visible joint-feasibility comparisons. Family-conditional vs max-confidence UDR delta flips sign across holdouts. | Define atomic pairwise ordinal comparisons and feasibility-conditioned denominator; budget-dependent *performance rankings* not established solely by budget feasibility. | **Descriptive reuse; limited extension** |
| **C03** Selection information | §§3.4,4.3–4.4,5.4 | Visible and blind results both reported, with notably different selected operating points; matched trajectories **not** demonstrated. | Prospectively paired visible/blind family × seed study, actual pair IDs, selection traces, matching test identities; preserve prior preprocessing exposure caveat. | **New paired runs required** |
| **C04** Partition sensitivity | §4.6 Figure 6 | Two CICIDS2017 constructions; DoS UDR **0.991127 to 0.100761**; six family-level changes. | Five genuinely distinct, admissible partitions per *feasible* construction; isolate partition seed from model seed and avoid test-row pairing across changed populations. | **Descriptive reuse plus new partitions** |
| **C05** Residual destinations | §4.5 Figure 5 | Fixed RF CICIoT: same dominant destination across five seeds for each of six holdouts; destination concentration and entropy recorded. | Define zero-residual = undefined; XGB / published methods only where destination semantics truly match. | **Descriptive categorical reuse** |
| **C06** Validation feasibility | §4.4 Figure 4 | Per-dataset, per-method feasibility at 1%, 3%, 5%, 10% budgets (32 aggregate method-budget cells). | Derive categorical transitions using explicit validation criteria; no model retraining needed to formalize these historical cells. | **Descriptive reuse** |
| **C07** Published method comparisons | §§3.8,4.7–4.8 Tables 5–6; SI S7–S12 | Competence check and published classifier-configuration transfers including six-family heterogeneity over five seeds. | At least two **faithfully implemented, methodologically distinct** published IDS/open-set algorithms, same valid evaluation conditions. | **Supporting evidence only; new method experiments required** |
| **C08** Independent environment | §§3.7,4.1,5.4 | NSL-KDD/UNSW-NB15 Protocol A external component/system results. They do **not** replicate C01/C03/C04/C05 directly. | Decide claim-matched estimand; audit independent source, taxonomy, leakage/proxy, grouping and label support; limited third-environment replication only if valid and separately approved. | **Validity/claim audit first** |

The release also supports three essential **supporting** observations: (S01) component scores vs composed final system, (S02) threshold-free AUROC/AUPR vs selected UDR as different measurement surfaces, and (S03) support/partition provenance. These are **not** fabricated numeric stability comparisons between incompatible metrics.

## Main analysis decisions

**Claim and inferential status.** The matrix describes what *was measured and published*, not which hypotheses are statistically supported under the new claim-aware protocol. A sign reversal in mean performance is an *observed reversal*, never a claim of statistically established ranking reversal until relevant uncertainty is justified. The inherited row bootstrap reflects fixed-model/test-row conditional uncertainty, not retraining or independent-source sampling. Differences between held-out-family test populations and partition test populations are not test-observation pairs.

**Denominators.** Protocol A final-system macro-F1 averages F1 over the supported ground-truth labels and does not artificially count an unsupported `Unknown` as a true class. Protocol B has genuine true-Unknown support. UDR divides successful Unknown decisions by **true Unknown** count; false Unknown rates on all represented traffic and on represented attacks use different denominators; rejection divides predicted Unknown by total evaluated rows; dominant sink share conditions on **unrejected true-Unknown residuals**, not all Unknown observations. The full definitions remain pinned to the original code and manuscript before any experiment freeze.

**Selection budgets.** The 1.5% benign-FPR target and the 3% overall rejection budget are **validation selection** conditions, not guaranteed realized test bounds. C02's visible common-case comparison depends on **joint validation feasibility**, not arbitrary exclusion of weak methods post-test. C06 categorical infeasibility is not a dataset eligibility failure.

**Scientific independence.** Repeated model seeds on one prepared split are not independent traffic sources. Different test rows preclude row-paired inference across partitions/holdouts. The historically released visible/blind records are not prospectively seed-matched; never silently match them using summary means.

## Important provenance anomaly (open)

`release/v2026.10-comnet/PROVENANCE.md` cites `outputs/summaries/data_audit_summary.csv` for Table 2. An exact GitHub fetch at the inspected source commit returned **NOT_FOUND** for that path. The matrix therefore records an **unresolved provenance reference**, rather than claiming the audit was missing historically or inventing an alternative source. Investigate local prepared-data manifests and historical output mappings; any correction to the released provenance must respect the immutable release boundary (publish a separate additive clarification, not silently modify the original snapshot).

## Precise source and scope caveats

1. The `source_files_verified` status means **the file exists at that Git path and hash**; it does **not** mean the underlying dataset has passed a new source/label/proxy/partition audit.
2. Reported example numbers in the CSV are tied to exact dataset, method, validation condition and manuscript section. Several are **rounded** transcriptions and must not be mixed with full-precision source values for inference.
3. The `NEW_*` and `CANDIDATE_*` statuses are genuine evidence gaps. The current eight C01–C08 families are provisional; they are **not** yet frozen atomic claims.
4. The current `config/claim_registry.json` has blank metric/denominator/anchor slots; the matrix is a source-of-truth *input* to defining them, not a replacement registry or an experiment runner.
5. No changes are made to `release/v2026.10-comnet/`, `release/v2026.10/`, the thesis, experiments, models, labels, or active configuration.

## P1 evidence freeze prerequisites, in order

1. Resolve the missing Table 2 provenance reference or explicitly document why the remote compact aggregate is unavailable.
2. For each `DESCRIPTIVE_REUSE` record, verify exact original split/holdout, model seed, method, validation information, test identity and eligible support from existing manifests. Mark source independence claims separately from row support.
3. Define atomic metric IDs, target label sets and denominators, true fixed anchors, allowed changed factor, test-population policy, appropriate inferential unit and multiplicity family. Avoid selecting practical-effect widths by inspecting outcomes.
4. Complete the symmetric five-dataset source/label/proxy/group/overlap/support matrix under the adopted user-approved audit scope. New IoT-23, TON_IoT, or Edge-IIoTset model runs remain unauthorized without a separate recorded decision.
5. Use only resulting eligible matched claim comparisons to decide P2 pilot needs; defer the full freeze to P3.

**Source-control policy:** This is an *additive historical audit* for the draft revision branch. The original COMNET release remains untouched. The `DESCRIPTIVE_REUSE` marker does not grant permission to relabel results as newly replicated or statistically significant.
