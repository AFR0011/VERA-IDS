# VERA-IDS Reckoning — Master reconciled campaign and decision gates

**Version:** reconciliation-draft-2026-10-23  
**Status:** DOCUMENTATION RECONCILIATION ONLY — **NOT an approved revised scientific protocol, experiment freeze, or dataset promotion**.  
**Canonical GitHub working copy:** `https://github.com/AFR0011/VERA-IDS/blob/post-comnet-archival-value/docs/RECKONING_MASTER_CAMPAIGN_PLAN.md`  
**Google Drive companion:** https://docs.google.com/document/d/1PnnSPFa9CMCI1iz7eYiTJmRjYaeMNqR6wit5ThBBpkw/edit in `Thesis 2: Reckoning`, document `10_MASTER_RECONCILED_CAMPAIGN_PLAN`.  
**Working branch:** `post-comnet-archival-value`, draft PR #3; **historical release:** `release/v2026.10-comnet/` must not be changed.  
**Scope of this reconciliation:** Preserve the complete original October 13 experimental commitments; incorporate established post-October-13 feasibility evidence and user-adopted decisions; separately enumerate October 23 *proposed* methodological extensions and unresolved scope conflicts. Nothing is silently adopted.

## 1. How to read this plan (authority and status)

- **ADOPTED:** Explicit user scientific decision recorded in `06_DECISION_LOG_AND_OPEN_QUESTIONS`, corroborated by a named GitHub decision memo when available. Later adopted decisions supersede earlier *conflicting* proposals, not the entire historical record.
- **ORIGINAL COMMITMENT:** Experimental objective from `02_REVISION_EXPERIMENT_CAMPAIGN_SPEC` (13 Oct), with repository implementation details in `03_REPO_IMPLEMENTATION_MAP`. Still the baseline unless explicitly amended; *conditional feasibility gates remain conditions*, not authorization to ignore failed evidence.
- **PROPOSED:** Additional or altered scope from `RECKONING_PROPOSED_FIVE_DATASET_CLAIM_MATCHED_PLAN.md` (23 Oct). Design material available for review, **not yet approved**.
- **EVIDENCE / OBSERVED:** Actual available artifact or source-audit finding. Evidence is not a decision to train, split, remap labels, exclude columns, promote datasets, or make an unverified inference.
- **OPEN:** A substantive choice requiring user approval and an entry in the decision log.
- **NOT ASSESSED / INADMISSIBLE / RESTRICTED:** Scientific claim-level assessment states; never encode failed eligibility as zero model performance.

**Precedence:** user-adopted dated decisions > immutable original evidence and existing protocol constraints > this reconciliation as a *navigation/execution map* > earlier proposal wording. Where scope conflicts remain open, **do not execute the disputed lane**. This document does not replace the original specification, the newer proposal, or the decision log. Once reviewed, it can become the single navigation entrypoint; only an explicit decision can make a new protocol operative.

## 2. Scientific objective and editorial problem

Computer Networks decision COMNET-D-26-07933 (13 Oct 2026): insufficient scientific significance/archival value; further substantiation, generalization to a wider range of conditions, and comparison with several similar literature approaches requested. VERA-IDS is an **evaluation methodology**, not a new IDS model, universal zero-day detector, SOTA claim, or universal scalar reliability score.

**Original objective retained:** Determine whether a claim-relevant conclusion is preserved or materially changes under admissible changes in operating constraints, held-out attack family, validation information, partition construction, model/method, and environment. Report **stable as well as unstable** conclusions. Use original Protocol A closed-set composed IDS, staged versus direct multiclass comparison under common FPR policy, support-audited Protocol B LOAO, validation-constrained rejection, residual routing, independent evidence/provenance and statistical summaries.

**New proposed methodological sharpening, not yet approved as a revised protocol:** Formal claim-specific eligibility and comparability gates, a versioned claim registry, and symmetric source/label/feature/provenance audits across five datasets. Validity threats or datasets listed in tables are **not** themselves replication of measured condition effects.

## 3. Original experiment workstreams — retained requirements

**O1 — Third primary environment (original, conditional).** Seek an independently collected modern environment supporting a defensible full VERA primary campaign, originally TON_IoT first. Gate on documented source/run provenance, taxonomy, leakage-safe feature policy, independently defensible splits, benign and known/unknown support, and runtime. If admissible, reproduce Protocol A RF/XGB and direct baselines, validation-aligned 1.5% benign-FPR architecture comparison, Protocol B LOAO, visible/blind selection, maximum-confidence/margin/entropy/family-conditional rejection, rejection-budget feasibility, residual routing, seeds 123–127 and possible alternative partition construction. **Current status:** search exhausted to final IoT-23 candidate; original requirement and replacement policy require user resolution (Section 8). It is not authorization to run invalid datasets.

**O2 — Paired validation-visible/blind (mandatory).** CICIoT2023, five seeds **123–127**, every eligible held-out family; TON_IoT only if Protocol B admissible; CICIDS2017 optional. For each holdout/method/seed, use **same train/validation/test partitions, model seed, fitted transformations, candidate model set, test record identities and inference semantics**. Change only held-out-family validation information *available to supervised model/threshold/rejector selection*, prospectively defined. Preserve the original caveat that preprocessing may already have seen held-out-family data; do not call this full data blindness. Save a shared `pair_id`, selected Stage-1 threshold, model/rejector/tau, matched test deltas: UDR, composed system macro-F1, known/attack FUR, benign FPR, overall rejection, applicable AUROC/AUPR. No causal interpretation beyond actual controlled information change. At six eligible CICIoT2023 holdouts, one declared fixed profile has a nominal **6 × 5 × 2 = 60** condition ledger entries, contingent on verified eligibility.

**O3 — Repeated CICIDS2017 partitions (mandatory).** **At least five distinct, admissible `partition_seed` realizations per *regenerable* construction family**, with `model_seed` separate. Examine (a) recovered contiguous raw source/day units, *not independent attack executions*, and (b) exact processed-identity group-safe partitions **only if deterministic regeneration and support checks succeed**. Exhaustive best-assignment selection in existing small-file code means changing an RNG seed alone does not produce five variants. Start with fixed RF model seed to isolate partition variability; pilot 5 × 5 crossed model/partition seeds only if justified. Save exact source/group assignment, split counts, row/feature/identity hashes, support-eligible holdouts, family UDR/macro-F1/FUR/rejection and residual destination. Compare per-partition magnitude and family ordering; repeated partitions share underlying observations and are not independent datasets.

**O4 — Genuine literature methods (mandatory).** At least **two methodologically distinct, faithfully implemented** published IDS/open-set algorithms, preferably three. Audit **EFC first**, then OpenMax, DOC, ODIN, CROSR, OpenNet or other reproducible candidates based on source paper/code commit/version/license, interface, training/validation/test separation, score semantics and runtime **before examining full comparison outcomes**. One-holdout smoke test first. Compare under common eligible CICIDS2017/CICIoT2023 splits, holdout/taxonomy, validation information, test population and operating policy; TON_IoT only if admitted. The existing Adewole/Hung/Keskin/Christy classifier-configuration transfers remain separate supporting evidence, **not** genuine method reproductions. Do not compare source-paper scores from incomparable datasets/splits.

**O5 — Stable and unstable conclusion evidence (mandatory).** Revisit fixed-RF CICIoT2023 stable dominant unrejected-unknown sink across six holdouts and five model seeds; extend to XGB and genuinely comparable methods/partitions. Record dominant represented class, share, destination entropy, preservation frequency, UDR/FUR/macro-F1 changes and undefined cases (no residuals). Do not force residual destinations on algorithms lacking comparable known-class routing.

**O6 — Machine-readable stability artifact (mandatory).** Create `outputs/17_revision_campaign/stability/conclusion_stability_records.csv`, `conclusion_stability_summary.csv`, `claim_evidence_manifest.json`. Each comparison records claim_id/type, dataset/representation/method, held-out family, partition/model seeds, condition and matched test identity, actual values/deltas, relation and preservation where relevant, uncertainty, evidence URI/path, code/config/split hashes, eligible/ineligible/missing/inconclusive state and interpretation boundary. **No universal scalar stability score**.

**O7 — Existing evidence verification and statistical synthesis (retained).** Preserve valid original Protocol A Stage-1 AUROC/FPR, Stage-2 attack macro-F1, final composed-system metrics; direct-versus-staged broad argmax versus **validation-aligned 1.5% benign-FPR** comparison; Protocol B 200-row support gates and LOAO semantics; maximum-confidence, margin, entropy and family-conditional rejectors with **primary validation 3% overall rejection budget** and 1%, 5%, 10% sensitivity; repeated RF/XGB and competence checks; external NSL-KDD/UNSW-NB15 closed-set stress and configuration transfers as SI where appropriate. Reuse released results when exact comparison conditions match; do not rerun by default. Statistical work: paired bootstrap of same fixed test pairs (original 1,000 replicates conditional on fixed model/data), Wilson rate intervals, dependence-aware/group resampling only if genuine independent groups exist, multiple comparison corrections (BH and Bonferroni sensitivity). Do not treat different test populations as paired.

**O8 — Reproducible release and manuscript (mandatory, last).** Full runs only after feasibility pilots, prospective protocol freeze, coverage and provenance audit. New release separate from `release/v2026.10-comnet/`. Rewrite with sharp technical style, no sentence headings, explicit RQ list, unsupported zero-day/SOTA/causal framing; prioritize claim-level evidence in main paper and bulky details in SI.

## 4. Proposed additions that can strengthen, not silently replace, the original work

**N1 — Formalization (PROPOSED).** For claim `q`, define evaluation condition `e`, metric vector `m(e)`, extraction `C_q(e)`, eligibility `A_q(e)`, comparability `M_q(e,e')`. Predeclare:
- **Ordinal:** signed difference between methods; preserve/reverse/inconclusive according to estimability and interval; point-estimate sign change alone does not prove population reversal. Optional scientific relevance/equivalence margin selected prospectively.
- **Categorical:** feasibility state or dominant residual destination and transitions, eligible denominators and no-residual/undefined handling. Losing protocol eligibility is not a model behavior change.
- **Magnitude:** `m(e')-m(e)`, effect size, uncertainty and practical significance; not automatically stable/unstable.
- Repeated admissible-comparison preservation frequency with explicit denominator and dependence warning; synthetic tests for ties, missing denominators, no residuals, reversal and ineligibility.

**N2 — Symmetric dataset validity instrument (PROPOSED).** One versioned machine-readable `dataset_protocol_eligibility.csv` record per dataset/representation/native taxonomy/claim/protocol/split construction. Review CICIDS2017, CICIoT2023, IoT-23, TON_IoT, Edge-IIoTset by the same categories: source/version/hash, provenance, collection/capture/attack-run unit proof, label-generation and normalization, direct/proxy label leakage, duplicate/near-duplicate checks, actual partition group disjointness, minimum row support **separate from independent-source count**, selected model information and outcome-specific pass/restricted/inadmissible/not-assessed reasons. Especially re-audit both incumbent primary datasets rather than grandfathering them. Common audit ≠ mandatory full model experiments on five datasets.

**N3 — Third-source claim-matched replication (PROPOSED ALTERNATIVE to O1).** Attempt at least one supported, measured conclusion-stability effect in a third independently collected dataset under an admissible comparable claim/protocol, **not necessarily third full Protocol B LOAO**, subject to user approval. If none is possible, narrow external-generalization claims and reconsider archival target/venue. Five validity case studies alone do not satisfy the original editorial generalization objection.

**N4 — Optional dataset sensitivity studies (NOT AUTHORIZED).** IoT-23 native label-rule vs rule-blind features or rule-only baseline would require approved behavior labels, predictor policy, source groups and new training authorization. TON_IoT bounded temporal/identity or Protocol A sensitivity requires independently approved feature/split policy. Edge curated shortcut ablation or any new ML run is not approved; its primary Protocol B candidacy remains closed. Existing read-only audit findings may be retained without turning them into experimental causal effects.

**Compatibility summary:** N1 strengthens O6; N2 adds eligibility discipline to O1/O2/O3/O4; N4 is optional, never prerequisite to original core; N3 is an **unapproved replacement** of O1's full-third-primary target, not a decision already made.

## 5. Dataset-role and scientific decision matrix (as of 23 Oct 2026)

| Dataset | Original role | Audited evidence/status | Future model authorization |
| --- | --- | --- | --- |
| CICIDS2017 | Incumbent primary Protocol A/B; repeated partition target | Existing outputs and recovered/group-safe comparison; partition-unit and label/shortcut re-audit proposed | Original core approved as a research objective; *new* group algorithm, seeds/partitions still require pilot and freeze |
| CICIoT2023 | Incumbent primary; five-seed paired visible/blind core | LOAO support for six holdouts in prior work; source `part-*.csv` chunks do not prove distinct attack executions | Original paired lane planned; exact pair semantics/preprocessing checks before new runs |
| TON_IoT | First-choice conditional third primary | Complete 22,339,021-row temporal/identity audit; family windows disjoint; independent per-family attack executions not authenticated; user has **not promoted** | NO new Protocol A/B/feature/split training approval |
| Edge-IIoTset | Optional third-candidate search | **USER-ADOPTED PRIMARY PROTOCOL B NO-GO** on 18 Oct; deterministic curated MQTT string label shortcut, attack-run lineage gaps, PCAP limitations | **No primary reopening or extra forensic/model runs** without new decision |
| IoT-23 | Final conditional third-primary candidate | Exact subset censuses match publisher; numeric support witness, but author rule-driven label/feature circularity, mixed labels, weak independent benign sources; final **NO-GO recommendation** | **User has not approved candidacy closure or further models** |
| NSL-KDD; UNSW-NB15 | Secondary external closed-set stress | Existing released evidence | Retain as SI; do not treat as primary LOAO or rerun by default |

**Do not silently apply the same *independent-execution* requirement to every type of claim:** source/capture independence, row-disjointness and within-corpus LOAO are distinct inferential targets. For each claim the split-unit evidence must match the inference actually asserted. **Do not quietly relax a full-primary criterion** either; that is the open scope decision.

## 6. Concrete protocol invariants, with status

1. **Adopted/current Protocol B numerical floors:** at least **200 benign validation and test**, **200 rows per surviving known family in train, validation, test**, **200 held-out unknown rows in validation and test**, and **at least two surviving known attack families**, with all remaining train families required to be valid; evaluate actual source independence separately. These are existing `config/protocol_b_loao.yml` constraints. Do not infer attack executions from filename, PCAP rotation or time gaps.
2. Current original Stage-1 benign-FPR validation constraint for relevant architecture comparisons: **1.5%**; this is a selection target, not guaranteed test FPR.
3. Existing open-set rejector primary validation budget: **3% overall rejection**; sensitivity at **1%, 5%, 10%**; validation feasibility and false-unknown constraints before test evaluation. Preserve original Protocol B **raw-probability semantics**; a calibrated B lane is separate new work.
4. Keep `partition_seed` distinct from `model_seed`; do not choose a favorable partition after seeing test results. Distinct comparison groups with different test units cannot use observation-paired tests.
5. Do not treat row support as proof of capture independence, labeling rules as truth of detected real-world attacks, or removal of literal `label` columns as proof of freedom from proxies.
6. Existing `release/v2026.10-comnet/` and its proof remain immutable; branch and draft PR stay separate; no datasets/raw models/prediction rows/private paths/credentials committed.

## 7. Unified execution sequence and gates (no automatic permission)

**G0 — Resolve scientific scope (OPEN NOW).** Record (a) final IoT-23 primary candidacy, (b) retain full-third-primary requirement or authorize its replacement by claim-matched third-environment replication attempt, (c) whether N1 and N2 are adopted as new mandatory methodology, (d) roles/permission for any optional IoT-23/TON/Edge ML diagnostics. Historical Edge closure remains adopted.

**G1 — Baseline and evidence inventory.** Confirm current revision branch + draft PR, clean local worktree, `uv run python -m pytest -p no:cacheprovider`, `uv run python scripts/16_verify_comnet_release.py`; compare existing dataset/protocol config, split identity and output hashes. Read only; no automatic reprocessing of historical results.

**G2 — Method semantics and primary feasibility.** Build proposed claim registry and eligibility/comparability tests *if approved*; otherwise preserve original claim type rules. Audit incumbent source/provenance and Protocol A/B eligibility if N2 adopted. Explicitly document any original conclusion whose source independence level is narrower than previously claimed.

**G3 — Limited pilots.** One actual matched visible/blind selection pair on CICIoT2023; one alternative admissible CICIDS2017 partition per construction (must be regenerable); EFC + at least one second-method one-holdout smoke; timing/memory and provenance. Optional third source only after specific separate permission. If a gate fails, report failure rather than rerouting design based on favorable test metrics.

**G4 — Pre-register and freeze full protocol.** Only after pilot gates: `config/revision_campaign.yml` + `outputs/17_revision_campaign/CAMPAIGN_FROZEN.json`, with dataset/representation/hash, selected claims and practical margins, labels, features, split algorithms, `partition_seed` and `model_seed` (seeds 123–127 in declared lanes), fixed 200-row support, FPR/rejection operating policy, methods, holdouts, comparability, multiplicity and uncertainty procedure, missing/ineligible handling, expected cells, expected artifacts. Document all deviations in Drive 06 before results. Code/config names are targets, **not current executable approval**.

**G5 — Full fixed runs.** Execute *only frozen, eligible cells*, with run/pair IDs, identical test hashes for paired contrasts, model/code/config/group hashes and measured resources. Store outputs only under `outputs/17_revision_campaign/`.

**G6 — Synthesis and audit.** Compare condition effects and both stable/unstable outcomes; eligibility and comparability first; matched/group-appropriate uncertainty; empirical across-partition distribution without independent-dataset claims; inspect unknown-family heterogeneity and residual destinations. Generate records/manifest; verify run coverage, audit tests and citations.

**G7 — New release and manuscript.** Prepare independent revision evidence snapshot, then rewrite scientific paper. Do not imply broad generalization if no admissible third-environment replication succeeded. Literature-method comparison must record source implementation and any deviations. The original manuscript and COMNET release are historical.

## 8. Pending user decisions (no default outcome)

**D1 — IoT-23 primary closeout:** accept final scientific NO-GO recommendation and stop the third-primary candidate search, OR seek authoritative new evidence under explicitly bounded review. Existing arithmetic group witness is insufficient admission proof.

**D2 — Third-environment requirement:** continue requiring **third FULL PRIMARY** admissible environment, OR formally amend to **attempt claim-matched independent third-environment effect** with transparent failure/scope limitation. **This is the principal conflict between original and new plans.** Neither outcome is already adopted.

**D3 — Formal method/audit upgrade:** decide whether formal claim registry/eligibility/comparability and common five-dataset source audit become mandatory core deliverables (N1/N2), while preserving all original O2–O8 experiment requirements.

**D4 — Third-source optional experiments:** approve or decline **specific** IoT-23 label-rule study, TON bounded experiment, Edge new ablation (Edge primary closure not reopened); new feature/taxonomy/group policy must be individually agreed.

**D5 — Feasibility choices downstream:** CICIDS2017 group-safe regeneration algorithm, candidate literature-method shortlist and faithful adapters, inference equivalence margins, source groups, runtime budget and any 5×5 crossed seeds, after implementation evidence and before freeze.

For each decision record: **date / exact question / prior status / selected option / reasoning / supporting evidence / resulting allowed code/experiment changes / effect on manuscripts / future gates** in `06_DECISION_LOG_AND_OPEN_QUESTIONS`. No retrospective rewriting of old decision entries.

## 9. Output contract and quality tests

After approval, versioned registry/eligibility tables plus original O6 files should include: claim ID and exact statement; categorical/ordinal/magnitude type; denominator and metric semantics; dataset/representation/taxonomy; baseline/changed condition; single changed factor and fixed anchors; eligibility reason; comparability state; method/source/version; holdout and seeds; source group assignments; `pair_id`; test/split/config/code hashes; measured values, signed differences, uncertainty, practical margin and inferred state where supported; path to immutable evidence. **Null/NA + explicit reason**, never artificially 0 for ineligible/not assessed.

Tests must cover deterministic partition regeneration, exact group and data nonoverlap, proper paired test identity and selection-only changes, support floors, tied/undefined claim relations, incomparable populations, resume/manifest integrity, expected cell counts, change-control and no mutation of COMNET release.

## 10. Repository implementation map (original, not yet implemented)

- Existing: `src/ids_eval_framework/_native/prepare_datasets.py`, `src/ids_eval_framework/src/two_stage_engine.py`, `config/datasets.yml`, `config/protocol_b_loao.yml`; original `scripts/01–16` runners and release verifier.
- Proposed new partition: `config/partition_stability.yml`, `src/ids_eval_framework/src/partition_stability.py`, `scripts/17_run_partition_stability.py` (avoid colliding with already committed forensic `17_*` script names).
- Proposed paired runner: `src/ids_eval_framework/src/validation_information_stability.py`, `scripts/17b_run_validation_information_stability.py`.
- Genuine methods: `config/literature_baselines.yml`, `src/ids_eval_framework/src/literature_baselines.py`, `scripts/18_run_literature_baselines.py`.
- Stability artifacts: `src/ids_eval_framework/src/stability_profile.py`, `scripts/19_build_stability_profiles.py`.
- Campaign freeze/audits: `config/revision_campaign.yml`, `scripts/20_audit_revision_campaign.py`, `scripts/21_verify_revision_release.py`. **These are original suggested paths, not a guarantee they already exist**.
- Do **not** repurpose the existing `external_profile_robustness.py` configuration-transfer lane as method-level comparison.

## 11. Source index and handoff discipline

**Original authoritative design (13 Oct):**  
- Drive `00_READ_ME_FIRST`, `01_COMNET_REJECTION_AND_REVISION_RATIONALE`, `02_REVISION_EXPERIMENT_CAMPAIGN_SPEC`, `03_REPO_IMPLEMENTATION_MAP`, `04_MANUSCRIPT_REWRITE_SPECIFICATION`, `05_NEXT_AGENT_STARTER_AND_EXECUTION_CHECKLIST`.  
- `06_DECISION_LOG_AND_OPEN_QUESTIONS` is the dated record of choices and feasibility outcomes.  
- `09_SOURCE — Computer Networks Editorial Decision` preserves the editor's actual language.

**Newer proposal:** https://github.com/AFR0011/VERA-IDS/blob/post-comnet-archival-value/docs/RECKONING_PROPOSED_FIVE_DATASET_CLAIM_MATCHED_PLAN.md (23 Oct). Preserve its full text as a historical *proposal*, not an accepted replacement.

**Specific evidence:** `docs/RECKONING_TONIOT_TARGETED_VALIDITY_RESULTS.md`, `docs/RECKONING_EDGEIIOT_PRIMARY_CANDIDACY_CLOSEOUT.md` (adopted), `docs/RECKONING_IOT23_FINAL_PRIMARY_ADMISSION_REVIEW.md` (recommended, decision open), and related source-audit manifests.  
**PR:** https://github.com/AFR0011/VERA-IDS/pull/3 (draft).

**Future-chat instruction:** Read this master plan first, then the latest *dated* decision-log entries and exact protocol configs, then the relevant full original and proposal sections. Do not treat the latest document's proposed language as user approval; do not skip step gates, guess native taxonomies or independent units, or run the next experiment until prerequisite scientific choices have been resolved.

## 12. Reconciliation change history

2026-10-23 — First reconciliation draft combining October 13 full campaign + October 23 claim-matched proposal + actual audited/approved decisions. **Documentation only**; conflicts D1–D5 explicitly open. No experimental result, dataset role, preprocessing rule, support floor, split design, source or model artifact changed. Google Drive and GitHub copies cross-link; original docs preserved for provenance.
