# VERA-IDS post-COMNET Reckoning: Proposed five-dataset, claim-matched revision campaign

**Prepared:** 2026-10-23  
**Status:** **PROPOSAL FOR USER REVIEW, NOT AN ADOPTED SCIENTIFIC CAMPAIGN OR FREEZE.** This replaces no prior adopted decision until expressly approved. Existing `config/`, model code, raw data, experiment outputs, and `release/v2026.10-comnet/` are unchanged. PR #3 stays DRAFT.  
**Primary editorial source:** Google Drive `09_SOURCE — Computer Networks Editorial Decision`, manuscript COMNET-D-26-07933, dated 2026-10-13. Editor described missing scientific significance/archival value; conclusions needed further substantiation, generalization under a wider range of conditions, and comparison with several similar literature approaches.  
**Existing handoff source:** `Thesis 2: Reckoning` Docs 00-06, especially `01_COMNET_REJECTION_AND_REVISION_RATIONALE`, `02_REVISION_EXPERIMENT_CAMPAIGN_SPEC`, `04_MANUSCRIPT_REWRITE_SPECIFICATION`, `06_DECISION_LOG_AND_OPEN_QUESTIONS`.

## Proposed central claim and change in framing

**Proposed claim:** An IDS result is interpretable only with respect to the evaluation condition and evidence needed for the particular claim; VERA-IDS produces a claim-specific protocol-eligibility record and tests whether the supported conclusion (ordinal ranking, categorical/feasibility assessment, magnitude effect) remains supported when a defensible evaluation condition changes.

This extends the original conclusion-stability methodology with **upstream dataset/protocol admissibility**, not a novel classifier, a zero-day detector, a blanket condemnation of imperfect datasets, a universal robustness score, or an unsupported causal ablation. The operative scientific question is NOT how many datasets report macro-F1, but under what combination of source provenance, labels, feature restrictions, split units, support and selection information each claim can be made.

## Editorial-to-experimental traceability

| Original editorial concern | Specific new evidence / reusable artifact | Failure condition |
|---|---|---|
| Conclusions under-substantiated | seed-paired CICIoT2023 visible/blind design, repeated admissible CICIDS2017 group partitions, explicit effect estimates, uncertainty and claim-level manifest | changed conditions not actually paired/group admissible |
| Wider range of conditions | symmetric validity/eligibility evidence across CICIDS2017, CICIoT2023, IoT-23, TON_IoT and Edge-IIoTset; additional fully matched cross-environment result **only where admissible** | five dataset case descriptions alone do not replicate an instability conclusion |
| Similar literature methods | >=2 genuinely distinct published IDS/open-set algorithms under common train/val/test, held-out family, and operating policy, not transferred hyperparameter profiles | cannot implement faithfully or same test was used for method selection |
| Archival value | reproducible claim-specific eligibility and exclusion reasons, partition/condition pairing manifests, stability records, end-to-end evidence/claim map | merely verbose benchmark diagnostics and no machine-readable artifact |

## Dataset roles and constraints

| Dataset | Existing evidence / notable limitations | Proposed role |
|---|---|---|
| CICIDS2017 | Main VERA primary; source/day provenance, recovered contiguous source-file segments and processed-identity group-safe comparison; source-era labeling defects, some related subfiles not independent executions; very large observed partition effect | **Core condition experiments**: Protocol A, support-audited Protocol B when valid, >=5 *admissible* partition seeds per construction family, plus symmetric source/label audit. No claim every unit equals independently executed attack |
| CICIoT2023 | Main primary with broad categories and per-family Protocol B support; original flow CSV `part-*.csv` chunks do not by themselves certify distinct independent executions; source/label rules and proxies need symmetric review | **Core condition experiments**: >=5 properly paired visible/blind trials with shared source partitions/model/test records; method comparison, conditional stability; audit scenario provenance and no label/feature shortcut |
| IoT-23 | Original 20 malware scenarios + three labeled benign sources, exact native C&C/DDoS/scanning recurrent presence; 3 full local malware census matches publisher; publisher public rule spreadsheet uses Zeek destination ports, IPs and `conn_state` to generate labels; mixed/compound identities | **Validity/label-rule case**; optional tightly scoped source-group model diagnostic only after separate approval of taxonomy and predictor-policy, never automatically third full primary LOAO. Original final primary NO-GO currently only a recommendation, not approved |
| TON_IoT | 22,339,021 processed rows audited, 9 attack-family time windows pairwise disjoint, attack/source identity and benign time support confounded; no authenticated repeated per-family executions | **Temporal/provenance stress case**; optional qualified Protocol A sensitivity in independent collection environment if a leakage-safe, support-eligible within-corpus protocol can be precommitted. Its primary Protocol B role remains unapproved |
| Edge-IIoTset | User-approved **PRIMARY Protocol B NO-GO** after audits: curated MQTT "0.0" string shortcut perfectly associated with attack row source in three columns, attack-run provenance insufficient and PCAP coverage/overlap anomalies | **Existing documented validity case ONLY**; conditional new model/representation sensitivity experiment requires a NEW explicit approval. Do not reopen its closed primary candidacy without authenticated new data |
| NSL-KDD / UNSW-NB15 | Old existing external closed-set stress results | Retain as secondary/SI if space; not LOAO-independent primary evidence; don't upgrade to seven full campaigns |

Every included dataset receives a **common metadata and scientific-claim audit**, not necessarily all Protocol A/B model experiments. Existing prior primary/main roles preserved during proposal review; Edge primary closure is NOT reopened.

## Common claim-specific eligibility instrument (deliverable V1)

One machine-readable record **per (dataset, representation, native taxonomy, claim, protocol, split construction)** that specifies:
- source dataset/version/hash, published collection/scenario provenance, capture/attack-execution IDs known or missing, data transformation and label rules;
- exact label/taxonomy and source case normalization, known-benign/unknown definitions; separately note labels inferred by endpoint/traffic rules;
- candidate predictors and known direct label-rule fields, plausible proxies, identity/time shortcuts, exact/near-duplicate audit performed/not performed; do not equate removing literal label columns with leakage-free;
- disjointness of *the actual claimed independence unit* (row/capture/session/attack execution), count of independent units and supportive source proof; never infer sessions from filename, hour gaps, or PCAP rotations;
- immutable support rules and observed val/test support, holdout eligibility, group assignments/hash and source handling; original Protocol B requires >=200 rows per known family in train/val/test, >=200 unknown in val/test, >=200 benign val/test, and >=2 surviving known families, if the relevant native protocol is used;
- selection information and operating constraint; which claims can/cannot be estimated, why, and whether only a diagnostic may be reported;
- explicit **pass / restricted / inadmissible / not assessed** *per claim*, with reason and possible re-audit evidence, not a blanket judgment that a whole dataset is "good"/"bad".

Important: Define separately the **minimum numerical rows** and **number/independence of underlying collection units**. Protocol B eligible row counts do not by themselves establish cross-execution generalization. Use the author's labeling-rule metadata as exposure evidence, not as proof that a particular fitted classifier exploited it.

## Experimental workstreams retained from original Reckoning

**W1 — Paired information condition (mandatory):** CICIoT2023, seeds 123–127. For each (holdout family, method, seed), use identical source partitions, training split, fitting/model seed, candidate models, transformation state, test records, and inference code; vary ONLY held-out family validation information available to supervised selection/tau tuning as defined prospectively. Report UDR, macro-F1, known/benign FUR, benign FPR, rejection, selected thresholds/rejector, and pairwise test differences with matched uncertainty. Be precise about what actually differs; no unearned "causal" claim. CICIDS2017 optional replicated lane if scientifically/operationally admissible.

**W2 — Repeated partitions (mandatory):** CICIDS2017 >=5 admissible partition seeds per candidate construction: (i) source/day contiguous recovered units (NOT new independent attack executions), (ii) exact processed-identity group-safe where regenerable. `partition_seed` separate from `model_seed`. Start fixed primary RF seed to isolate partition changes, optional 5x5 crossed model/partition seeds after resource pilot. Save group and raw-row provenance, support eligibility and split/feature identity hashes, repeated family-level UDR/macro-F1/FUR/rejection, dominant residual class and pairwise/rank changes. No claims of independent observations from overlapping realizations.

**W3 — Genuine published methods (mandatory):** audit EFC first, then shortlist OpenMax, DOC, ODIN, CROSR, OpenNet or other faithful methods using published source code/version/license/inputs/runtime; select >=2 methodologically distinct methods *before* full comparison results. Compare under identical eligible CICIDS2017/CICIoT2023 partitions, known/unknown semantics, validation information, operating constraint, and test records. Preserve the original external-profile RF/XGB/LGBM/etc configuration transfers as separate supportive SI evidence, NOT method comparisons. Reject an implementation that requires invented algorithmic steps.

**W4 — Stable and unstable conclusions (mandatory):** explicitly seek both. Revisit fixed RF residual destination stability across seeds and extend to XGB and selected genuinely distinct methods where outputs are semantically comparable; quantify dominant residual destination, share, destination entropy, preservation frequency, UDR/FUR/macro-F1 differences. Do not force residual routing for an algorithm without comparable represented-class destinations; don't assert one universally stable conclusion.

**W5 — Cross-dataset validity and diagnostic effects (NEW):** complete symmetric claim-specific I/O/label/group audit for all five. Prioritize one *controlled, predeclared* validity-factor comparison per additional dataset where possible (e.g. existing documented Edge curated shortcut exposure as observational finding; optional paired feature ablation ONLY if new permission; IoT-23 original label-rule field vs rule-blind variant ONLY if native task/feature policy explicitly approved; TON timestamp/source proxy audit and optional bounded paired Protocol A). Avoid reporting ineligible LOAO model scores as evidence. A documented violated assumption is not an empirically quantified effect until matched tests measure a change.

**W6 — Cross-environment replication gate (NEW):** To make a **positive wider-generalization claim** answering COMNET, demonstrate at least one claim-matched effect in a third independent dataset under an explicitly admissible protocol, or qualify the manuscript's scope and reconsider whether the archival-value target was met. Merely finding five separate validity threats is not a replication of a measured ranking reversal/visible-blind delta. Across different native taxonomies compare conclusions/conditions, NOT absolute accuracy as a leaderboard.

**W7 — Machine-readable scientific artifacts (mandatory):** per-claim records for ordinal/categorical/magnitude outcomes with actual values and deltas, preservation state where meaningful, CIs, limitation flags, dataset/method, `partition_seed`, `model_seed`, holdout, information condition, group ID, `pair_id`, split/config/model code hashes, citation/evidence URI and artifact status. Suggested new files: `outputs/17_revision_campaign/claim_evidence_manifest.json`, `dataset_protocol_eligibility.csv`, `conclusion_stability_records.csv`, `conclusion_stability_summary.csv`. Output "inadmissible"/"not assessed" explicitly rather than artificial zero performance.

## Protocol freeze and operational stage gates

**P0. Approve scientific scope:** user decides adoption of five-dataset claim-aware role matrix and whether Edge remains a read-only audit case. No existing adopted Edge closure overwritten. Decide no universally mandatory third full Protocol B environment vs a third independent replication requirement at claim-level; identify exact claims to attempt to replicate.

**P1. Reaudit BOTH incumbent primary datasets symmetrically:** source attack/capture unit provenance, label-generation mechanism, potential feature shortcuts, source identity and near-duplicate risks, actual existing Protocol A/B split manifests and conditions. Check which existing claims pass which independence level. Preserve valid previously released results while narrowing claims where necessary.

**P2. Limited feasibility pilots:** one paired visible/blind holdout/seed, one independently regenerable CICIDS2017 partition per construction, EFC + second candidate method one family smoke test, and only the cheapest extra dataset diagnostic needed to establish a third admissible replicated claim. Kill unsupported lanes early, not results-driven exclusions.

**P3. Pre-register/freeze:** implement and peer-review `config/revision_campaign.yml` plus `outputs/17_revision_campaign/CAMPAIGN_FROZEN.json`: datasets/roles, representation/version/hashes, target map, direct/proxy feature restrictions, split/partition algorithms, seeds [123..127], group definitions, unchanged 200-row support for eligible Protocol B, retained 1.5% Stage1 benign-FPR validation budget where applicable, model/method versions, candidate selection, rejection budgets, tests, uncertainty and missing/ineligible case handling. Note deviations before test results, never select a favorable partition after testing.

**P4. Full fixed experiments:** run only pre-registered eligible workstream cells; save a unique run/claim ID, dataset/partition/model/selection hashes, provenance, GPU/CPU environment and resource consumption. Comparison cells with different test units are explicitly not paired.

**P5. Audit and interpret:** condition effects with uncertainty appropriate to independent units (group/cluster-level if possible; row-wise conditional bootstrap clearly limited), predeclared multiple-comparison handling, inspect confounding, model method generality, event/family ranking and residual routing; regenerate machine-readable stability records and claim-evidence manifest; qualify infeasible/failed cases.

**P6. Manuscript restructure:** Only after results and campaign gate freeze. Recommended flow: Motivation and claim validity; related benchmark/open-set evaluation methods; claim-specific eligibility and conclusion-stability framework; datasets/label and source conditions; matched experimental protocols; findings (admissibility profile, paired information, partition distribution, method-level comparison, cross-environment replication, stable counterexamples); discussion and limitations. Avoid sentence-style headings, meta-language, unsupported novelty, universal zero-day, SOTA or causal claims. Figure and table selection prioritize the editor's three objections. Retain older closed-set and configuration-transfer results in SI as appropriate.

## Decision-relevant unresolved questions

- Can we prove or restrict CICIoT2023 source/capture group independence beyond `part-*.csv` chunks without invalidating central Protocol B interpretation? A within-published-corpus LOAO question is still valid when explicitly labeled as such.
- Can CICIDS2017 processed-identity group-safe partition be regenerated and repeated under a defensible algorithm, or must only one construction family be repeated and the other remain a fixed comparator?
- Are the 3 native IoT-23 behavior labels under author rule-based annotation appropriate for ANY supervised cross-condition experiment, or should IoT-23 remain a robust rule-circularity exposure case only?
- Can TON_IoT or IoT-23 support even one accepted, claim-matched **independent-environment replication** of an original VERA effect without laundering source identity/labels?
- Will two faithful literature methods fit the standard feature interface without method-changing choices?
- Is the resulting paper still plausibly suited to a higher-ambition journal if the third replication gate fails? Do not assume five datasets in a table answers the editorial generalization concern.

## Frozen invariants preserved

Original Computer Networks release `release/v2026.10-comnet/` never modified; branch `post-comnet-archival-value` and PR #3 DRAFT; no new dataset/model/taxonomy/split/feature/threshold protocol adopted by proposal; author data untouched; no unapproved Edge new runs; standalone TON and IoT-23 roles remain recommendations/potential, not promotions; all genuine study decisions remain user-mediated and logged in Google Drive `Thesis 2: Reckoning/06_DECISION_LOG_AND_OPEN_QUESTIONS`. This proposed plan has NO executable approval effect until explicitly adopted.
