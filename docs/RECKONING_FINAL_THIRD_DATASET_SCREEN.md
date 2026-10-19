# VERA-IDS: Final third-primary environment search — source-screen outcome

**Screen date:** 2026-10-19  
**Decision state:** FINAL PAPER-LEVEL SCREEN COMPLETE. **No new dataset admitted, acquired, trained or split.** One candidate nominated for a strictly bounded feasibility attempt, conditional on user's separate approval and source permission. If it fails, **STOP searching**, return to the already planned paired visible/blind, repeated-partition, genuine-method and conclusion-stability campaigns on CICIDS2017/CICIoT2023.

## Motivation and strict criteria

The Computer Networks desk decision demands stronger independent generalization and genuine comparable methods, not a decorative third dataset. Previous full source investigations:
- **TON_IoT Network:** user-local 23-file/22,339,021-flow census and sampled 13,824 GT flow-key agreement. Nine attack family windows do not overlap. Original processed flow chunks lack independently authenticated repeated attack execution/capture IDs, and there are strong source/time confounders. TON remains **NOT APPROVED** for primary Protocol B and potential alternative roles undecided.
- **Edge-IIoTset:** user adopted **NO-GO** for third primary Protocol B on 2026-10-18 after C3–C10. Do NOT reopen. Incomplete original attack PCAP, repeated benign header metadata, curated binary shortcut, and mostly no recoverable repeated attack operations.
- **Primary datasets still CICIDS2017 and CICIoT2023**; NSL-KDD/UNSW-NB15 are only historic closed-set stress tests.

Required admission gates for final candidate, in order:
1. Authenticated **individual capture/execution units**, not attack-family filenames, arbitrary timestamps or shuffled rows; distinct independent units must support the **same** multiple known attack categories in train, validation and test partitions and honest LOAO of unknown behavior/family. Apply *existing* support thresholds rather than modifying them to admit candidate.
2. Original benign/malicious ground truth adequate for Stage 1; native, scientifically interpretable attack taxonomy adequate for Stage 2. Avoid pretending malware identity and malicious behavior label are identical classes.
3. Exact capture-to-labeled-flow provenance; no cross-split exact identity overlap, malicious event overlap or pseudo-independent duplicates; no use of literal labels, timestamp/capture IDs, source/host/network identities or rule keys as shortcut predictors without separately approved frozen feature policy.
4. Fit a bounded, reproducible pilot with acceptable file sizes/run time, deterministic group manifests and no test-based selection.
5. Clarify permission/licensing ambiguity **before acquisition/use**. No bulk download or training at screening stage.

## Candidate ranking, authors' sources versus our inference

### 1. IoT-23 original version 1.0.0 — ONLY FINAL FEASIBILITY CANDIDATE

**Author description:** Stratosphere Laboratory / Avast-AIC, twenty separately identified malware **execution/capture scenarios** and three benign IoT device capture scenarios, with original PCAPs, each scenario README and Zeek `conn.log.labeled`. Data from 2018–2019; original release January 2020, DOI **10.5281/zenodo.4743746**. Labeling based on analyst review and capture-specific flow classification rules. The official source page provides per-scenario distributions and **individual files** without downloading the 8.7 GB compressed complete flow collection or ~20 GB full PCAP collection.

Official source: https://www.stratosphereips.org/datasets-iot23  
Individual files index: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/  
Author sample single-capture directory: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/  
Source exemplar labeled file index: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/bro/ (2.8MB); 44-1 labeled flow file index ~30KB; 20-1 ~410KB; 3-1 ~23MB. **None downloaded.**

**Our reproducible manual enumeration from the author's *native, exact*, per-scenario label tables:**
- Exact label `DDoS`: observed in **7** distinct malicious scenarios: 34, 43, 44, 60, 17, 35, 7.
- Exact label `PartOfAHorizontalPortScan`: observed in **11** scenarios: 34, 43, 49, 52, 17, 33, 48, 39, 9, 3, 1.
- Exact label `C&C`: observed with nonzero support in **10** scenarios: 34, 43, 44, 49, 52, 8, 35, 39, 3, 1.
- These counts are SOURCE FILE PRESENCE, **not** number of independent attack executions per label, admissible split assignment, within-scenario sub-episodes, or sufficient train/val/test support. Other labels mix behavior, malware family and composites (`Okiru`, `C&C-HeartBeat`, `C&C-Torii`, etc.). Do **not** invent a family mapping from these strings; candidate's semantic admissibility hinges on a user-approved native-label taxonomy and adequate capture-level support.
- Benign labels occur in infected capture logs and three separate benign-device captures. Whether benign among infected captures meets the protocol's false-positive measurement/support/independence requirements needs verification.

**Scientific opportunity:** genuinely named capture scenarios with several *behavior-level* classes observed in multiple distinct scenario collections, something TON/Edge lacked under their attack-family schemes. Potentially support-auditable LOAO across capture-independent splits **if** capture IDs correspond to independent malware runs, repeated behavior classes have enough observations and risk of capture-specific labeling shortcuts can be controlled. This is a *plausible route*, not an admitted dataset.

**Blockers/gates still open:**
- Attack-behavior labels are heterogeneous with malware-family identifiers; only some behavior labels occur with useful volume per scenario, and the original author labeling rules can themselves use endpoint context. Need audit native class semantics, capture-level labels and leakage before approval.
- Different malware samples in the same malware family may confound model/capture domain and rare labels; no source-group assignments or model fits approved.
- Version 2 website explicitly describes v2 as **work in progress** and same raw captures with improved labels and expanded benign captures. No mixing v1/v2 or v2 substitution without user choice and version certification: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset-v2/
- A v1 individual scenario README explicitly says **“You need authorization from the Stratosphere Lab to use these files.”** This conflicts with generic public download visibility; obtain rights clarification from dataset stewards rather than assuming unrestricted permission. Exact example: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/
- **Do not** download malware binary files bundled near PCAPs. The only possible authorized input would be selected `README.md` and `bro/conn.log.labeled` files, not payloads or full tarballs.

**Recommended SINGLE limited pilot after permission + user approval:** inspect official full capture×exact-label table and a bounded handful of small, native author labeled-flow/README sources (e.g. capture 34-1 ~2.8MB, 44-1 ~30KB, 20-1 ~410KB and 3-1 ~23MB, if these provide complementary structural evidence). Verify label preservation, Zeek field definitions, source-capture identifiers, per-label denominators, explicit benign support, any missing columns and metadata leakage red flags. No feature blacklists, train/test groups or classifiers. **Before any actual grouping**, construct a *proposed* whole-source capture×native-class support feasibility matrix; demand independent source units and sufficient support under frozen support checks. If any gate fails, close third-primary search **without trying another candidate**.

### 2. UGR'16 (University of Granada) — provenance notable, primary fit not established

Original site: https://nesg.ugr.es/nesg-ugr16/  
Original test weeks e.g. https://nesg.ugr.es/nesg-ugr16/july_week5.php and https://nesg.ugr.es/nesg-ugr16/august_week1.php

Real Spanish ISP NetFlow v9 with artificial attack injections, original author weekly labeled CSVs and explicit `attack_ts_*.csv` timestamp annotation files; attack batch schedules documented over repeated July/August days. This is unusually good **event-timing provenance** and can isolate some injected attack episodes. Distinct source environment and collection network.

But the source's `background` flow label does **not** guarantee confirmed benign: live ISP traffic may include unrecognized anomalous/malicious flows. It is not equivalent to clean Stage-1 benign validation/test support for bounded FPR. Original preprocessed multiweek flow archive also enormous (third-party catalogue estimates ~236GB compressed) and collected in **2016** (older than intended modern IoT comparison). Paper-level screening therefore **does not qualify it as a third primary** under current benign-support and practical gates. Published `attack_ts` entries alone do not authenticate full positive class/benign labels.

Details: https://fkie-cad.github.io/COMIDDS/content/datasets/ugr16/ ; original paper DOI **10.1016/j.cose.2017.11.004**.

### 3. CICIoMT2024 (CIC/UNB) — contemporary, no verified third split group

Original publisher: https://www.unb.ca/cic/datasets/iomt-dataset-2024.html  
2024 IoMT network: 18 executed attacks, five high-level categories (DDoS, DoS, Recon, MQTT, Spoofing) across 40 real/simulated healthcare devices and Wi-Fi/MQTT/Bluetooth. Original PCAPs and Wi-Fi/MQTT CSVs organized in `train` and `test` folders. This is a promising *modern* environment, but publisher page alone does not establish a third independently executed **validation group** for multiple known attack categories, event IDs, train/test split *independence*, or source-deduplicated families. It is not sufficient to assume train/test folders imply independent executions; files may represent attack-category packaging or segments of the same run. Distinct Bluetooth/packet and Wi-Fi/MQTT feature modalities must not be conflated. Hence **paper-only hold**, not one final pilot.

### 4. CIC-DDoS2019 and CTU-13 — not selected

- CIC-DDoS2019 official: https://www.unb.ca/cic/datasets/ddos-2019.html. Two documented attack days with 12-vs-7 types and limited overlap, with full timing tables. Narrowly DDoS-centered; two days alone do not establish independent train/validation/test repetitions per family without extra evidence. No primary pilot.
- CTU-13 official: https://www.stratosphereips.org/datasets-ctu13 ; catalogue https://fkie-cad.github.io/COMIDDS/content/datasets/ctu_13/. Thirteen distinct 2011 botnet scenarios; some malware types repeated but others single-capture, mixed benign/normal/background taxonomy and age/problem scope limit broad Protocol B known-family source support. No primary pilot.

## Research stopping rule

**Final attempt means final attempt.** IoT-23 v1 is the ONLY candidate carried from paper screening to a *possible* small source-level feasibility check. Its role is **NOT approved**. Before obtaining data, resolve the author's use-permission note and obtain user's consent to the specific, limited file list; no bulk retrieval. If exact native attack behavior labels, benign support, capture-level provenance, matched feature validity or rights fail under fixed acceptance requirements, **terminate the search**.

Proceed to original revision-plan items independently of third dataset: 5-seed matched visible/blind CICIoT2023, repeated defensible CICIDS2017 partitions, at least 2 genuine literature method comparators, cross-method residual-routing/stability and machine-readable claim records. Scope change from the original mandatory third-primary ambition would then require explicit user decision and a dated campaign-spec addendum before freeze.

**No scientific policy, code, source files, experiments, release artifacts, campaign freeze or manuscript claims changed by this desk research.** All existing approvals and constraints remain.
