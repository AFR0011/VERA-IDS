# IoT-23 final primary Protocol B admission review — source-rule and capture-support gate

**Date:** 2026-10-23  
**Decision state:** **RECOMMEND NO-GO as a third full PRIMARY Protocol B dataset under the currently specified VERA-IDS validity and comparability standard. This is a scientific recommendation, NOT a user-approved closure yet.** User authorized the final source review; no dataset-role, taxonomy, feature, source-group, model or experiment policy has been adopted.

## Evidence and external source attribution

1. Repository: `docs/RECKONING_IOT23_PUBLISHER_FOUR_LABEL_MATRIX.json`, original Stratosphere v1.0.0 malicious capture-level exact label counts transcribed from the official source, validated against actual complete native local labels in captures 34-1, 44-1 and 3-1 (179,485 total flows) and actual complete standalone benign capture 5-1 (1,374 flows).
2. Official dataset collection/labeling source: https://www.stratosphereips.org/datasets-iot23 . The authors report 20 infected-IoT malware scenarios (each with a specific malware sample executed on a Raspberry Pi) and 3 stand-alone real benign IoT device captures. Source original network PCAPs were manually inspected; an analyst-developed Flaber program applied flow-based labeling rules. PCAP rotations of the SAME capture are not independent executions.
3. **Primary, public, capture-specific original label design:** https://docs.google.com/spreadsheets/d/1HRqgKJp0XoSUIfW3rCQKoD_LnSCJ1k-k61PndJXWq_o/edit . The sheets expose exact native Zeek feature/field names, comparison criteria, native attack labels and connectors for all 20 malware captures. Reviewed native tabs for all 20 in bounded A1:H45 reads; input-field and label metadata only are reported below. Raw IP values, individual labeled flow rows and rule addresses are NOT reproduced or added to Git.
4. **Frozen current Protocol B support config, unmodified:** `config/protocol_b_loao.yml`, `config/protocol_b_support_ciciot2023.yml`. Both require >=200 benign validation and test flows; >=200 held-out unknown validation/test flows; >=200 train, validation and test flows per known family; at least two other known families after holdout, with every retained known family support-valid.

## Gate 1: Numerically feasible for three exact native behaviors — PASS as arithmetic ONLY

For the *exact, unmerged, mutually distinct* native detailed-label strings `C&C`, `DDoS`, `PartOfAHorizontalPortScan`, a whole-capture-group **mathematical existence witness** (NOT A RECOMMENDED OR APPROVED PARTITION) is:

| Abstract partition label | Original malware source IDs | Published C&C | Published DDoS | Published PartOfAHorizontalPortScan | Published native Benign within infected captures |
|---|---|---:|---:|---:|---:|
| train | 43-1 | 3,498 | 65,803 | 37,911,674 | 20,574,934 |
| validation | 34-1, 39-1 | 8,236 | 14,394 | 73,559,559 | 9,260 |
| test | 8-1, 17-1 | 8,222 | 13,655,172 | 27,311,187 | 33,619 |

All rows reproduce the publisher's exact source support and have no capture ID shared *between* abstract partitions. The three candidate behavior labels each satisfy at least 200 rows in each group. Under a conditional protocol that **removes the held-out behavior from training rows** and retains the other two as known labels, each individual held-out behavior can numerically meet the existing val/test Unknown >=200 rule; the two remaining known behavior classes meet >=200 train/val/test. There is no adopted split, no class filtering or taxonomy mapping, and no claim of certified independent runs.

**Critical limitations to this arithmetic:** The source grouping is an illustration, not a reproducible random partition protocol with multiple valid partition seeds, source-unit independence proof, no-identity-overlap assurance or train/val/test test-blind selection. A large number of **other** original native malicious labels (e.g. `Okiru`, `Attack`, compound C&C variants) are present in source captures and excluded from the four-label source support matrix. Deciding to drop, merge, reassign or reject these observations would materially alter Stage 2's native classification task and needs separate approval. Known-family support in train may come from a **single malware execution/source**, even though it has millions of flows: row counts must not be presented as repeated experiments. The three exact behavior classes are not semantically the same taxonomic scheme as CICIDS2017/CICIoT2023 attack families.

## Gate 2: Author-provided label-rule overlap with candidate model features — SCIENTIFIC BLOCKER

The original author-published per-scenario spreadsheet shows that labels are assigned with deterministic field comparison rules applied to native Zeek flow attributes. For all native exact-label-positive scenarios identified in the original four-label capture-level matrix, corresponding public rule tabs contain predictor fields as follows:

| Exact native malicious label | Positive source captures | Public rule tabs using readily available native Zeek fields |
|---|---:|---|
| `C&C` | 10 | All 10 include destination port and/or destination IP (sometimes protocol) |
| `DDoS` | 7 | All 7 include destination port; several also include destination IP or protocol |
| `PartOfAHorizontalPortScan` | 11 | All 11 include destination port; 8/11 also include `conn_state` |

Author **metadata inspection**, not replayed data: rule rows may include conjunctions, priority rules, aliases/typos such as `PartOfAHorizontalPortscan`, and residual/compound labels. Counts of flows deterministically reconstructible from an individual field have **not** been quantified and no exact rule-to-label equivalence is claimed.

Especially relevant to the mathematical existence witness:
- Capture 43-1: source C&C uses `id.resp_p`/`id.resp_h`, DDoS `id.resp_p`, scanning `id.resp_p` + `conn_state`.
- Capture 34-1: C&C uses `id.resp_p`, DDoS `id.resp_h`/`id.resp_p`, scanning `id.resp_p`.
- Capture 39-1: C&C `id.resp_p`, scanning `id.resp_p` + `conn_state`.
- Capture 8-1: C&C `id.resp_p`.
- Capture 17-1: DDoS `id.resp_p`; scanning `id.resp_p` + `conn_state`.

The original publisher also explicitly describes behavior heuristics including repeated destinations, bytes transmitted and suspicious C&C endpoints. Thus training an IDS to predict these labels using the same native columns risks **annotation/predictor circularity**. Removing only `label` and `detailed-label` is plainly insufficient; including `id.resp_p`, `id.resp_h`, `conn_state` can reproduce parts of author labeling rules. Removing direct rule fields alone is NOT a validated fix: service, bytes, packet history, timing and source-specific patterns may remain proxies for the same constructed labels, and removing ordinary network semantics changes the task substantially. No such feature blacklist or reduced-feature protocol has been approved.

This is a *source-grounded high-risk incompatibility* with treating accuracy/LOAO detection as independent proof of generalizable attack recognition. It does NOT imply the malicious traffic never happened, all labels are false, or every sample label is fully determined by one column. A separate **label-rule/proxy bias investigation** could have research value, but is not automatically a third primary VERA Protocol B replication.

## Gate 3: Benign capture and source independence — CONDITIONAL, NOT CERTIFIED

Actual three independent-capture-ID native benign source counts: 4-1=452, 5-1=1,374, 7-1/Somfy-01=130, total **1,956 source flow observations across THREE labeled source contexts**. Two individually exceed the configuration's numeric 200-row validation/test floor; Somfy-01 does not. The publisher's other Somfy child folders contain PCAPs but lack native labeled Zeek flow logs and cannot be counted as independent benign labeled observations. A separate three-way fully benign-device-capture source assignment with >=200 in **every** partition is **impossible with just these three standalone groups**; note this is a stronger diagnostic requirement than the existing config, which explicitly demands 200 in val/test but has no numeric benign train minimum.

Publisher also reports **30,858,735 native uppercase `Benign`** flows inside 20 infected-source captures, including counts in the abstract existence witness. They can meet raw val/test minima, but those rows share a source context with malicious flows and may be benign only under the rule annotation. Treating them as freely exchangeable with standalone healthy-device captures or as independent Stage 1 generalization contexts is not justified without an explicit, prospective benchmark interpretation and provenance/label-quality checks.

Captures represent separately executed malware samples on controlled Raspberry Pis, which is substantially better documented than TON_IoT/Edge original CSV chunks. However, hardware/network reuse, capture-specific malware family, destination/port rules and shared environment can confound source-independent generalization. A capture scenario is a defensible **candidate** holdout group, not a guarantee of event-independent attack family replication.

## Final scientific recommendation and stopping rule

**Primary Protocol B full replication: RECOMMEND NO-GO under the current scientific bar, pending explicit user decision.** Even though the **mathematical support existence gate PASSES**, label-target generation is explicitly entangled with features a natural Zeek-based IDS would otherwise use. Only three exact recurring behavior labels support a minimal two-known-family LOAO problem under the illustrative grouping, and raw source flow counts are massively imbalanced. An arbitrary reduced-feature/label-filtered task cannot silently substitute for native-family, source-audited comparable Protocol B experiments. A no-go should be framed as a **methodological admissibility verdict**, not dataset corruption or inability to represent malicious traffic.

**Recommended project action after user agrees:** Close this last third-primary acquisition search without another dataset probe; archive IoT-23 as conditional non-primary **label-rule validity / stress-case** material, explicitly NOT approved for training. Retain CICIDS2017 and CICIoT2023 as the original primary datasets and focus on planned paired validation-visible/blind trials, five admissible CICIDS2017 partition repetitions, 2+ genuine published method baselines, and machine-readable conclusion-stability records. Explicitly amend the previously intended third-primary mandatory spec **only with user approval** before campaign freeze. Maintain TON role separate and Edge adopted primary no-go; historical `release/v2026.10-comnet/` unchanged, draft PR #3 unmerged.

**No model runs, local source file modifications, label decisions, new partitions, full Zeek census or group assignments have been enacted.** This report is a decision proposal, not a protocol/closure approval.
