# IoT-23 I3 — Three user-local complete capture censuses reconcile with original publisher

**Date:** 2026-10-23  
**Scope:** Read-only full-file exact native Zeek `conn.log.labeled` census of *three selected malware captures*, NOT the 23-capture/roughly 47.06 GB labeled-flow corpus, NOT model training, and NOT Protocol B admission.  
**Input evidence:** User-uploaded `capture34_native_counts.json`, `capture44_native_counts.json`, `capture3_native_counts.json`. Script: `scripts/17_audit_iot23_capture_support.py --phase census --expected-captures 1`, revision branch `post-comnet-archival-value`.  
**Author's original distribution tables:** https://www.stratosphereips.org/datasets-iot23 (2020, v1.0.0; DOI: 10.5281/zenodo.4743746). Independent verification used the publisher's *complete label distribution* for each of the three specific captures, not just selected-label GitHub matrix. Corresponding previously committed subset table: `docs/RECKONING_IOT23_PUBLISHER_FOUR_LABEL_MATRIX.json`.

## Actual local complete-record results

| Exact native `detailed-label` | CTU-IoT-Malware-Capture-34-1 | Capture-44-1 | Capture-3-1 |
|---|---:|---:|---:|
| `-` (binary `Benign`) | 1,923 | 211 | 4,536 |
| `C&C` | 6,706 | 14 | 8 |
| `DDoS` | 14,394 | 1 | 0 |
| `PartOfAHorizontalPortScan` | 122 | 0 | 145,597 |
| `C&C-FileDownload` | 0 | 11 | 0 |
| `Attack` | 0 | 0 | 5,962 |
| **Total** | **23,145** | **237** | **156,103** |

**Independent internal audit invariants** checked per input JSON:
- One and only one capture per output; `mode=census`, `rows_per_capture_limit=0`, `complete_source_row_census=true`, exact canonical source capture ID; no source report `problems` or `source_metadata_warnings`.
- Across all three: **179,485** parsed flow rows, **0** malformed field widths, **0** invalid or implausible timestamp rows, 23 identical native Zeek fields. All `native_binary_label_counts`, `native_detailed_label_counts`, `native_label_pair_counts`, and `utc_day_counts` sum to the corresponding total row count, with consistent benign vs malicious detailed labels.
- 34-1 spans UTC Dec 21–22, 2018; 44-1 Jan 10, 2019; 3-1 May 19–21, 2018. These source timestamps do not certify multiple independent attack executions *within* a scenario.
- Each local **entire native detailed-label distribution** (all listed exact strings and counts) matches its corresponding original publisher table, not merely the four source strings transcribed into the earlier machine-readable matrix:
  - 34-1 `Benign` 1,923, `C&C` 6,706, `DDoS` 14,394, scan 122: **all exact**.
  - 44-1 `Benign` 211, `C&C` 14, `C&C-FileDownload` 11, `DDoS` 1: **all exact**.
  - 3-1 `Attack` 5,962, `Benign` 4,536, `C&C` 8, scan 145,597: **all exact**.
- Hence the previous prefix-sample differences are explained by position/coverage, not parser failure in these captures. The full local `Attack` and `C&C-FileDownload` occurrences, absent from our four-label *subset* matrix, also agree with publisher tables.

**Evidence-strength interpretation:** This is strong count/schema evidence that the downloaded original v1 data and parser align with the public distributions for **three captures only**. It increases confidence in using the original published full-capture label tables for the next *feasibility screen*. It does NOT prove per-flow labels are attack-independent truth, validate all remaining twenty captures, establish independent capture operations, settle the source-vs-publisher capture 33 `Okiru`/`Okiru-Attack` discrepancy, demonstrate source-clean test partitions, or establish absence of label-proxy leakage.

## Remaining scientific eligibility gates, prioritized

1. **Semantic contract:** Publisher native `detailed-label` strings mix behavior labels (`PartOfAHorizontalPortScan`, `DDoS`, `C&C`), compound labels (`C&C-FileDownload`, etc.), a generic `Attack` label, and botnet-family indicators (`Okiru`, `Okiru-Attack`). The source documents explicit manual labeling rules using network flow/identity/traffic statistics. Preserve exact source labels and specify any eligible subset or hierarchical taxonomy *before training*, with user approval. Do not silently fold `Okiru` variants or unknown compounds.
2. **Capture-level independent-group feasibility:** The publisher describes 20 separately identified malware-infected IoT captures plus 3 standalone benign device captures. This is a plausible unit boundary. It is not a guarantee of independent traffic environments or adequate known-family and benign support under the *existing* Protocol B train/validation/test group support thresholds. Multiple PCAPs inside one capture folder may be rotations, not repetitions. Source rule-based labeling may imprint capture-dependent IP/destination/service identity into input features.
3. **Benign operating-policy feasibility:** The author reports **30,858,735** benign-labeled flows embedded across 20 infected scenarios. Those are source-correlated with malware and cannot be automatically treated as interchangeable with the three standalone benign scenarios. Two independent native benign-device logs are fully small (4-1: 452, 7-1 Somfy-01: 130), while capture 5-1 was only prefix-sampled; its entire `conn.log.labeled` is ~182 KB and can be counted cheaply, if needed. The native lowercase `benign` vs malware-source uppercase `Benign` must be documented before any feature/label transformation, and matched FPR policy needs adequate validation/test benign denominators.
4. **Source version mismatch:** In capture 33-1, initial local 512 records contained **160 exact `Okiru`**, while the original publisher's table reports **13,609,467 exact `Okiru-Attack`** and does not list `Okiru` for 33-1. No correction or unapproved full 7.87GB capture scan is justified just by that fact; investigate narrow source metadata/label annotations if this capture or class becomes scientifically necessary.
5. **Compute and test leakage:** Full 23 native labeled logs are ~47.06 GB, so training on every flow is not mandatory and data caps may be defensible only after a predeclared, group-preserving sampling policy. VERA's original stage/rejector selection logic and the 1.5% benign FPR validation design must not be silently rewritten.

## Next scoped study proposal — no heavy scan authorized

**Recommended immediately:** Treat the first-stage publisher table/real-file correspondence check as PASSED, but evaluate whether the *existing* support policy and a capture-level provenance boundary are even jointly feasible for the three recurrent exact behavior labels **without adopting a split**. Research source labeling design and independently captured malware units in the original per-scenario READMEs; verify complete `CTU-Honeypot-Capture-5-1` benign count if needed (~182 KB) rather than source-wide 47 GB census. Produce a clear stop/go proposal, preserving known-vs-unknown support, limited valid families and leakage risks. If foundational source identity and rule leakage are not defensible, **stop the final third-primary search** as promised and return to original paired-visible/blind, repeated-partition, literature-method and claim-stability campaign.

**No third primary acceptance**, feature/taxonomy/partition rule, model training, source modification, full source census, or historic COMNET release mutation is authorized by the three completed small censuses. Draft PR #3 remains unmerged. 
