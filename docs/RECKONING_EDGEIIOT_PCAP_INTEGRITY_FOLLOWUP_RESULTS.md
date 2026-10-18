# Edge-IIoTset Phase C10 — Targeted PCAP structural and shared-metadata audit

**Date:** 2026-10-18  
**Status:** User-local targeted diagnostic COMPLETE. Evidence validated against the user-uploaded `outputs/17_revision_campaign/edgeiiot/pcap_integrity_followup.json`. No training, source repair, CSV reconstruction, accepted train/test split, capture-group definition or dataset promotion.

## Evidence provenance and scope

- Report schema: `edgeiiot-c9-structural-followup-v1`; tool: `scripts/17_probe_edgeiiot_pcap_integrity.py`.
- Inputs: previously audited `full_pcap_timeline_census.json` (24 original PCAPs, 21,124,492 parsed header records, 23 reached EOF and 1 stopped at an invalid record) and the user's local original PCAP files.
- The follow-up scans PCAP record headers, **seeks rather than reads packet payloads**, and exports only aggregate structural fields and ordered timestamp/length-field SHA-256. It does not inspect packet contents, IPs, attack commands, process logs or original CSV row-level labels.
- The uploaded JSON was independently read and checked: prior and new scanner header counts agree; the failed byte offset plus remaining bytes equals file size; the captured-length violations are consistent; and the two benign day-subset SHA-256 strings and counts are equal.

## 1. Vulnerability Scanner: Repeatable PCAP record parsing failure

The same original `Vulnerability scanner attack.pcap` stops at a record boundary after **265,827 fully traversed packet records**; this number is a *known-good prefix under parser checks*, not a complete file or training-support count.

| Report field | Value |
|---|---:|
| Complete preceding packet-header records | 265,827 |
| First failing record-header offset | 111,714,300 bytes |
| Original PCAP file size | 140,377,652 bytes |
| Bytes remaining from failure offset | 28,663,352 |
| Bytes remaining after 16-byte header | 28,663,336 |
| Declared captured packet length | 136,146,411 bytes |
| Declared original packet length | 1,096,012,611 bytes |
| PCAP global-header snapshot length | 262,144 bytes |
| Timestamp fraction within range | **No** |
| Failing position resembles a recognized PCAP global magic | **No** |
| Reason codes | Captured length exceeds snaplen; exceeds remaining source bytes |

**Certain:** The 16-byte structure at this position is **not a valid packet record** under the classic PCAP rules used by the audit; the timestamp fraction is also out of range. Replaying the file produces the same stop point as Phase C9. **Not proven:** whether the publisher supplied damaged bytes, the user's original download is truncated or modified, a valid PCAP block is embedded later, or parsing became misaligned before this position. No automatic file salvage, repair or full packet counts should be inferred. Remaining ~28.66 million bytes cannot be claimed as traversed packets.

Scientific consequence: complete native-family capture coverage for Vulnerability Scanner **not verified**. All previous per-class date/gap statistics for that file are specifically *prefix-only*. Capture integrity failures cannot be silently treated as additional independent attack executions or valid label support.

## 2. Modbus vs Temperature/Humidity: Exact ordered header-metadata repetition

For UTC **2021-12-20**, the complete original PCAP scans found **3,537** packet records within each of the following normal device source files:

| Metadata | Modbus.pcap | Temperature_and_Humidity.pcap |
|---|---|---|
| Original file traversal | complete | complete |
| Total packet records in whole file | 159,514 | 1,629,749 |
| Records dated Dec 20 | 3,537 | 3,537 |
| First packet epoch/fraction in day subset | (1639959767, 981993) | (1639959767, 981993) |
| Last packet epoch/fraction in day subset | (1639960855, 43322) | (1639960855, 43322) |
| SHA-256 of ordered timestamp/length record fields | `23004b6434786150034c4700cc63a13d4f6ded2484327955096a946e3f3634f7` | **same** |

The audit serializes `ts_sec`, `ts_fraction`, `incl_len` and `orig_len` as fixed-endian 32-bit values for each Dec 20 record in **file order**, then hashes that ordered sequence. The report explicitly states `ordered_packet_header_sequences_identical_for_day: true`. This is much stronger than merely equal record counts and matching first/last timestamp values.

**Certain:** The ordered **packet header timestamp + length-field sequences** of the two day subsets are identical. **Not proven:** identical packet payload bytes, raw flow identity, execution/collection lineage, whether the day subset is contiguous in each larger file, or whether device-level labels were attached in the same way. It is appropriate to classify this as **reused/shared header metadata** and a potential cross-source independence contamination risk, not definitive byte-for-byte packet duplication.

Scientific consequence: treating normal source-file names as independent provenance groups would be unsafe without additional deduplication/lineage evidence and user-approved grouping rules.

## Cumulative Edge-IIoTset adjudication, not an unapproved policy change

1. Curated ML and DNN CSVs contain deterministic *verbatim string* binary-label proxies (`"0.0"` for three MQTT categorical fields in every attack row, never benign), independently reproduced in full on user-local curated files (Phase C3). This observation does not automatically prove the original PCAP packet content is mislabeled.
2. Original 24 CSVs have 63-field sampled row structure but inconsistent/incomplete `frame.time` metadata; 21 original CSVs exhibited sample correspondence of 885/1,344 parseable same-ordinal times under a candidate +1-hour relation to their paired PCAPs (C8). This does not prove a full packet-row join or timezone.
3. Full C9 PCAP headers provide useful chronology: 23 files reached EOF; one attack capture is structurally incomplete by our parser. **13 of 14 attack source files' inspected records have timestamps on only one UTC day**, and only Backdoor, Password and Ransomware exhibit observed file-order forward gaps >=1 hour. These are not source-authenticated repeated *same-family* attack executions. The original 2022 paper explicitly reports repeat attack experiments, but supplies no verified record-to-run manifest in the materials audited.
4. C10 confirms both a persistent nontrivial attack-source structural anomaly and exact repeated header metadata in two supposedly distinct benign source files.

**Proposed scientific verdict:** **NO-GO for admission as a third *primary* Protocol B LOAO dataset under the stated requirement for independently recoverable execution groups**, unless new authoritative attack-execution logs / capture-operation manifests / independently verifiable grouping metadata become available. This is a **recommendation awaiting explicit user approval**, not a unilateral change to the campaign specification.

A separately defined **qualified external stress/closed-set lane** might be scientifically defensible if the user first approves its limited role and if leakage, representation, overlap and capture-integrity restrictions are specified. A stress-only role is not automatically safe, and it must not be described as independent attack-run generalization.

### Decision options to resolve with user

- **A (recommended):** Stop Edge as a third primary Protocol B candidate; archive C3–C10 results; retain conditional stress-study potential without scheduling new experiments.
- **B:** Seek independent author-provided per-execution logs, original PCAP checksums/known-good copy and capture-to-CSV lineage. Do NOT infer independent sessions from observed time gaps.
- **C:** Explicitly develop a separate external-stress evaluation design, *not Protocol B*, conditional on leakage-safe preprocessing and verified source support. This requires its own user-approved specification.

**No more scanning is required to choose among those options.** Do not perform payload comparisons, PCAP salvage or classification training without further scientific purpose and explicit approval.

## Repository and handoff controls

Revision-only GitHub branch: `post-comnet-archival-value`, draft PR #3 **unmerged**. Historical COMNET release immutable. Phase C10 is a documentation/evidence checkpoint, **not a source-code change**. Preserve uploaded aggregate report in user-local ignored outputs and this distilled description in versioned docs. Canonical Drive execution, implementation and scientific decision logs must agree.
