# Edge-IIoTset: primary Protocol B candidacy closed

**Decision ID:** EDGE-PRIMARY-2026-10-18  
**Decision date:** 2026-10-18  
**Decision maker:** User (explicitly selected **Option A** after C10)  
**Status:** **ADOPTED: NO-GO as a third primary Protocol B LOAO dataset under current evidence.**  
**Branch:** `post-comnet-archival-value` (draft PR #3, unmerged); historical COMNET release unchanged.

## Decision and permitted scope

The Edge-IIoTset third-primary feasibility investigation is **closed**. The previously proposed Option A was explicitly accepted by the user ("A it is then...") after review of the complete Phase C3–C10 evidence. **This replaces earlier 'recommendation only / decision pending' statuses in chronological audit notes.**

**Do not** add Edge-IIoTset to the primary Protocol B evaluation, create independent execution groups from PCAP timestamps/gaps, or schedule further Edge source audits, extra downloads, source recovery, experiment runs, leakage preprocessor changes or model fitting as a continuation of this closed candidacy.

**Keep only conditional potential** for a *separately approved*, clearly qualified external stress/closed-set lane. This decision does **not** approve such a lane. It would require its own scientific protocol, leakage/feature safeguards, source-integrity/overlap handling, measurement plan and explicit user authorization before any code or experiments. A future author-provided, authenticated per-execution manifest or equivalent stronger evidence could justify **asking the user** to reopen primary candidacy; do not reopen unilaterally.

Primary CICIDS2017 and CICIoT2023 work continues under the existing revision campaign, with its previously approved conditions and unresolved decisions intact. The status of TON_IoT is a **separate, unresolved methodological decision**, not implicitly changed by the Edge decision. No alternative third primary dataset is automatically approved or required by this closure; any future change to that scientific requirement must be decided explicitly.

## Evidence basis: archived C3–C10 findings

1. **Original author distribution present:** 14 attack CSV/PCAP pairs, 10 normal device CSV/PCAP pairs, both curated ML/DNN CSVs. Original CSV headers consistently 63 columns in the 6,144 first-row field-alignment sample. Complete 24-PCAP inventory.
2. **Curated representation contains deterministic literal-text label leakage:** All attack rows and zero normal rows have exact `"0.0"` in each of three MQTT categorical fields `mqtt.conack.flags`, `mqtt.protoname` and `mqtt.topic` in both full curated CSVs (DNN 2,219,201 rows; ML 157,800 rows). This independently replicates a reported 2026 preprint serialization issue; it does **not** prove all original packet features or labels invalid.
3. **Original CSV time semantics questionable:** Several `frame.time` columns contain incomplete year-plus-clock fragments, IP literals or small numbers. Across first 64 ordinal rows/packets, 885 of 1,344 parseable paired time fragments match PCAP time-of-day exactly at a candidate +1-hour offset; this does **not** prove full row-to-packet joins, original timezone or restore missing dates.
4. **PCAP chronology insufficient for execution independence:** The uncapped header-only scan traversed 21,124,492 record headers; 23/24 PCAPs reached EOF. Among 14 attack file observed portions, 13 occupy one UTC calendar date and Ransomware two. Only Backdoor, Password and Ransomware show observed consecutive forward packet-time gaps >=1 hour. Author paper documents repeated attack activity, but **no verifiable record-to-independent-execution group labels were recovered for most native attack types**. No gap-based grouping can be defended as independently authenticated runs. File order includes backward timestamp transitions, limiting naïve discontinuity-based inference.
5. **Integrity/overlap limitations:** Vulnerability Scanner PCAP stops after 265,827 traversed records at byte offset 111,714,300 due to an invalid classic PCAP record (caplen 136,146,411 > snaplen 262,144 and remaining file data; timestamp fraction invalid); exact cause unverified and complete support unknown. Modbus and Temperature/Humidity normal PCAPs have **identical ordered timestamp+packet-length header sequences** for the same 3,537 records dated Dec 20, 2021. Payload identity and origin remain untested. Distinct filename is not assurance of independent normal source observations.

Together these observations make primary Protocol B source-group claims scientifically unsupported *under the current evidence and intended independent-execution requirement*. They are not grounds for asserting the underlying reported attacks never happened or that every aspect of the source data is unusable.

## Archive and handoff rules

**Preserve** all read-only diagnostics, synthetic tests, GitHub source-audit documentation, and user-local aggregate JSON reports under `outputs/17_revision_campaign/edgeiiot/`. The raw Edge-IIoTset acquisition remains local, ignored by Git; do **not** upload PCAP/CSV source bytes, alter or delete local raw data, or merge the draft PR as part of this choice.

Canonical supporting evidence:
- `docs/RECKONING_EDGEIIOT_SOURCE_FEASIBILITY.md`
- `docs/RECKONING_EDGEIIOT_PREFLIGHT_RESULTS.md`
- `docs/RECKONING_EDGEIIOT_ALIGNMENT_RESULTS.md`
- `docs/RECKONING_EDGEIIOT_PCAP_TIMESTAMP_RESULTS.md`
- `docs/RECKONING_EDGEIIOT_CSV_PCAP_CORRESPONDENCE_RESULTS.md`
- `docs/RECKONING_EDGEIIOT_FULL_PCAP_TIMELINE_RESULTS.md`
- `docs/RECKONING_EDGEIIOT_PCAP_INTEGRITY_FOLLOWUP_RESULTS.md`

**Authoritative cross-chat state:** Edge third-primary candidacy **CLOSED, user-approved Option A**; a hypothetical Edge external stress lane **NOT APPROVED**; independent source grouping **NOT VALIDATED**; all new Edge tasks **STOPPED**. If a historical document still says "decision pending", interpret it as its time-stamped record and supersede with this decision.

**Next thesis revision priority:** Return to the previously planned CICIDS2017/CICIoT2023 primary protocol, baseline/comparative methods and conclusion-stability study. Address the separate TON_IoT role and other campaign freeze questions by explicit user discussion rather than assuming an alternative third primary source. No model runs or scientific changes are triggered by this decision memo.
