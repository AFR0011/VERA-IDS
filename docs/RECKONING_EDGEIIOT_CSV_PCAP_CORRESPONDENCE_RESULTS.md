# Edge-IIoTset C8 — Bounded CSV/PCAP head correspondence (2026-10-18)

**Evidence completed.** User uploaded the actual `outputs/17_revision_campaign/edgeiiot/csv_pcap_head_correspondence.json` from the existing revision-branch read-only scanner. Results recomputed independently from the 24 per-file records, with no actual data rewrites and no model training. This remains only a **first-64-row/packet ordinal timestamp audit**, not a validated CSV-to-packet join or independent experiment census.

## Recalculated observations

- Exactly **24/24 original CSV/PCAP pairs**, 64 ordinal positions each = **1,536 first-position comparisons**. All 24 PCAPs in the prior phase provided plausible full first-packet epochs.
- **21/24** CSV pairs have year + clock + microsecond fragments parseable without fabricating their absent month/day, so **1,344/1,536** ordinal pairs had comparable text times. The other **192** positions belong entirely to DDoS-UDP, MITM, and Modbus, whose original CSV first-field values were previously shown to be IP or numeric placeholders. These **cannot be interpreted as timestamp mismatches**.
- **885/1,344** parseable ordinal positions have an **exact microsecond** equality for year + time-of-day after adding a candidate **+1 hour** to the PCAP-presented UTC time. **459** parseable ordinal positions do not match at the identical head index. No other candidate integer hour shift in [-12,+14] produced any match in any file.
- **21/21 parseable source pairs have at least one exact +1-hour match**. **12/21 parseable pairs** have **all 64/64 ordinal positions matching**; 9 have incomplete ordinal correspondence; 3 more pairs have zero parseable CSV text times.
- Attack subset **426/768** matching parseable ordinals (**55.47%**) across 12/14 parseable attack files. Normal subset **459/576** (**79.69%**) across 9/10 parseable normal files. Combined **65.85%** match among parseable comparisons, or 57.62% of ALL 1,536 positions; **do not conflate these denominators** with packet-level recall or source coverage.

### Pair-level exact microsecond correspondence

| Category | Source CSV basename | Matched/parseable ordinal comparisons at +1 h |
|---|---|---:|
| Attack | Backdoor_attack.csv | 6/64 |
| Attack | DDoS_HTTP_Flood_attack.csv | 64/64 |
| Attack | DDoS_ICMP_Flood_attack.csv | 64/64 |
| Attack | DDoS_TCP_SYN_Flood_attack.csv | 64/64 |
| Attack | DDoS_UDP_Flood_attack.csv | 0/0 (unparseable original CSV time) |
| Attack | MITM_attack.csv | 0/0 (unparseable original CSV time) |
| Attack | OS_Fingerprinting_attack.csv | 11/64 |
| Attack | Password_attack.csv | 64/64 |
| Attack | Port_Scanning_attack.csv | 6/64 |
| Attack | Ransomware_attack.csv | 2/64 |
| Attack | SQL_injection_attack.csv | 64/64 |
| Attack | Uploading_attack.csv | 64/64 |
| Attack | Vulnerability_scanner_attack.csv | 16/64 |
| Attack | XSS_attack.csv | 1/64 |
| Normal | Distance.csv | 64/64 |
| Normal | Flame_Sensor.csv | 64/64 |
| Normal | Heart_Rate.csv | 24/64 |
| Normal | IR_Receiver.csv | 2/64 |
| Normal | Modbus.csv | 0/0 (unparseable original CSV time) |
| Normal | phValue.csv | 64/64 |
| Normal | Soil_Moisture.csv | 64/64 |
| Normal | Sound_Sensor.csv | 64/64 |
| Normal | Temperature_and_Humidity.csv | 49/64 |
| Normal | Water_Level.csv | 64/64 |

## Scientific interpretation

**[Certain]** The author-origin PCAPs have plausible absolute epoch timestamps and **21 original CSVs exhibit repeatable exact source clock alignment with them** in at least one first-64-pair comparison. Agreement in microseconds at the same ordinal index is a powerful clue for source-level correspondence, and the fixed +1-hour relationship is strikingly consistent in this bounded evidence.

**[Not established]** The source CSV publisher's actual timezone, complete packet-row identity, source *completeness*, original CSV label correctness, absence of feature serialization artifacts in original CSVs, or independent repetitions **within each attack family**. Strong head matches in some files DO NOT license injecting PCAP dates into source CSVs. Partial matches may be caused by different filtering, omitted protocol rows, sorting or CSV extraction, but none of those mechanisms was specifically established by the report. Zero parseable in three CSV families is a source metadata issue; their **PCAP timestamps were themselves valid** in Phase C7.

**Critically:** The published paper states that each attack was repeated at different hours/days, but the present samples cannot distinguish sessions or tell how many independent repetitions per native family are present. Dates across different attack-specific files are not independent-repetition support **for the same family**. This remains the main Protocol B gate.

## Next proposed evidence gate: full PCAP timestamp-metadata census

A whole-file, packet-HEADER-ONLY scan on the original 24 PCAPs is justified to determine **total captured packet records, date/hour occupancy, first/last/min/max UTC times and simple observed consecutive time gaps**, without any payload decoding or CSV repair. Time gaps are described as **episode proxies only, not authenticated independent attack runs**; even a gap of hours or days cannot by itself prove new attack executions. Bounded first-64-packet comparison can later be extended if it materially helps verify original CSV record identity.

Any proposed source-run grouping rule for Protocol B would be an actual methodological decision and **must return to the user for approval**, after reviewing time statistics and available capture/attack-log corroboration. No model experiments, split plans, feature policies or primary dataset promotion are authorized here.

No changes to historical COMNET release; all research additions must remain on `post-comnet-archival-value`. Canonical Drive execution log and decision log should mirror this result.
