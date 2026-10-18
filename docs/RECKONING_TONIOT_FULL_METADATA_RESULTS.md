# TON_IoT Network full metadata results — 2026-10-18

**Status: full row-count / timestamp metadata audit passed; leakage, ground-truth mapping, experimental splits, and primary-dataset eligibility NOT approved.** Evidence: user-uploaded `full_metadata_audit.json` produced locally by `scripts/17_audit_toniot_metadata.py`. SHA-256 of processed sources was not requested in this run (`hashes_complete=false`).

## Verified complete-source facts

- **23 processed CSVs**; **22,339,021 rows** counted; exact agreement with the author's official Network statistics and all ten published class totals. 22 files contain 1,000,000 rows; file 23 contains 339,021 rows. Local metadata report does not contain raw rows.
- **21,542,641 attack rows / 796,380 normal rows**; nine attack families. The only `label`/`type` pairs are `0/normal` and `1/`each attack category, with no observed pair contradictions.
- All 23 have valid finite numeric `ts` entries and zero malformed-width rows by this parser. **22 common 46-column schemas**; file 6 has an extra `uid` column (47 fields). Numeric parsability is not proof of correctly calibrated timestamps or source-event labels.
- **Ground-truth headers**: all 18 files use `ts,src_ip,src_port,dst_ip,dst_port,proto,type`. They were **not scanned for rows, label agreement or direct record correspondence**. User separately confirmed whole-file SHA-256 identity of `GroundTruth_Network_15.csv` and `GroundTruth_Network_16.csv`; the current metadata auditor did not hash them. No filename-based mapping is justified.

| Family | Full count | File numbers with records | First attack timestamp (UTC, assuming Unix seconds) | Last attack timestamp (UTC) |
|---|---:|---|---|---|
| scanning | 7,140,161 | 1–8 | 2019-04-23 12:10:10 | 2019-04-24 09:22:45 |
| dos | 3,375,328 | 8–11 | 2019-04-24 18:13:42 | 2019-04-25 02:50:21 |
| injection | 452,659 | 11–12 | 2019-04-25 05:49:20 | 2019-04-25 10:14:19 |
| ddos | 6,165,008 | 12–18 | 2019-04-25 14:48:47 | 2019-04-26 07:33:32 |
| password | 1,718,568 | 18–20 | 2019-04-26 13:45:48 | 2019-04-27 03:47:45 |
| xss | 2,108,944 | 20–22 | 2019-04-27 04:54:22 | 2019-04-27 12:40:53 |
| ransomware | 72,805 | 22 only | 2019-04-28 03:49:50 | 2019-04-28 06:05:25 |
| backdoor | 508,116 | 22–23 | 2019-04-28 07:29:59 | 2019-04-29 09:45:33 |
| mitm | 1,052 | 23 only | 2019-04-29 12:20:42 | 2019-04-29 14:45:29 |

**Temporal observation:** For all 22 successive file boundaries, previous `ts.max == next ts.min`, consistent with contiguous chronological segmentation. The nine attack-family `ts` spans do not overlap, and each attack type is confined to one relatively narrow capture interval. **Files are not demonstrated independent acquisition units; the source is strongly confounded by attack family and acquisition time.** A global time-ordered split cannot put every family into all train/validation/test partitions. Random within-family splits could supply class counts but introduce within-episode dependence and do not establish broader temporal generalization. Removing `ts` from model predictors alone is not sufficient to rule out proxies.

**Rare-family gate:** `mitm` has just 1,052 rows entirely in processed file #23 within a ~2.41-hour window; `ransomware` has 72,805 rows entirely in processed file #22 within ~2.26 hours. Total counts do not show multiple independent attack executions or meet Protocol B admissibility in independent time partitions. LOAO suitability remains unresolved for both, especially MITM.

## Next mandatory investigation

1. Stream-read the **18 actual ground-truth CSVs**; count every `type`, validate non-null/parseable timestamps, report per-file event/family spans, order violations, malformed widths, and schema differences. Only aggregate anonymized metadata; no raw network identifiers.
2. Investigate what a ground-truth row represents. Header `ts,src_ip,src_port,dst_ip,dst_port,proto,type` may be flow-level event labels rather than an independent attack schedule. Filename/index mismatch and GT #15/#16 duplication must not be repaired by presuming one-to-one pairing.
3. Separately test **label provenance and row-level overlap** against the processed source, identity-based leakage proxies, chronological source boundaries, and distinct attack events. These checks will need explicit methodological choices if evidence supports alternative protocols.
4. Decide with the user whether TON_IoT can serve Protocol A and/or Protocol B under a clearly qualified design, or should be limited to an **external stress test** rather than automatically promoted to a primary LOAO environment.

All findings are about the dataset. **No model has been trained, no feature blacklist/holdout/split threshold approved, no training hyperparameters chosen, and frozen historical release `release/v2026.10-comnet/` remains untouched.**
