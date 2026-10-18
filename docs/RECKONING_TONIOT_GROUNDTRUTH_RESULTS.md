# TON_IoT Network ground-truth census and reconciliation — 2026-10-18

Status: **Full 18-file ground-truth census completed on user's local source; direct event-to-flow reconciliation, independent labels, splits, and primary dataset acceptance all remain OPEN.**

Source: `outputs/17_revision_campaign/toniot/groundtruth_full_census.json` provided by user. Compare with `full_metadata_audit.json` (22,339,021 processed Network rows). Ground-truth census command reported [PASS] Examined 17,801,188 ground-truth rows **as stored**.

## Verified ground-truth metadata

- 18 `GroundTruth_Network_*.csv` files, 17 containing 1,000,000 rows each and file 18 containing 801,188 rows; all have `ts,src_ip,src_port,dst_ip,dst_port,proto,type`. The source contains *attack-family records only*, with no `normal` in reported class counts. No malformed-width records, absent required fields, or invalid finite-number timestamps.
- One **backward timestamp transition** in `GroundTruth_Network_1.csv`; all other files report none. This requires row-level examination before declaring chronology sorted throughout.
- Files 15 and 16 contain 1,000,000 rows each, **identical source bytes per independent user-run SHA-256 checks**, including identical `password` (534,871) and `xss` (465,129) class counts. The new ground-truth census itself had `hashes_complete=false`; its empty duplicate-hash mapping is **not** evidence of no duplicates.
- After excluding one COPY of that confirmed duplicate whole file, the ground-truth archive contains **16,801,188 rows as stored in distinct source files** (not necessarily unique rows). Preserve both original input files, never delete them or assume source file numbers map to processed file numbers.
- File 14 ends at numeric timestamp 1556295451, file 15 begins at 1556323577, a gap of 28,126 seconds; 15/16 repeat the same period. The source is not proven to be independent attack schedule records.

## Source class-count discrepancy

| Attack family | Processed attack rows | GT rows as stored | GT rows excluding one duplicate file | Adjusted GT / processed |
|---|---:|---:|---:|---:|
| scanning | 7,140,161 | 7,138,336 | 7,138,336 | 99.974% |
| ddos | 6,165,008 | 6,031,012 | 6,031,012 | 97.827% |
| dos | 3,375,328 | 196,227 | 196,227 | 5.814% |
| injection | 452,659 | 451,136 | 451,136 | 99.664% |
| password | 1,718,568 | 1,253,031 | 718,160 | 41.788% |
| xss | 2,108,944 | 2,535,943 | 2,070,814 | 98.192% |
| ransomware | 72,805 | 5,629 | 5,629 | 7.732% |
| backdoor | 508,116 | 188,864 | 188,864 | 37.169% |
| mitm | 1,052 | 1,010 | 1,010 | 96.008% |
| **Total attack** | **21,542,641** | **17,801,188** | **16,801,188** | **77.990%** |

These are **only ratios of class counts, not verified row matches or coverage**. In particular, if two files use distinct flow-generation / formatting rules, the numerical ratios do not indicate correctness or recall. The class-selective gaps (especially DoS, ransomware, password, and backdoor) make use of these ground-truth records as independent, exhaustive label verification scientifically indefensible without further evidence.

## Next source-consistency pilot

New read-only script: `scripts/17_pilot_toniot_flow_match.py`. It deterministically reservoir-samples up to 512 raw GT rows **per GT file per attack type**, takes exact textual six-field flow identities (`ts,src_ip,src_port,dst_ip,dst_port,proto`) and scans processed files to see whether the same six-field key appears with `label=1` and the same `type`. Reports aggregate match, identity-only label discrepancy, and absence counts. **Does not export IPs, identities, hashed IDs, or raw records.** Includes GT #15 and #16 in as-stored results and reports both separately; do NOT count them as independent evidence. If matches fail, check field-formatting differences before inferring false labels.

From the repository root:

```powershell
git fetch origin
git merge --ff-only origin/post-comnet-archival-value
uv run python -m pytest -p no:cacheprovider
uv run python scripts/17_pilot_toniot_flow_match.py --groundtruth-root "data/raw/ToN_IoT/SecurityEvents_Network_datasets" --processed-root "data/raw/ToN_IoT/Processed_Network_dataset" --out "outputs/17_revision_campaign/toniot/groundtruth_flow_match_pilot.json"
```

Each full scan reads source data, so runtime may be nontrivial. Do not claim the script has been tested on the user's full source until output is received. Synthetic tests on the revision branch still require the user's updated full-suite run.

**Scientific limits:** This is cross-file *internal consistency*, **not independent ground truth**, independent source/time group detection, full join, provenance audit, label reconstruction, duplicate flow audit, or LOAO eligibility. It cannot establish unseen-family test support, the number of independent attack executions, or whether features encode attack timestamp/source identity. The method of using TON_IoT for Protocol A, Protocol B or external stress testing is reserved for user approval after audit. Historical COMNET release is unchanged.
