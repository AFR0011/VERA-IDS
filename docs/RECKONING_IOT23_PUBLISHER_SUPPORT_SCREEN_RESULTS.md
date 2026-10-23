# IoT-23 I2 — Publisher full-population support screen, not a local full census

**2026-10-23** — Final VERA-IDS third-primary dataset candidate, author-published v1 native per-capture label tables compared with user-uploaded first-512-row `native_capture_preflight.json`.

**Status:** **PASS only for paper-level recurrent behavior presence.** Dataset Protocol B admission, capture independence, classifier feature policy and split strategy NOT approved. No 47.06GB scan, PCAP parsing, model fitting, labeling transform, threshold change, or partition creation has occurred.

## Sources and reproducible summary

Primary published source: https://www.stratosphereips.org/datasets-iot23 (IoT-23 v1.0.0, Stratosphere Laboratory; original scenario distributions under "Individual details for IoT-23 captures"). Publication DOI 10.5281/zenodo.4743746. Native author captures are malware-execution scenarios on Raspberry Pi. Official website says PCAP rotation can reflect duration/file size, so multiple PCAP files inside one capture directory are **not independent executions**. The source explains labeling through analyst-developed flow rules that may rely on destination IP, source/port context, byte patterns and temporal repetition, and its labels mix attack behaviors, compound behaviors, botnet identities and malware families.

Machine-readable transcription: `docs/RECKONING_IOT23_PUBLISHER_FOUR_LABEL_MATRIX.json` contains 20 author-malicious source scenario IDs and EXACT published per-capture counts for native `Benign`, `C&C`, `DDoS` and `PartOfAHorizontalPortScan`. Other native labels and all 3 standalone benign-device sources are explicitly OUTSIDE the matrix. Sum checks against original publisher's stated DDoS and scanning totals **PASSED**.

| Native exact label | Published flows across 20 malicious captures | Captures where >0 | Captures where >=1000 |
|---|---:|---:|---:|
| `Benign` | 30,858,735 | 20 | 19 |
| `PartOfAHorizontalPortScan` | 213,852,924 | 11 | 10 |
| `DDoS` | 19,538,713 | 7 | 6 |
| `C&C` | 21,995 | 10 | 5 |

The 1,000 cut is a **descriptive screening indicator only** and is NOT a replacement for the VERA Protocol B support rules. A same-family presence count across distinct capture directories is *not* per-family independent train/validation/test certification or any adopted scientific split.

### Original capture-level label examples

- **DDoS:** Capture 17-1 = 13,655,172; 60-1 = 3,578,457; 35-1 = 2,185,302; 43-1 = 65,803; 7-1 = 39,584; 34-1 = 14,394; 44-1 = only **1**. Thus *six* separately named captures have >=1,000 DDoS flow labels, not seven adequately populated capture groups.
- **C&C:** Only five captures reach >=1,000 exact flows: 8-1 = 8,222; 34-1 = 6,706; 43-1 = 3,498; 49-1 = 1,922; 39-1 = 1,530. Others include 35-1 =81, 44-1 =14, 1-1 =8, 3-1=8, 52-1=6. Publisher 42-1 explicitly has C&C =0.
- **Port scanning:** 11 captures with nonzero author counts; 10 >=1,000. Extreme dominance by 39-1 =73,559,437; 33-1=39,459,055; 43-1=37,911,674; 17-1=27,311,187; 52-1=19,779,564, etc. Capture 34-1 has only 122.
- **Benign:** Although 30,858,735 normal-labeled rows are reported *inside malware scenario captures*, those flows are not equivalent to independent benign-device captures and may share host/IP/collection context with their malware cohort. The three standalone original benign-device logs in our local preflight contain only 452, 512 prefix rows and 130 rows; more exact support may exist only in 5-1. Consequently, Stage1 matched 1.5% FPR support on fully distinct benign capture groups is NOT yet validated. Case `Benign` vs `benign` differs between source kinds; no data normalization authorized.

## Reconciliation with actual user preflight

Actual prefix `native_capture_preflight.json`: 23/23 labeled capture files found, 11,059 first rows inspected, same 23-field Zeek schema, zero prefix row-width/timestamp violations. Prefix-observed exact labels C&C 9 captures, DDoS 3, scan 9, Okiru 5; these are necessarily lower bounds on author-published source-wide **capture presence**, never representative class frequencies. Capture 9-1 first 512 rows all `Benign` but publisher reports 6,355,745 scan rows later in source: a concrete example why prefix-only non-observation cannot exclude a family.

**Source inconsistency needing attention:** Original v1 publisher capture 33-1 table labels 13,609,467 flows as exact `Okiru-Attack`, but user's actual first 512 native rows contain **160 exact `Okiru` rows**, alongside 326 scan and 26 Benign. The published table does NOT list any exact `Okiru` category for 33-1. Do not rename `Okiru` or `Okiru-Attack`, silently merge them or treat publisher table as infallible. Candidate core three behavior labels are not directly invalidated by this discrepancy, but source-version/label-table consistency needs a bounded check before admitting any new taxonomy.

Other scientific limitations: the author's rule-based labels may use the same IP/destination/connection-length information that would be passed into a classifier. **Accuracy against this labeling scheme risks validating a rule approximation rather than attack-independent detection.** The fact that multiple malware samples execute the same behavior is promising, but differences in label rules, device identity, malware origin and capture timing can confound cross-capture evaluation. No privileged identifier may become a predictor without explicit prior scientific approval.

## Economical next local experiment: three SMALL native source censuses only

**Do NOT launch full 47GB source scan yet.** To verify local source version and exact native label behavior with the existing read-only parser, fully scan three deliberately small original flow logs: 34-1 (~3.0MB), 44-1 (~31KB) and 3-1 (~24.4MB). These have overlapping but very different attack-class supports and cover one very small all-237-row capture; their complete publisher native label distributions are written on the original website. This is an implementation/metadata feasibility check, not a model experiment, independent-capture split or substantive row filter.

After fast-forwarding revision branch and receiving PASS on `uv run python -m pytest -p no:cacheprovider`, from VERA-IDS root:

```powershell
uv run python scripts/17_audit_iot23_capture_support.py --input-root "data/raw/IoT23/CTU-IoT-Malware-Capture-34-1" --expected-captures 1 --phase census --out "outputs/17_revision_campaign/iot23/capture34_native_counts.json"
uv run python scripts/17_audit_iot23_capture_support.py --input-root "data/raw/IoT23/CTU-IoT-Malware-Capture-44-1" --expected-captures 1 --phase census --out "outputs/17_revision_campaign/iot23/capture44_native_counts.json"
uv run python scripts/17_audit_iot23_capture_support.py --input-root "data/raw/IoT23/CTU-IoT-Malware-Capture-3-1" --expected-captures 1 --phase census --out "outputs/17_revision_campaign/iot23/capture3_native_counts.json"
```

Each reports `complete_source_row_census:true` **only for the selected 1 source**, with `capture_scenarios_found:1` explicitly recorded; do NOT confuse this with a 23-source-wide audit. No row-level IPs or payloads exported. Published 34-1 total=23,145 records, 44-1 total=237, 3-1 total=156,103, but the actual local results may diverge and must not be forced. Upload the 3 small aggregate JSONs. No further deep scan before reviewing them.

The decisive scientific follow-on, after cheap record-count/version verification: assess whether the author's full capture-level support plus precise behavior taxonomy and independent collected scenarios **can** pass the original support criteria under a prospective leakage-safe predictor policy. This may require an additional explicitly authorized source identity/group overlap audit or a selective full census for a frozen subset of candidate captures. If not, honor the already agreed FINAL-ATTEMPT stopping rule and return to the core CICIDS2017/CICIoT2023 revision workstreams. Any third-primary spec relaxation or experimental addition requires user approval.

**Do not rewrite manuscript or promote third primary dataset based on author tables alone.** No source file mutation, train/test splits, models or historic COMNET release changes occurred.
