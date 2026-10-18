# Edge-IIoTset Phase C7 — original PCAP packet timestamp probe (2026-10-18)

**Status: Actual 24-file bounded PCAP source scan COMPLETE; full-capture chronology and independence of repeated attack operations remain UNVERIFIED.** No models, source rewrites, feature exclusions, source-row labels, splits or manuscript claims changed.

## Observed evidence

The user uploaded `outputs/17_revision_campaign/edgeiiot/original_pcap_timestamp_probe.json` from the corrected `scripts/17_probe_edgeiiot_pcap_headers.py` on the local full original distribution. Report read and aggregates independently recalculated from its per-file entries.

- **24/24** expected original PCAPs successfully read: **14 attack**, **10 normal**, paired by author-supplied filename to their matching original CSV filenames (13 explicit attack aliases, 11 same-stem). **Filename-level correspondence ONLY**, not per-packet-to-CSV-record correspondence or label confirmation.
- All **24** classic PCAPs have little-endian byte order, **microsecond** timestamp resolution, Ethernet link type **1**, version **2.4**, snaplen **262,144**, with a full successful **64 packet-header** sample per file, **1,536 sampled packet headers**.
- Exactly **0** invalid timestamp fractional fields and **0** backwards time transitions among adjacent sampled packet headers. All 24 first packet epochs are plausible dated UTC instants according to the tool's range check. There is NO evidence of PCAP header/timestamp corruption in this bounded sample.
- First packet observed across all files ranges from **2021-11-21 10:44:10.081753 UTC** (`Temperature_and_Humidity.pcap`, normal) to **2022-01-05 07:46:13.592934 UTC** (`Soil_Moisture.pcap`, normal). First packets in the 14 attack PCAPs fall on **nine distinct UTC calendar days**: 2021-11-24, 11-25, 11-27, 11-29, 12-02, 12-04, 12-05, 12-06, 12-27. These first-packet dates are consistent with the author-documented collection window but DO NOT identify the full range, session gaps, or number of independent attack operations inside any one source file.
- Total inventoried PCAP file sizes: 14 attack approximately **1,133,162,788 bytes**, 10 normal approximately **1,072,762,450 bytes**. Sizes do NOT measure observation support or capture duration. Some heavy flood PCAP first64 packets span under a second; first64 window is not full capture length.

### Attack-PCAP initial capture times (actual observed UTC)

| Native attack source | First packet date/time UTC |
|---|---|
| Backdoor | 2021-12-05 16:58:23.150209 |
| DDoS HTTP | 2021-11-25 10:35:29.350784 |
| DDoS ICMP | 2021-11-24 22:17:49.149462 |
| DDoS TCP SYN | 2021-11-24 18:02:37.608273 |
| DDoS UDP | 2021-11-24 22:27:10.139612 |
| MITM | 2021-11-27 17:50:54.178322 |
| OS Fingerprinting | 2021-12-27 21:14:30.939803 |
| Password | 2021-12-04 00:45:35.961572 |
| Port Scanning | 2021-11-24 15:03:29.475073 |
| Ransomware | 2021-12-02 15:02:07.998055 |
| SQL Injection | 2021-11-29 20:35:18.238171 |
| Uploading | 2021-12-06 22:08:32.886770 |
| Vulnerability Scanner | 2021-12-06 18:22:08.593687 |
| XSS | 2021-11-29 16:44:47.438207 |

## Connection to original CSV's missing-date timestamps

The user earlier inspected actual `frame.time` values in a few source CSV files. **Backdoor** first CSV timestamp string has **year and local-clock-like time-of-day** `2021 17:58:23.150209000`; paired PCAP's first packet time is **2021-12-05 16:58:23.150209 UTC**. **Distance** CSV first value is `2021 23:58:21.314757000`; paired PCAP first packet at **2021-12-27 22:58:21.314757 UTC**.

Both first-record comparisons have exactly matching microseconds and a **+1 hour CSV-minus-PCAP time-of-day relationship**. **Inference, not independently verified full join**: the source CSV extract may represent the corresponding packet time in an offset local display, after losing month/day text. Two matching first records are credible cues for an ordinal/time comparison but not sufficient to claim the exporter timezone, row-wise packet correspondence, time-date repair, or that every individual capture has the same offset. Missing month/day is NOT reconstructed in any source file.

The prior CSV alignment scan had 6,144/6,144 exactly 63-field rows yet `frame.time` values in three source families that are IP-address or numeric literals rather than dates, and another 21 with incomplete year-time fragments. Thus **PCAP provenance is more informative**, but original CSV semantic inconsistency remains open. In particular, attack-file name identity is not an observation-level packet-to-CSV key.

## Next bounded source-correspondence diagnostic (Phase C8, not an experiment)

New code `scripts/17_compare_edgeiiot_csv_pcap_heads.py` and four synthetic tests `tests/test_edgeiiot_csv_pcap_head_correspondence.py` are committed to the same revision branch. The standalone code's two-source synthetic fixture demonstrated an exact +1-hour match for two Backdoor time-fragment records, zero parseable MITM numeric timestamps, correct aggregate-only output and traversal-path rejection. **The complete new pytest and user's 24-source run are pending**.

The script reads the **existing** `original_pcap_timestamp_probe.json` as the trusted local pairing inventory, validates explicit relative file paths, then samples first **64 original CSV time values and 64 matched PCAP packet time headers per pair**. For parseable year-time fragments, it tests *exact microsecond* matches at the **same ordinal index** over integer clock offset candidates **-12 to +14 hours**. It reports aggregate per-file counts and candidate offsets only. It does not reconstruct missing dates, decode packet payloads, export IPs or packet contents, train a model or identify attack executions. Negative/missing matches may simply indicate filtering/reordering/different number of CSV versus capture observations; do not label as dataset corruption from this alone. Hour offsets are **candidate mathematical relationships**, not asserted source timezone or clock calibration.

Execute locally from repo root **after** fast-forward and passing complete pytest:

```powershell
git status --short
git fetch origin
git merge --ff-only origin/post-comnet-archival-value
uv run python -m pytest -p no:cacheprovider
uv run python scripts/17_compare_edgeiiot_csv_pcap_heads.py --input-root "data/raw/Edge_IIoTset" --pcap-report "outputs/17_revision_campaign/edgeiiot/original_pcap_timestamp_probe.json" --out "outputs/17_revision_campaign/edgeiiot/csv_pcap_head_correspondence.json"
```

User should upload `csv_pcap_head_correspondence.json`. If the existing `original_pcap_timestamp_probe.json` is missing locally, do not substitute by invented locations. No need to rerun previous source scans. The script is first-head evidence only; independent repeated attack executions require a separate, more demanding **full-capture** chronology and validity campaign after approval of grouping criteria.

**Scientific decision:** Edge-IIoTset is more promising for recovering source chronology than the CSV fields suggested, but **NOT YET APPROVED** as third full primary Protocol B LOAO environment. The original paper's statements about repeated attacks must be supported with traceable per-execution evidence and defensible support partitions. Full independent dataset role, feature policy, campaign freeze and training remain pending user decision. Historic COMNET release unchanged and draft PR unmerged.
