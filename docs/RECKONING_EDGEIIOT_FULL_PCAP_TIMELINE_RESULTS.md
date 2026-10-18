# Edge-IIoTset Phase C9 — Full original PCAP timeline audit (2026-10-18)

**Actual uploaded evidence:** `outputs/17_revision_campaign/edgeiiot/full_pcap_timeline_census.json`. We read the actual 24-file user-local report, independently recomputed per-group counts, checked all reported UTC day/hour packet-count sums against parsed packet counts, and inspected the one incomplete source separately. This is **packet HEADER chronology only**, not record-level labels, packet-content hashes or authenticated independent attack operations.

## Whole-file scan status and integrity

- `max_packets_per_file: 0`; total headers parsed **21,124,492** across all 24 named original source PCAPs (14 attack, 10 normal).
- **23/24** files reached `full_file_scanned` and EOF; one attack PCAP, `Vulnerability scanner attack.pcap`, stopped at `invalid_record_size_or_truncation` after **265,827 valid preceding packet headers**. The actual culprit might be a bad captured length, truncation, a nonstandard embedded header, or another structural change. **The report does NOT diagnose which**, and source completeness for that file is UNKNOWN. Do not equate 265,827 with its complete observation support.
- Parsed attack prefix packet headers **9,865,867** (contains incomplete Vulnerability Scanner prefix); fully scanned normal packet headers **11,258,625**. These are capture packet counts, **not dataset labeled sample counts**.
- **Zero** sampled-to-scanned packet headers have implausible epoch seconds or timestamp fraction >= timestamp resolution. **19,346 backward adjacent timestamp transitions** were counted in the stored packet order (**14,228 attack, 5,118 normal**). Hence consecutive forward gaps are descriptive observations in file order, NOT sorted-time event segmentation. All source UTC day/hour packet-count sums reconcile with the parsed records (no impossible timestamp epoch values).
- Historic scanner printed `[PASS] Saved metadata` with a warning for incomplete status. That message meant the JSON was SAVED, **not** that all sources reached EOF. To prevent confusion in future runs, `scripts/17_census_edgeiiot_pcap_timeline.py` now returns **exit code 2 and prints [PARTIAL]** whenever one or more scans are incomplete/capped, while retaining the diagnostic JSON. The two existing incomplete/capped regression tests now assert this. **Do not rerun the full 21M-header scan just to test the exit-code correction.**

## Native attack capture chronology

| Attack source | Parsed packet headers | Occupied UTC days | Largest adjacent forward gap | Scan complete |
|---|---:|---:|---:|---|
| Backdoor | 24,914 | 1 | 1.56 h | Yes |
| DDoS HTTP | 229,142 | 1 | 4.02 s | Yes |
| DDoS ICMP | 2,914,356 | 1 | 1.20 s | Yes |
| DDoS TCP SYN | 2,020,152 | 1 | 1.63 s | Yes |
| DDoS UDP | 3,215,732 | 1 | 1.57 s | Yes |
| MITM | 1,229 | 1 | 10.12 s | Yes |
| OS Fingerprinting | 1,176 | 1 | 40.46 s | Yes |
| Password | 1,053,893 | 1 | 14.88 h | Yes |
| Port Scanning | 23,329 | 1 | 2.25 s | Yes |
| Ransomware | 11,030 | **2** | **30.10 h** | Yes |
| SQL injection | 51,228 | 1 | 51.56 s | Yes |
| Uploading | 37,644 | 1 | 38.58 s | Yes |
| Vulnerability scanner | 265,827 | 1 in scanned prefix | 52.24 s | **No** |
| XSS | 16,215 | 1 | 37.95 s | Yes |

- **13 of 14 attack PCAP files**, including the incomplete Vulnerability Scanner prefix, contain observed packet timestamps on only ONE UTC day. Only Ransomware has capture packet times on multiple days (**December 2 and 3**).
- Only **three attack PCAPs** contain one or more adjacent forward gaps >=1h in their scanned records: Backdoor **5,599.893 s**, Password **53,582.801 s**, and Ransomware **108,365.690 s**. Only Ransomware has a gap >=24h. Eleven others, including the partially scanned Vulnerability Scanner file, have **no observed >=60 s gap**. These boundaries are candidate time discontinuities, NOT independently recorded execution attempts, source-independent attack sessions, or Protocol B test folds.
- Seven of the 14 attack classes have quite short continuous windows and several have only tens of thousands or ~1,000 captured packets. The millions of ICMP/TCP/UDP flood packets occur in single-day, single-hour source windows: large packet count is **not** independent capture diversity.
- The Ferrag et al. TechRxiv paper claims attacks were repeated across different hours/days. Our observations DO NOT falsify that author statement, but the *distributed original PCAPs* do not provide verifiable repeat-execution IDs for most classes. Repetitions could be within a continuous capture or not individually preserved in distribution. No arbitrary gaps should be promoted to authenticated sessions.

## Normal capture chronology and possible shared segment

- **5 of 10** normal-device PCAPs have observed packets on multiple UTC calendar days: IR Receiver (Jan 4–5); Modbus (Dec 20, Jan 5); pH Value (Dec 27–28); Temperature/Humidity (Nov 21–23, Dec 20); Water Level (Dec 28, Jan 4).
- **3 normal files** have gaps >=24h: Modbus ~16.2 days, Temperature/Humidity ~26.1 days, Water Level ~7.7 days. This is multi-period BENIGN data evidence, **not** repeated attack families.
- **Potential shared benign-source segment:** normal Modbus and Temperature/Humidity each have **exactly 3,537 PCAP packet headers on 2021-12-20**. Both the first and last timestamps of this date-specific segment coincide at **00:22:47.981993** and **00:40:55.043322 UTC**. This is highly suspicious of shared/reused capture metadata but NOT YET proof of identical packet records or raw payload. Separate read-only check will fingerprint the **ordered timestamp + captured length + original length packet HEADER fields only** on that UTC day in both files. If fingerprints match, that proves identical ordered header metadata, **not identical packet payload bytes** or verified provenance.

## Next narrow investigation, no methodology adoption

A new tool `scripts/17_probe_edgeiiot_pcap_integrity.py` and `tests/test_edgeiiot_pcap_integrity.py` have been added on branch `post-comnet-archival-value`.

1. For the original Vulnerability Scanner PCAP only, replay record-header traversal (about 265,827 headers) until the first failing record, and report byte offset, remaining bytes, caplen, original length, snaplen and which length constraints failed. It also reports whether the failing position looks like a new PCAP global header. It reads **no packet payload bytes**. This should resolve whether the earlier generic error was due to declared record length or a plausible concatenated format boundary, but cannot automatically repair or authenticate data.
2. For the two NORMAL device PCAPs only, calculate SHA-256 of the ordered **packet header timestamp and length fields** for records on 2021-12-20, recording counts, first/last metadata, and only header-sequence equality. Does **not** read, hash or expose packet payloads, IPs, flow identifiers or CSV rows.
3. Four synthetic tests passed in isolated authoring sandbox for intentionally invalid lengths, equal vs unequal timestamp/length fingerprints, path traversal and capped report refusal. User-local full repository test run on latest revision branch PENDING.

Run from existing Windows PowerShell repository root, ONLY after updating branch and full pytest PASS:

```powershell
git status --short
git fetch origin
git merge --ff-only origin/post-comnet-archival-value
uv run python -m pytest -p no:cacheprovider
uv run python scripts/17_probe_edgeiiot_pcap_integrity.py --input-root "data/raw/Edge_IIoTset" --timeline-report "outputs/17_revision_campaign/edgeiiot/full_pcap_timeline_census.json" --out "outputs/17_revision_campaign/edgeiiot/pcap_integrity_followup.json"
```

Upload `pcap_integrity_followup.json`. **No full 24-PCAP rescan needed.** New tool is an integrity follow-up, not attack grouping or a new Protocol B methodology.

## Scientific recommendation, pending user approval

Under the proposed strict requirement for **independently recoverable repeated attack-execution groups within the same family**, this dataset is **not yet demonstrably admissible** as a third PRIMARY Protocol B LOAO dataset. Original PCAP timestamps are useful and some benign recording periods can be differentiated, but most attack classes have only one observed capture day and no authenticated per-execution manifest. The original CSVs contain malformed/incomplete timestamp fields, the curated ML/DNN variants have independently confirmed deterministic binary-label serialization leakage, and the Vulnerability Scanner PCAP is incomplete under the current classic-PCAP structural rules.

Recommended conditional decision **for user review after structural follow-up**: do not promote Edge-IIoTset to third primary Protocol B without recovering authoritative attack logs/session IDs or equivalent defensible independent source units. A carefully framed, leakage-aware *external stress/closed-set* lane may be useful if separately specified and explicitly approved. Do not treat published source capture names or elapsed time gaps as independent runs. The older COMNET release must remain untouched and the revision PR unmerged.
