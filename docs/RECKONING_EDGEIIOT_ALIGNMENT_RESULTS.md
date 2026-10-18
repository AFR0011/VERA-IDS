# Edge-IIoTset original CSV alignment results — Phase C5 (2026-10-18)

**Status: complete bounded positional audit of actual original CSV sources; independent capture/attack-run groups remain UNRESOLVED.** Evidence: user-uploaded `outputs/17_revision_campaign/edgeiiot/original_field_alignment_probe.json`. This is based on at most 256 **first rows per file**, not an exhaustive source census. No data repair, feature policy, model or split decision was made.

## Verified original-CSV field counts and data shapes

- **24 of 24 original CSVs:** 14 attack and 10 normal-device files, each with 63 declared columns. Positional `csv.reader` sampled **256 initial records per file**, **6,144** total.
- **Zero row-field-count mismatches in all 6,144 sampled records**. Each inspected row contains exactly **63 fields**; this removes evidence of *present*, simple extra-field unquoted commas in this sample but **does not establish historical export correctness**, correct value semantics or full-source row widths. A prior lossy or consistently aligned export may still produce exactly 63 fields.
- First-field `frame.time` shape across the 6,144 sampled records:
  - `year_time_fragment`: **5,376 records in 21 files**. Actual user-provided examples show text similar to `2021 17:58:23.150209000`, which lacks both calendar month and day.
  - `ip_address`: **497 records**. DDoS_UDP first 256/256 and normal Modbus first 241/256.
  - `number`: **271 records**. MITM first 256/256 and Modbus first 15/256. The earlier literal inspector showed examples `0.0`, `6.0` for MITM and `0` for Modbus; no plausible 2021–2022 Unix epoch times were established.
- `ip.src_host` and `ip.dst_host` positions are generally IP-shaped for packets with present IP fields; a substantial number of numeric fields in these positions are also consistent with missing/not-applicable protocol placeholders. Do **not** treat every numeric placeholder under an address column as a shifted CSV record.
- `Attack_label` field in the bounded sample is `1` for all fourteen attack-specific files and `0` for all ten normal-specific files. All but OS_Fingerprinting yielded recognized type strings. **OS_Fingerprinting had 256/256 classified as `other` because the diagnostic allowlist omitted this native spelling, not because the source label is invalid**. The code now recognizes both `Fingerprinting` and `OS_Fingerprinting`, with a new synthetic regression test.
- This does **not** verify full native attack counts, class purity beyond the first 256 rows, duplication, address-time identity independence, or model feature leakage in the original CSVs.

## Interpretation

**[Certain]** In the inspected source records the comma-separated fields have normal widths, but `frame.time` values are semantically heterogeneous and frequently missing absolute calendar date components. That supports a **field-semantic/source-transformation validity problem**, not a simple universal delimiter-count error. Which transformation caused it is unknown.

**[Possible, not proven]** Some original timestamp strings might have been truncated or altered by preprocessing; some files could contain source-specific column-mapping problems; the CSV may be a lossy projection of valid PCAP data. The sampled report cannot determine whether original file rows correspond one-to-one to packets. Do **not** repair CSV timestamps by assuming any missing month/day, using filename as an execution-ID, using source-row position as packet identity, or inferring independent sessions from arbitrary gaps. Do not infer that PCAPs are corrupt.

The original Ferrag et al. paper explicitly claims repeated attack experiments at different hours/days, but **no per-execution manifest or defensible row-to-session mapping has yet been verified**. The complete curated ML/DNN datasets were independently found to encode binary labels perfectly through raw text `0.0` in three MQTT fields; that separate leakage finding remains valid and applies to the curated representation, not automatically to original capture rows.

## Next scoped, read-only gate: original PCAP headers

New `scripts/17_probe_edgeiiot_pcap_headers.py` and `tests/test_edgeiiot_pcap_header_probe.py` on revision branch. The PCAP tool:
- checks 24 source PCAP filenames have exactly matching source CSV stems in the attack or normal directory;
- reads each classic PCAP global header and up to **64** packet headers **without reading packet payloads** (seeks over bytes), interpreting microsecond/nanosecond little-/big-endian magic and cautiously presenting plausible packet epoch times in UTC;
- reports per-file header/packet sample status, link type, resolution, first/min/max timestamps of only the initial packet sample, sample span and backward order transitions, without IPs, URLs, payloads, flows or hashed identifiers;
- identifies PCAPNG but **does not** parse PCAPNG timestamp blocks; skips/marks unknown, empty and damaged formats rather than falsely treating them as valid classic PCAPs;
- does **not** establish full capture duration, independent repeated attack runs, CSV-to-PCAP row mapping, or label provenance.

Four synthetic PCAP tests passed in an isolated authoring sandbox (both byte orders/resolutions, PCAPNG refusal to invent timestamps, packet length validation and missing CSV fail). Five original alignment tests have been committed, including the native OS_Fingerprinting spelling regression. **Entire updated user-local pytest has not yet been observed**, including the prior publication-safety diagnostic documentation fix.

Local PowerShell from clean repo root:

```powershell
git status --short
git fetch origin
git merge --ff-only origin/post-comnet-archival-value
uv run python -m pytest -p no:cacheprovider
uv run python scripts/17_probe_edgeiiot_pcap_headers.py --input-root "data/raw/Edge_IIoTset" --out "outputs/17_revision_campaign/edgeiiot/original_pcap_timestamp_probe.json"
```

Stop before PCAP audit on Git/test failure. The PCAP probe runs quickly because it only reads sample packet headers from the already-downloaded 24 PCAP files. Upload `original_pcap_timestamp_probe.json` to assess whether source capture timestamps are usable enough to warrant later full metadata-only sampling and independent repetition verification.

**Protocol decision:** Edge-IIoTset remains UNDER INVESTIGATION, not a third approved primary Protocol B benchmark. No model training, adopted feature exclusions, inferred run partitions, source mutation, or historical release modifications.


## Phase C6 — source-name correspondence bug corrected (2026-10-18)

**Actual user-local failure:** The user ran the original PCAP timestamp script with the expected root, but received `[FAIL] PCAP timestamp probe: Original CSV/PCAP filename correspondences are incomplete` before any packet headers were opened. There is NO new observation about PCAP timestamps, quality, capture repetition, CSV content or dataset eligibility from that failure. The user did not provide latest full-suite pytest output in this step.

**Root cause grounded in the previously uploaded actual source_preflight.json manifest:** All 14 attack CSV and 14 corresponding named attack PCAPs exist, but **13 of the 14 attack PCAP files have descriptive stems different from their source CSV stems**. E.g. DDoS_HTTP_Flood_attack.csv vs DDoS HTTP Flood Attacks.pcap; MITM_attack.csv vs MITM (ARP spoofing + DNS) Attack.pcap. Only Backdoor_attack has a matching attack stem; all 10 normal-device CSV/PCAP stems match. The original probe demanded exact same-stem matches for all 24 named pairs, so it rejected a valid distribution layout. This is a *tool precondition bug*, NOT proof of incomplete author files.

**Repaired revision-branch code:** `scripts/17_probe_edgeiiot_pcap_headers.py` now has a fixed **13-entry explicit attack PCAP-stem-to-CSV-stem map** transcribed from the previously uploaded source manifest. The script uses same-stem matching for the remaining 11 pairs, accepts no fuzzy matches, still fails without generating a report if any source file remains unpaired, and exports each chosen paired CSV relative path and `pairing_basis` (alias or same stem). The map is about publication file naming, not proof of packet-to-CSV observation/label correspondence or independent attack-run records. No timestamp/packet parsing logic, source bytes, model, feature or split policy were changed.

**Regression evidence:** Two tests added to `tests/test_edgeiiot_pcap_header_probe.py`: one creates the entire **14 named original attacks + 10 normal pairs** and checks that 13 documented aliases plus 11 same-stem pairs resolve; another verifies that an unknown attack PCAP name is not silently accepted. In an isolated local sandbox test run, **6/6 focused PCAP tests passed** (four existing plus two new). The 14-attack aliases were additionally compared directly against all actual filenames in the uploaded author source manifest: **14/14 attack mappings match; zero unmatched**. This validation checks exact *filenames*, not packet content. The user's complete repository pytest after this new fix is **PENDING**.

**Local continuation:** Pull the latest `post-comnet-archival-value` branch, run full `uv run python -m pytest -p no:cacheprovider`; only if successful, rerun the same PCAP script command from above (the original run did not create the desired result). Upload `outputs/17_revision_campaign/edgeiiot/original_pcap_timestamp_probe.json` for evidence-based interpretation. No PCAP session or source provenance conclusions have been reached; Edge-IIoTset's primary Protocol B role remains open.
