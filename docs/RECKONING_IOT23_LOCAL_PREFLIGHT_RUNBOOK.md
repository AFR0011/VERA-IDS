# IoT-23 v1 — Tomorrow's bounded preflight runbook

**Prepared:** 2026-10-22. **Run on:** Friday, 2026-10-23 (local Windows PowerShell).  
**Status:** Code/test files committed on `post-comnet-archival-value`; **full user-local pytest and actual IoT-23 archive scan NOT yet executed**. Scientific admission and Protocol B group rules remain OPEN.  
**Dataset:** Original IoT-23 v1.0.0, 20 CTU-IoT-Malware captures, 3 CTU-Honeypot captures, `bro/conn.log.labeled`, README and original PCAPs. Zenodo DOI 10.5281/zenodo.4743746. The full public archive is approximately 21.5 GB compressed.

## Primary source and use conditions

Official dataset page: https://www.stratosphereips.org/datasets-iot23  
Official Zenodo record: https://zenodo.org/records/4743746  
Official publisher FAQ: https://www.stratosphereips.org/datasets-faq  
Author scenario README example: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/

The publisher's current FAQ explicitly says dataset permission **need not be requested provided the dataset is appropriately cited**. An older individual scenario README says authorization is required. These statements conflict; the source/FAQ distinction is recorded. For a publication relying on the dataset, author contact can further clarify provenance/reuse if desired. Do not claim an independent license grant beyond the actual official statements. Zenodo lists v1.0.0, and **v2 is a distinct version** with revised labels and structure; do not combine the two.

**Never execute binary files included within the archive.** No need to unpack packet captures or malware binaries for this first gate.

## Files now available on the revision branch

- `scripts/17_extract_iot23_flow_sources.py`: stream a full `.tar.gz`, extract only scenario `README.md` and `bro/conn.log.labeled`; no PCAP/binary extraction, no overwrite on rerun, safe path restrictions and audit manifest. Reads the compressed archive sequentially and copies roughly the original flow-log corpus, which may still require several GB of free disk.
- `scripts/17_audit_iot23_capture_support.py`: discover all CTU scenario directories under any nested extracted root; check 20 malware + 3 benign scenario structure, README presence, native Zeek columns, tab/space label quirk, raw exact `label` and `detailed-label` strings, record widths, initial timestamp plausibility, **capture-by-label observed support**, and aggregate-only JSON. `preflight` samples first 512 flow rows per source. `census` streams all native Zeek rows per source only on explicit command and returns exact capture-level counts.
- `tests/test_iot23_flow_extraction.py`: 4 synthetic test cases (selected extraction, existing file refusal, parent traversal rejection, incomplete manifest).
- `tests/test_iot23_capture_support_audit.py`: 5 synthetic test cases (native mixed delimiters, gzip/full census, partial malformed rows, invalid capture dir, missing native labels).

**Scientific restrictions:** Neither script maps detailed labels into attack families, drops model features, repairs data, creates independent experiment IDs, constructs training/validation/test partitions, or fits models. 512-row preflight label counts are **PREFIX OBSERVATIONS, not complete support or class prevalence**. Publisher capture names are candidate provenance units only; independent execution must be checked separately. No support thresholds are changed to admit a candidate. The final version, benign source support, leakage and true class semantics must all pass before primary dataset adoption.

## 0. Get revision branch and run the test gate (from existing VERA-IDS repo root)

```powershell
git status --short
git fetch origin
git merge --ff-only origin/post-comnet-archival-value
uv run python -m pytest -p no:cacheprovider
```

If `git status --short` shows unexpected local modifications, or git cannot fast-forward, STOP and resolve safely rather than forcing checkout/reset. If pytest fails, STOP and share the traceback. Prior GitHub file readback checks succeeded, but a successful updated full user-local pytest has not yet been observed.

## 1. Locate the archive and inspect disk requirements

Assumes the official complete archive is downloaded into the user's Windows Downloads folder. Edit **only this PowerShell variable** to point to its real location.

```powershell
$archive = Join-Path $HOME "Downloads/iot_23_datasets_full.tar.gz"
$sourceRoot = "data/raw/IoT_23"
$reportRoot = "outputs/17_revision_campaign/iot23"
Get-Item "$archive" | Select-Object Name,Length,LastWriteTime
Get-FileHash "$archive" -Algorithm MD5
New-Item -ItemType Directory -Force "$reportRoot" | Out-Null
```

**Checksum note:** Zenodo v1.0.0 currently lists MD5 `7132e603f9750b8580b6cebdbcd43e9c` for its full tarball. If the user fetched the CTU mirror and its bytes differ, record that difference instead of assuming the file is altered, malicious or automatically acceptable. The archive may be ~21 GB; hash computation reads the whole compressed file and can take time.

If the archive is still downloading or insufficient disk space is available, do not start extraction. Verify that the dataset is v1 and fully downloaded before this step.

## 2. Extract ONLY the native labeled logs and README files

**Recommended for full archive users who have not already extracted the source files.** This avoids extracting malware binaries and PCAPs. It still must decompress/traverse the complete archive and write the large Zeek flow logs.

```powershell
uv run python scripts/17_extract_iot23_flow_sources.py --archive "$archive" --output-root "$sourceRoot" --out "$reportRoot/flow_source_extraction_manifest.json"
```

Expect exactly **23 original capture scenarios**, **23 native flow logs** and **23 capture READMEs**, i.e. 46 selected files. Output path `flow_source_extraction_manifest.json`. The script intentionally fails rather than overwrites selected files on a second run. Do NOT retry on an already partially extracted tree; first inspect the error and output locations.

**If the dataset has already been fully extracted elsewhere:** skip step 2, point `$sourceRoot` to that extracted directory instead. The scanner supports the native archive's deeply nested `opt/Malware-Project/BigDataset/IoTScenarios/` structure and does NOT need relocated source files. Do not copy or flatten individual captures.

## 3. First native Zeek preflight: 512 records × 23 captures

```powershell
uv run python scripts/17_audit_iot23_capture_support.py --input-root "$sourceRoot" --phase preflight --sample-rows-per-capture 512 --out "$reportRoot/native_capture_preflight.json"
```

Expected maximum of **11,776 first-flow rows sampled** (512 × 23) if every file has at least 512. Report gives source capture inventory, exact native label combinations, schema consistency, record-width anomalies and coarse timestamp ranges of the sampled prefixes. It excludes all raw IP addresses, flow IDs and packet payloads.

**Upload exactly these aggregate reports** for review:
- `flow_source_extraction_manifest.json` (if step 2 was used).
- `native_capture_preflight.json`.

Also paste the final console summary and `pytest` totals. **STOP HERE for the initial day**. The prefix samples cannot decide Protocol B eligibility; inspect them and agree on the next specific gate before running any heavy analysis.

## 4. Full native-label census: prepared, but DO NOT run automatically

Only if the full prefix preflight validates original schema/labels and the user explicitly approves this next costly operation:

```powershell
uv run python scripts/17_audit_iot23_capture_support.py --input-root "$sourceRoot" --phase census --progress-every 2000000 --out "$reportRoot/native_capture_full_census.json"
```

The publisher cites roughly **325 million labeled Zeek flows**; this can be a long CPU/storage pass. It never reads PCAPs, does not train models and preserves native labels. It enables an actual capture-by-exact-label count table and UTC-day occupancy, not independent-execution proof. No row caps or unapproved label aggregation in the whole-source support accounting.

## Gate after full census: final primary admission decision

Review exact **malicious behavior label** semantics (e.g. DDoS, `PartOfAHorizontalPortScan`, C&C), the distinction from native malware-name labels and compound categories, clean/mixed benign support in multiple capture groups, file/run independence, per-family training/validation/test group support, source-rule-derived shortcut risks, and compute feasibility.

**Pass:** write a proposed explicit capture group and taxonomy plan, run existing support audit with unchanged thresholds, and ask the user to approve scientific method before ML work.  
**Fail:** close the final third-primary search and resume the main manuscript strengthening campaign (five-seed paired visible/blind, repeated CICIDS2017 partitions, at least two genuine literature comparators and machine-readable stability evidence). No fourth candidate search. Any relaxation of the originally mandatory third primary dataset requirement needs its own explicit user decision.

**Historical COMNET release:** `release/v2026.10-comnet/` stays untouched; PR #3 remains DRAFT, unmerged. Local raw data and JSON reports remain under ignored source/output trees. No upload of original full source logs, PCAPs or malware artifacts.
