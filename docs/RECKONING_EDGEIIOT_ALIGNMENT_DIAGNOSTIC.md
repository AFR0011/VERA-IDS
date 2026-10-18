# Edge-IIoTset Phase C4b: raw CSV field alignment and timestamp semantics

2026-10-18. **Status: sample inspected, diagnostic code committed, user-local testing and real-data positional results pending.** No models, splits, feature exclusions or raw-source modifications.

## Literal observations from four actual original source CSVs

- Backdoor_attack.csv frame.time: ' 2021 17:58:23.150209000 ', ' 2021 17:58:23.150231000 ', ' 2021 17:58:23.166306000 '.
- Distance.csv frame.time: ' 2021 23:58:21.314757000 ', ' 2021 23:58:21.314960000 ', ' 2021 23:58:21.315192000 '.
- MITM_attack.csv frame.time: '0.0', '6.0'.
- Modbus.csv frame.time: '0', '192.168.0.128', '192.168.0.1', '192.168.0.101'.

**Certain:** These are not consistently full capture date/time values under the declared frame.time field. Year plus time lacks month/day; small numeric values do not look like the reported 2021-22 capture Unix epoch; IPv4 addresses are not timestamps. This does NOT prove PCAP data are bad or the repeated attacks did not occur.

**Possible, NOT verified:** unquoted comma in timestamp, original export/CSV parse error, shifted column mapping, earlier lossy preprocessing, or separate extract issues. A DictReader key showing an IP is compatible with various failure modes. Do not choose a parser, reconstruct missing dates or invent sessions until raw positional evidence is available.

## New read-only field alignment probe

Script: scripts/17_probe_edgeiiot_alignment.py
Tests: tests/test_edgeiiot_alignment_probe.py (four synthetic test cases committed, user local full pytest PENDING).

Streams only the FIRST 256 rows of each original attack/normal CSV through csv.reader (positional), NOT the two curated CSVs. Counts observed fields per row versus header count, shape categories of the first eight positional values (date_time_text, year_time_fragment, month_day_fragment, ip_address, time_only, number, empty, other_text), first-four-value shape patterns, and allowlisted Attack_label / Attack_type positions. Output contains no raw values, IPs, raw timestamps, payloads, or rows. Default expects 14 attack +10 normal original CSVs and refuses to write a new report if missing.

Interpretation: field count greater than header count can indicate an unquoted delimiter, but cannot independently prove it; equal widths also cannot rule out prior lossy conversion or swapped data. Matching year/time and month/day fragments across adjacent positional fields could motivate a separate source parsing investigation. It does not recover original timestamps or justify independent attack-run groups.

## Local PowerShell from repository root

    git status --short
    git switch post-comnet-archival-value
    git fetch origin
    git merge --ff-only origin/post-comnet-archival-value
    uv run python -m pytest -p no:cacheprovider
    uv run python scripts/17_probe_edgeiiot_alignment.py --input-root "data/raw/Edge_IIoTset" --out "outputs/17_revision_campaign/edgeiiot/original_field_alignment_probe.json"

Stop if Git cannot fast-forward or pytest fails. Upload original_field_alignment_probe.json. Full original-capture time/label support and PCAP-derived absolute-time verification are later gated steps subject to user methodological approval.

**Scientific scope:** repeated attacks are documented in original paper but cannot yet be partitioned as source-independent executions. Edge-IIoTset remains an unapproved third-primary Protocol B candidate. The historical COMNET release is immutable.


## User-local execution checkpoint — 2026-10-18

The user fast-forwarded local revision branch from d09c3f7 to a3a6232, then executed `uv run python -m pytest -p no:cacheprovider`: **54 collected, 53 passed, 1 failed**. All four `tests/test_edgeiiot_alignment_probe.py` tests passed; the sole failing `tests/test_release_contents.py::test_public_text_has_no_absolute_local_paths` flagged the new `scripts/17_probe_edgeiiot_alignment.py`.

**Root cause of the original script failure:** the publication-safety checker misread consecutive regex digit-class tokens adjacent to a colon as a Windows drive-root pattern. The script was corrected with equivalent fixed-width digit quantifiers in commit `00b844e41b9b63334fa9089f94bb58e0341dad86`. No real private path was exposed, and the safety test was not changed.

The user separately ran `uv run python scripts/17_probe_edgeiiot_alignment.py --input-root "data/raw/Edge_IIoTset" --out "outputs/17_revision_campaign/edgeiiot/original_field_alignment_probe.json"`. It printed 24/24 source CSVs and `[PASS] Sampled 6,144 positional rows`. **The actual JSON report has NOT yet been uploaded**. Consequently sample field-width mismatches, first-field shape patterns, and possible delimiter shifts are **unknown**; do not treat `[PASS]` as CSV correctness or independent-run support.

**Next required evidence:** user pulls revision commit `00b844e`, reruns full pytest (expected 54/54, pending), and uploads the already-created `outputs/17_revision_campaign/edgeiiot/original_field_alignment_probe.json`. Do not run another complete scan merely because tests failed. After reviewing JSON, decide the next read-only source/PCAP corroboration gate. No source repair, timestamp reconstruction, train/val/test splitting, model fitting, dataset admission or historical release edits were authorized.


## Second local verification checkpoint — 2026-10-18

The user updated local revision branch to commit `6310327` and reran the complete test suite: **54 collected, 53 passed, one release-content check failed**. All four alignment tests and all other tests passed. The offending file is now **this documentation file** because its previous explanation reproduced a sequence that the checker recognized as an absolute-path pattern. This is documentation-only, and no private source path was present. We have rewritten that explanation without reproducing suspicious text. The release test remains unchanged.

The **local second test output is verified**; the test result on the documentation-only fix is still **PENDING**. The original 24-file alignment scan has already completed with PASS, but its JSON report still has not been provided for interpretation. User should fast-forward, rerun pytest, and upload existing `outputs/17_revision_campaign/edgeiiot/original_field_alignment_probe.json` without rescanning.
