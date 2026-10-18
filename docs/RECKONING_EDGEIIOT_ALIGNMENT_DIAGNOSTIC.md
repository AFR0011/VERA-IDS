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

**Root cause independently verified against the unchanged release-test regex:** two legitimate script regex literals included written sequences `\\d\\d:\\d\\d` that the test interpreted as `d:\\` (Windows absolute path). This is **source-code text false positive**, NOT a leaked private filesystem path. Two patterns were replaced with regex-equivalent `\\d{2}:\\d{2}` forms in script only, commit `00b844e41b9b63334fa9089f94bb58e0341dad86`. Verified with the same no-absolute-path predicate that there are no remaining matching spans. Representative date/time string matches remain equivalent. The safety test was NOT weakened. **The corrected full suite has NOT yet been run on the user's machine.**

The user separately ran `uv run python scripts/17_probe_edgeiiot_alignment.py --input-root "data/raw/Edge_IIoTset" --out "outputs/17_revision_campaign/edgeiiot/original_field_alignment_probe.json"`. It printed 24/24 source CSVs and `[PASS] Sampled 6,144 positional rows`. **The actual JSON report has NOT yet been uploaded**. Consequently sample field-width mismatches, first-field shape patterns, and possible delimiter shifts are **unknown**; do not treat `[PASS]` as CSV correctness or independent-run support.

**Next required evidence:** user pulls revision commit `00b844e`, reruns full pytest (expected 54/54, pending), and uploads the already-created `outputs/17_revision_campaign/edgeiiot/original_field_alignment_probe.json`. Do not run another complete scan merely because tests failed. After reviewing JSON, decide the next read-only source/PCAP corroboration gate. No source repair, timestamp reconstruction, train/val/test splitting, model fitting, dataset admission or historical release edits were authorized.
