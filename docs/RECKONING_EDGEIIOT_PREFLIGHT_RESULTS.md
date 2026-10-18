# Edge-IIoTset original-source preflight — verified local results (2026-10-18)

**Phase C3:** The user uploaded the real read-only JSON at `outputs/17_revision_campaign/edgeiiot/source_preflight.json`. That report was read and independently checked against its 52 manifest entries and both curated source audits. **Original source data are already present locally.** No source bytes, raw network addresses, payloads or model predictions were uploaded. No dataset has been admitted to Protocol B.

## Manifest integrity

| Category | Files found |
|---|---:|
| Original attack CSVs | 14 |
| Original attack PCAPs | 14 |
| Original benign device CSVs | 10 |
| Original benign device PCAPs | 10 |
| Curated ML and DNN CSVs | 2 |
| Readme.txt and TechRxiv PDF | 2 |
| **All inventoried files** | **52** |

All **26 CSV files** have **63 columns and no duplicate header names**. The 24 original attack/device CSVs include `frame.time`, `ip.src_host`, `ip.dst_host`, `Attack_label`, and `Attack_type`. The actual Heart_Rate original CSV and PCAP are present, confirming that the README's repeated Flame_Sensor text was a documentation typo. All originally listed attack source files are present, contrary to the README's erroneous count of 13. There are **50 dataset files excluding the two documentation files**, versus the original TechRxiv paper's stated total of 49. Listing/sizes only: **no original CSV row counts, type distributions, timestamp or independent run support were yet checked.**

The locally inventoried files sum to approximately 11.25 billion bytes including documentation, but the report does not certify full downloads via checksums or independently authenticate the publisher version.

## Complete curated dataset census

| Curated subset | Rows scanned | Benign | Attack | Width errors | Attack types |
|---|---:|---:|---:|---:|---:|
| DNN-EdgeIIoT-dataset.csv | 2,219,201 | 1,615,643 | 603,558 | 0 | 14 |
| ML-EdgeIIoT-dataset.csv | 157,800 | 24,301 | 133,499 | 0 | 14 |

Both have 15 labels including Normal, and the sums of all listed `Attack_type` counts equal their total row counts. Those counts describe **curated selected subsets**, not original attack/benign capture totals or independent LOAO partition support. Notably both curated sources show `Fingerprinting` = 1,001 and `MITM` = 1,214. The original Table XI printed Total vs Train+Test inconsistency therefore must not be used to infer support.

## Independently reproduced exact-text label shortcut

The pilot counted verbatim CSV strings `0` and `0.0` **before numeric conversion** for all four protocol columns flagged by the Aug 2026 Galal preprint.

For each of `mqtt.conack.flags`, `mqtt.protoname`, and `mqtt.topic` in **BOTH** curated datasets:
- **Every attack row** has the literal text `"0.0"` (DNN: 603,558/603,558; ML: 133,499/133,499).
- **No normal row** has the literal text `"0.0"` (DNN: 0/1,615,643; ML: 0/24,301).
- Therefore the deterministic, untrained rule `Attack_label_hat = 1 if raw_token == "0.0" else 0` correctly assigns **every** benign and attack label in both entire curated datasets. This is *logically implied by the full-text census*, not an estimated metric from a model.
- The other flagged column, `dns.qry.name.len`, has attack-side `"0.0"` in DNN **603,331/603,558** and ML **133,272/133,499**, so do NOT describe its own literal exact-zero rule as universally perfect based on current evidence.

This is independent **local reproduction of the textual source artifact** reported in the 2026 preprint, not a new exclusive scientific discovery. It is also not direct evidence about the original raw 24 source CSVs, causal use by any trained model, or independently valid attack labels. Numeric `0` versus `0.0` are numerically equal; coercion could hide the serialisation distinction. **The curated versions in their original textual categorical form must NOT be accepted as leakage-free binary datasets.** Other shortcuts and collection/capture confounding remain untested even after excluding this one.

## Immediate next audit: original timestamps, then full native rows

A small sample-only format probe has been added in `scripts/17_probe_edgeiiot_timestamps.py` with three synthetic tests in `tests/test_edgeiiot_timestamp_probe.py` (all three passed in isolated authoring sandbox; entire repository user-local pytest still unverified).

- Reads **at most 256 first rows** from each of the 24 original CSVs. Skips the 2 curated CSVs. Only writes counts of recognized timestamp display forms, calendar dates *as displayed*, and sample binary/type labels. It exports no raw timestamp strings, IPs or payloads.
- It is intended **only to learn `frame.time` representation**, whether all 24 sources can be consistently interpreted and whether a complete source-wide temporal audit is feasible. The first 256 rows are NOT representative evidence about the original attack period, number of independent attacks or per-class support.
- Do not change inferred timezone to UTC without checking actual textual zone, and do not derive run IDs merely from dates or timestamp gaps. A later full scan and possibly PCAP-level attack-operation corroboration will be necessary.
- After the sample JSON is reviewed, propose a full **read-only** per-original-file record/type/date/timestamp/duplicate-leakage audit subject to user's process approval. No trainer/preprocessor modifications permitted.

User-local PowerShell from VERA-IDS root:

```powershell
git status --short
git switch post-comnet-archival-value
git fetch origin
git merge --ff-only origin/post-comnet-archival-value
uv run python -m pytest -p no:cacheprovider
uv run python scripts/17_probe_edgeiiot_timestamps.py --input-root "data/raw/Edge_IIoTset" --out "outputs/17_revision_campaign/edgeiiot/original_timestamp_probe.json"
```

If git or pytest fails, STOP before proceeding. Then upload `original_timestamp_probe.json`. No need for additional downloads: all original CSVs and paired PCAPs are already locally present.

**Phase status:** C3 preflight and curated token audit complete; C4 timestamp probe implemented, local execution pending; independent session IDs, original label support, leakage-free preprocessing, accepted splits, and third primary dataset role **OPEN**. No changes to old COMNET release or frozen campaign specifications.
