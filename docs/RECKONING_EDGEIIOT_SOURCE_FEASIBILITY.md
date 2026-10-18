# Edge-IIoTset feasibility review: original provenance and known leakage (2026-10-18)

**Status: first-pass published-source feasibility review; local acquisition and row-level verification NOT YET performed; dataset role NOT approved.**

Purpose: screen Edge-IIoTset as a possible third independent primary evaluation environment in the post-COMNET VERA-IDS archival-value campaign. This is a gate, not a protocol change, and does not supersede the decision to retain CICIDS2017 and CICIoT2023 as the already-established primary sources.

## Authoritative starting points

- Original Ferrag et al. IEEE Access paper (2022): **Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning**, DOI https://doi.org/10.1109/ACCESS.2022.3165809.
- Authors' Kaggle distribution: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot (data explorer indicates v5, ~11.25 GB uncompressed distribution, subject to version changes).
- Original IEEE DataPort dataset: https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications (site could not be independently navigated by the agent; use author Kaggle).
- New, very relevant 2026 **preprint** (not verified as peer reviewed): M. M. Galal, *Provenance, Not Behaviour: A Serialisation Artifact in Edge-IIoTset and a Leakage-Free Benchmark for Precision-Agriculture Intrusion Detection*, 2026-08-16, https://arxiv.org/abs/2608.15761 . The preprint supplies replicable column and token diagnostics, but is NOT automatically accepted as factual for our source version until locally reproduced.
- Independent warning/code example: https://github.com/fdhaou-prog/edge-iiotset-leakage identifies related protocol-field shortcuts in the ML curated subset; verify source methodology rather than take published performance at face value.

## What the original dataset describes (not a verified local inventory)

- Purpose-built IoT/IIoT testbed with real sensors, edge/network components, 14 named attacks in five broader threat categories.
- Original distribution is separated into **10 normal-device CSV/PCAP capture pairs**, **14 attack-specific CSV/PCAP pairs** and two already-curated selected CSVs for classical ML vs DNN. Some secondary listings incorrectly describe the 14 attack files as 13; the original paper enumerates 14 actual names. Names from paper include `Backdoor_attack`, `DDoS_HTTP_Flood_attack`, `DDoS_ICMP_Flood_attack`, `DDoS_TCP_SYN_Flood_attack`, `DDoS_UDP_Flood_attack`, `MITM_attack`, `OS_Fingerprinting_attack`, `Password_attack`, `Port_Scanning_attack`, `Ransomware_attack`, `SQL_injection_attack`, `Uploading_attack`, `Vulnerability_scanner_attack`, and `XSS_attack`.
- Curated `ML-EdgeIIoT-dataset.csv` is reported as **157,800 x 63**, and `DNN-EdgeIIoT-dataset.csv` as **2,219,201 x 63** (61 extracted features plus `Attack_label` and `Attack_type`). The distributions and preprocessing of these subsets are NOT equivalent to original source capture sessions. Their published record totals are not yet independently verified.
- The original paper's evaluation recipe explicitly merges source CSVs, removes some identifier/payload features and uses **random train/test splits**. Distinct original files are useful source provenance, but *one file per attack class* does not demonstrate **repeated independent attacks per family**. Neither file numbering nor disjoint train/test rows can certify LOAO source/time independence.

## Specific label-leakage risk to audit BEFORE preprocessing or model training

The 2026 Galal preprint reports a **serialization artifact** in both ML and DNN curated subsets: normal-source placeholder missing-protocol value written as the **string `"0"`**, attack-source as **`"0.0"`**. It reports that any one of these four columns `dns.qry.name.len`, `mqtt.conack.flags`, `mqtt.protoname`, `mqtt.topic` perfectly separates binary labels in each curated dataset. The seven columns in the original dummy-encoding recipe also include `http.request.method`, `http.referer` and `http.request.version`, with partial shortcuts reported. **These are findings reported by a preprint, NOT our verified experiments yet.**

Critical implementation rule: **read such CSV fields as text and count exact `"0"` vs `"0.0"` by `Attack_label` before pandas/sklearn inference**. A numeric conversion erases the distinction and can hide the source of apparent perfect separability. The dataset author originally recommended dropping IP/port/timestamps/payload. That does NOT neutralize protocol missing-value serialization shortcuts. A later corrected pipeline or raw-PCAP re-extraction may be necessary if choosing to use this dataset, and must be declared before test evaluation.

The preprint's AgriEdge construction uses original per-device captures and uniform parsing, but its leave-one-device-out evaluation concerns **sensor transfer with some attack traffic**, not the same scientific question as VERA's attack-family LOAO. It is not independently verified proof that Edge-IIoTset meets our Protocol B support/independence gates.

## Initial judgment and unresolved gates

**Initial judgment: HIGH RISK as the third full primary Protocol B dataset; not automatically rejected.** Its original, device-attributed raw captures may support a valuable alternative closed-set/device-shift stress lane, but the curated files are **not acceptable as-is** for a leakage-free generalization claim without reconstructing/testing their generation pipeline.

- **Source provenance:** Do raw attack captures contain independent sessions/executions **of the same family**, accompanied by an authoritative manifest? A single attack-specific PCAP per class is insufficient.
- **Support:** Are all 14 native attack labels present with appropriate per-family training/validation/test support under source-admissible splits and VERA's unchanged support rules?
- **Leakage:** Can the exact `0` / `0.0` label proxy be reproduced in the author's currently distributed curated files? Inspect file-specific raw CSV header/dtypes, timestamps, IPs and duplicate rows; only after that propose and approve exclusions/canonicalization.
- **Task definition:** 14-native-family Stage 2 / LOAO may not be comparable with dataset's broader five threat categories. Do not silently collapse families. Stage 1 binary vs attack must not use `Attack_label` or `Attack_type` as model inputs.
- **Independent unit:** Within one synthetic testbed and per-attack captures, independence across *families* is not the same as independent *repetitions* and capture-condition robustness. Avoid claiming actual episodes from filenames.

## Read-only preflight implementation

Committed to revision branch only:
- `scripts/17_preflight_edgeiiot_source.py`: recursive original file inventory (relative names, bytes, CSV headers, candidate identity/time field names, PCAP presence), optional full curated CSV textual counts of `"0"` vs `"0.0"` in four affected fields by `Attack_label` and native attack counts. Exports no raw IPs, URLs, payloads or rows. Does not detect all possible token aliases or run a classifier. **It is a first-stage artifact check, not causal leakage proof.**
- `tests/test_edgeiiot_source_preflight.py`: three synthetic tests passed in sandbox using the validated local script; entire repo pytest and real Edge-IIoTset dataset scan PENDING.

Recommended **minimal** source acquisition (if user agrees to proceed): author-origin `Readme.txt`, `ML-EdgeIIoT-dataset.csv` (the smaller curated subset for quick textual leakage check), and **original CSVs from both** `Attack traffic` and `Normal traffic`. `DNN-EdgeIIoT-dataset.csv` and original PCAP files can wait until preflight indicates whether a full reconstruction is scientifically justified. Keep original nested directory names and never commit the data into Git.

From user's local VERA-IDS root, after putting originals under `data/raw/Edge_IIoTset/`:

```powershell
git fetch origin
git merge --ff-only origin/post-comnet-archival-value
uv run python -m pytest -p no:cacheprovider
uv run python scripts/17_preflight_edgeiiot_source.py --input-root "data/raw/Edge_IIoTset" --out "outputs/17_revision_campaign/edgeiiot/source_preflight.json" --audit-curated
```

If only raw original CSVs have been downloaded, the script still inventories them and returns empty `curated_audits`; no token-leakage conclusion is possible until curated subset exists. If raw files are not yet downloaded, do not fabricate outcomes from the existing publication. This step imports no trained model and changes no existing VERA data split.

**Decision remains with user**: continue author-source acquisition and local audit despite known risks; or prioritize a different source with stronger independent attack execution provenance if this is intended to be a full third primary Protocol B condition.
