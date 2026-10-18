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

**Updated judgment (see source correction below): HIGH RISK for leakage but now more promising for independent-run provenance; third full primary Protocol B suitability NOT verified.** Its original, device-attributed raw captures may support a valuable alternative closed-set/device-shift stress lane, but the curated files are **not acceptable as-is** for a leakage-free generalization claim without reconstructing/testing their generation pipeline.

- **Source provenance:** Original TechRxiv paper explicitly reports repeated experiments at different times (see correction below); can we *recover and verify distinct execution groups* within attack CSV/PCAP source using dates or an authoritative manifest? A single attack-specific file still cannot be treated as a single independent run.
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

## Primary-source correction and stronger provenance evidence — 2026-10-18

The user supplied the actual author README (last updated 18 March 2022, pasted in conversation) and the complete 25-page original TechRxiv preprint PDF `Edge_IIoTset__DatasetFL.pdf`. **These materials supersede the earlier provisional assertion that independent repetitions were not documented.** On *PDF page 12 (printed p. 11)*, the authors explicitly state that the attack experiments occurred on **different hours and days between 21 November 2021 and 10 January 2022**, and that **each attack experiment was conducted multiple times**. They describe normal traffic as collected noncontinuously over the same period and individual PCAP capture periods up to three hours. This is **documented repeated-execution provenance** and warrants pursuing the original-source audit. It does **not** demonstrate how to map original rows to distinct repeat executions: the public CSV/PCAP directory manifest enumerates attack *types*, not individual dated runs. Before Protocol B independent-session split acceptance, demand actual extraction of defensible non-overlapping run/capture groups and support across train/validation/test. Never relabel file names or arbitrary time gaps as authenticated sessions.

Primary source: Ferrag, Friha, Hamouda, Maglaras and Janicke, TechRxiv original preprint (2022), DOI **10.36227/techrxiv.18857336.v1**, https://doi.org/10.36227/techrxiv.18857336.v1. The uploaded PDF's cover declares **CC BY 4.0 for the preprint**. The separately supplied dataset author README grants perpetual **academic research use**, while commercial use requests author permission. These two statements apply to different resources; do not mistakenly claim the dataset itself is licensed CC BY solely because the manuscript is. Original manuscript also appeared in IEEE Access (DOI 10.1109/ACCESS.2022.3165809); retain separate citations without assuming identical publication pagination.

### Source-paper bookkeeping discrepancies (do not repair by guessing)

- README: says **13** attack CSV and PCAP documents but enumerates **14 filenames** of each. PDF section IV-C (PDF pages 16–17) explicitly specifies **14 CSV + 14 PCAP** attack source files, confirming 14 native attack types. The README normal Heart_Rate item mistakenly repeats the Flame_Sensor filename and sensor description. Use actual local source file contents, not this typo, for inventory and labeling.
- PDF says **49 files** overall, while the described inventory implies **10 normal CSV/PCAP pairs (20 files) + 14 attack CSV/PCAP pairs (28) + 2 curated CSV = 50 files**. We cannot determine whether the 49 is a typo, an omitted file, or a different counting convention without a complete distribution manifest.
- PDF **Table X**, PDF page 20: summing its **14 individual attack rows gives 9,729,709**, whereas the printed attack subtotal is **9,728,708**; discrepancy **1,001**, equal to the fingerprinting class. Consequently the printed grand total **20,952,648** is also 1,001 below the sum of all class rows (**20,953,649**). These numbers are documentary claims, **NOT** observed dataset rows.
- PDF **Table XI**, PDF page 20: numerous printed class-level Total values do not equal the displayed Train + Test counts. E.g. normal **24,301 != 19,281 + 4,820 = 24,101**, MITM **1,214 != 286 + 72 = 358**, and fingerprinting **1,001 != 682 + 171 = 853**. Table XI therefore cannot be accepted as a validated support/split map. The PDF mentions SMOTE for fingerprinting and MITM but does not resolve this accounting.
- Original paper says attack experiments are repeated, but does not report a reliable table of **number of independent attack operations or execution-group IDs per native attack class**, nor a source-run-to-CSV mapping. It describes **random** train/test splitting, not source-separated evaluation. Thus the existence of repeats is substantially more promising than previously thought but does not yet meet Protocol B admissibility.

**Revised assessment:** Edge-IIoTset is a **higher-priority source-acquisition and independent-run-identification candidate** than the preliminary inventory suggested, but is **still not approved as a third primary Protocol B source**. The 2026 serialization-label artifact risk remains independently serious for the *curated* ML/DNN sources and should be checked locally before any numerical coercion. Prioritize original attack and normal CSV timestamp columns (`frame.time`), README and the smaller curated ML CSV. Preserve PCAPs as a potential later independent corroboration of run boundaries; avoid downloading/reprocessing all PCAPs before the inexpensive metadata gate.


## Phase C3 — verified complete local source manifest and label shortcut (2026-10-18)

User has **already downloaded** the original distribution. Their read-only local preflight `source_preflight.json` was uploaded and inspected independently. Any earlier instruction to download the same source files is obsolete.

**Manifest, from actual user-local source directories:**
- 52 files inventoried overall: **14 original attack CSVs + 14 attack PCAPs**, **10 normal CSVs + 10 normal PCAPs**, **2 curated CSVs** (ML and DNN), and **2 docs** (Readme.txt and TechRxiv PDF). Thus 26 CSV + 24 PCAP + 2 docs; 50 actual dataset files, *compared with 49 asserted in original paper*. The author README's stated '13' attacks is only a prose/count inconsistency: all 14 named attack files are actually present.
- All 26 CSVs have the complete identical **63-column schema** and no duplicate header names in the preflight. All original CSVs retain `frame.time`, source/destination host fields, `Attack_label`, and `Attack_type`. CSV headers and file sizes were inventoried only; **original 24 CSV row counts, timestamp syntax, dates, time gaps, actual attack execution groups, raw IP distributions and label coherence have NOT YET BEEN SCANNED**.
- Curated **DNN** file: 2,219,201 rows (zero width errors), 1,615,643 `Attack_label=0` and 603,558 `Attack_label=1`; 15 types including Normal. Curated **ML** file: 157,800 rows (zero width errors), 24,301 benign and 133,499 attacks; 15 types.
- Every curated attack row in both datasets contains raw text **`"0.0"`** in each of `mqtt.conack.flags`, `mqtt.protoname`, and `mqtt.topic`. **No benign row contains the raw text `"0.0"` in any of these three fields**. Thus the literal-text rule `Attack_label_hat = 1 iff mqtt.conack.flags == "0.0" else 0` has **exact 100% accuracy for all 2,219,201 DNN and 157,800 ML rows**, established arithmetically from the exhaustive token census, without model training. The same holds for `mqtt.protoname` and `mqtt.topic`.
- `dns.qry.name.len` shows strong but **non-exhaustive** class-paired literal zero spellings (DNN 603,331 of 603,558 attack rows as `"0.0"`; ML 133,272 of 133,499). Do not assert it is itself universally perfect on the basis of only these token counts. Counts were recorded before numeric coercion; numerical `0.0` vs `0` conversion can erase the witness to this specific text artifact.
- This **independently confirms the critical label shortcut** reported in Galal's 2026 preprint in the actual curated version at hand. It is *internal, not independent ground-truth validity*. We have NOT verified model use of this feature or its effects, and **not** audited the original capture CSVs for the same artifact. Do NOT feed these literal-encoded columns into VERA-IDS classification and declare it a legitimate benchmark.
- Local preflight finished; user's full updated project pytest status has NOT been provided in this turn. No data, features, splits or model outputs were changed.

**Next bounded read-only audit:** initial `frame.time` format/sample inspection in each of the 24 **original** CSVs, not the curated two. The goal is to verify timestamp syntax and usable provenance before a full raw-CSV temporal/label census. Author paper reports repeated attacks at different times, but sample time records alone cannot prove execution-level independence. Original 24 paired PCAPs are already on user's local disk if later packet-level corroboration becomes necessary.

**Dataset role remains OPEN.** A complete inventory is encouraging, while a perfectly predicting text artifact is a severe obstacle for the curated versions. No automatic Protocol B approval, raw file modifications, feature policy or split plan.


## Final status supersession — 2026-10-18

**CLOSED / NO-GO as a third primary Protocol B dataset, by explicit user decision (Option A).** Early sections of this document were written before the C3–C10 local source, leakage, PCAP chronology and integrity audits and remain as historical progress notes. Do **not** interpret earlier "role open" or "next audit" text as an active task. No further Edge experiments or source investigation have been authorized. Retain only a *potential*, not approved, separate external-stress role.

Authoritative closeout and evidence chain: `docs/RECKONING_EDGEIIOT_PRIMARY_CANDIDACY_CLOSEOUT.md`. The CICIDS2017/CICIoT2023 revision plan and separate TON_IoT decision are not automatically changed.
