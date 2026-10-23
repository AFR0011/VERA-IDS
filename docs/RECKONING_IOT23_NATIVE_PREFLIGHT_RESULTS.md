# IoT-23 v1 source preflight — actual local result and source-layout caveat

**Evidence received:** 2026-10-23, user-uploaded `outputs/17_revision_campaign/iot23/native_capture_preflight.json`, produced by `scripts/17_audit_iot23_capture_support.py` on locally extracted original archive under `data/raw/IoT23`. All values below independently recalculated from the 23 per-capture entries. **This is a first-512-row *prefix* sample per native labeled flow file, NOT a full census, partition or model evaluation.**

## Aggregate checks

- Schema `iot23-native-source-label-audit-v1`; `mode=preflight`, `rows_per_capture_limit=512`, `complete_source_row_census=false`.
- **23 distinct source scenario IDs**, exactly **20 malware** and **3 benign-device**; 23 published native `bro/conn.log.labeled` Zeek logs located, one in an additional nested `Somfy-01` subdirectory.
- **11,059** source records sampled (maximum 11,776; three source logs have fewer than 512 rows: benign Honeypot 4-1 **452**, Honeypot 7-1 Somfy-01 **130**, malware 44-1 **237**). Other 20 have a 512-row prefix.
- One consistent **23-field Zeek schema** across all 23; required `ts`, `uid`, `id.orig_h`, `id.resp_h`, `label`, `detailed-label` columns present. **0** malformed sampled row widths, **0** invalid or implausible sampled timestamps. No failed source record checks in this bounded sample.
- `label` totals: original *case-sensitive* `benign` **1,094** (three pure benign-device logs); `Benign` **3,921** (benign-labeled rows appearing within malware scenarios); `Malicious` **6,044**. `detailed-label=-` **5,015** and native malicious exact detailed-label rows **6,044**. These are prefix counts, **not representative prevalence**. Normalization of `benign` vs `Benign` remains a FUTURE explicit data policy, not a source-audit rewrite.
- Metadata lists **~47.06 GB** of uncompressed native labeled flow log files across the 23 captures, so an all-record exact-support census is a costly distinct run. The present report neither counts source-wide label support nor demonstrates independent train/val/test units.

### Native exact `detailed-label` presence in FIRST sampled flow positions

| Exact source text | Source capture IDs with at least one sampled row | Prefix rows across those captures |
|---|---:|---:|
| `-` (benign detail) | 19 | 5,015 |
| `PartOfAHorizontalPortScan` | 9 | 3,459 |
| `C&C` | 9 | 806 |
| `Okiru` | 5 | 1,428 |
| `DDoS` | 3 | 179 |
| `C&C-FileDownload` | 5 | 43 |
| `C&C-HeartBeat` | 4 | 102 |
| `C&C-Torii` | 2 | 12 |
| `FileDownload` | 2 | 9 |
| `C&C-HeartBeat-FileDownload` | 1 | 5 |
| `C&C-Mirai` | 1 | 1 |

**Critical semantics:** these label strings **mix observable behaviors (DDoS, port scanning, C&C), compound behaviors, and malware names (Okiru)**. Do not silently treat every string as a homogeneous attack family. Native C&C variants are NOT merged, and no feature or taxonomy change has been authorized. Zero prefix occurrence does NOT prove the capture lacks a label later.

## Source metadata warning: CTU-Honeypot-Capture-7-1 (published exception)

The original output reported a single `problems` item: `At least one native capture README.md is missing`. This is the only source of its original `[PARTIAL]` return status. The 23rd source is **not missing its Zeek label log**: it exists as `CTU-Honeypot-Capture-7-1/Somfy-01/bro/conn.log.labeled` and the first **130/130** native rows parsed successfully as lowercase `benign`.

Original author index:
- https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-Honeypot-Capture-7-1/
- https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-Honeypot-Capture-7-1/Somfy-01/
- https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-Honeypot-Capture-7-1/Somfy-01/bro/

The publisher's public index **contains no root README** for this scenario and instead lists **six separate Somfy-01 through Somfy-06 child directories**. Somfy-01 includes native labeled Zeek flow log; **Somfy-02 through Somfy-06 list PCAP files but no labeled Zeek connection log** in their public directory indexes. Therefore missing *root* README is a genuine publisher-side documentation caveat, not evidence the user's archive extraction is corrupt. The root-level `root_pcap_file_count=0` is also misleading in this one source because PCAPs live under child subcapture folders. The original capture-group support model must not silently infer *six labeled independent benign executions* from these filenames.

**Narrow code correction committed after inspecting the report:** `scripts/17_audit_iot23_capture_support.py` now distinguishes data-validation `problems` from nonfatal `source_metadata_warnings` for root README absence and nested PCAP-only subfolders. It records `pcap_file_count_including_nested` without opening PCAPs. One synthetic regression test in `tests/test_iot23_capture_support_audit.py` emulates Somfy-01 with Zeek label log plus five PCAP-only child directories and checks preserved provenance caveats. Existing user report from old version remains valid for its observed rows; **no repeat of the already completed preflight is necessary merely to change status phrasing**. New test/full repo suite still must run user-locally before the next large scan; do not claim verified pytest success.

## Scientific status, next controlled gate

**Preflight's source/schema/initial timestamp gate is PASSED**, with a visible benign-source documentation caveat. **IoT-23 is NOT admitted as a third primary Protocol B dataset.**

The remaining binding acceptance questions are:
1. Whole-source **capture × native exact malicious detailed-label support**, including record counts and benign support (no source-specific prefix prevalence claims).
2. Scientifically coherent **attack behavior/family taxonomy** without post-hoc lumping of C&C variants or calling Okiru a traffic behavior. No label mapping approved yet.
3. Distinct author-documented malware capture scenarios must constitute **independent experimental collection units for the intended claim**. Potential same-malware environment/provenance confounding, source-rule-based labeling shortcuts and train/test identity overlap require support audit.
4. Sufficient known-family plus benign support in **train, validation and test** after whole-capture grouping, with unchanged existing Protocol B support rules. NO random row split to manufacture independence.
5. No label or source ID/IP/timestamp proxy may enter model features without a prospective, defensible feature policy.

**No full 47 GB / roughly 325m flow census is automatically authorized by receipt of the preflight JSON.** A targeted author-README capture×native-label count matrix may be a cheaper intermediate validation, but original labeled-flow exhaustive counts remain necessary to certify support if paper-level tables cannot certify the exact version and relevant labels. Ask user before expensive next scan. If this final IoT-23 attempt fails, honor prior explicit stop rule: end third-primary dataset search, return to five-seed paired visible/blind, repeated partitions, genuine literature methods and machine-readable stability records, and ask user before changing original mandatory third-dataset requirement.

No model runs, split/group assignments, source changes, campaign freeze, historic COMNET release edits or manuscript claims were made by this diagnostic.
