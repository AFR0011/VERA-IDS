# IoT-23 I4 — Standalone benign-source census complete and Protocol B support-rule check

**Date:** 2026-10-23  
**Status:** Local source-count check COMPLETE; feasibility of **third primary Protocol B still UNDECIDED**. No model fitting, feature/label changes, capture-to-split assignments, whole-corpus census or dataset promotion authorized.

## User-local evidence and arithmetic

User supplied `outputs/17_revision_campaign/iot23/capture5_benign_native_counts.json`, produced by `scripts/17_audit_iot23_capture_support.py --input-root data/raw/IoT23/CTU-Honeypot-Capture-5-1 --expected-captures 1 --phase census`.

The report independently passed these invariants:
- `schema_version=iot23-native-source-label-audit-v1`, `mode=census`, `complete_source_row_census=true` for **this ONE capture**; `capture_scenarios_found=1`, source type `benign_device`, native `bro/conn.log.labeled` file of 182,147 bytes and 23 native Zeek columns.
- **1,374 complete flows**, exactly 1,374 case-sensitive lower-case `benign` and 1,374 native `detailed-label="-"`, with one matching native `label`/`detailed-label` pair; UTC-day source counter 1,374 on 2018-09-21. Min/max timestamps 2018-09-21 09:40:22.965530 to 15:09:33.267978 UTC.
- Zero malformed-width lines, zero invalid/implausible timestamps, `problems=[]`, `source_metadata_warnings=[]`, capture README present and one original PCAP file (metadata only). Label totals, pair totals and day totals all equal 1,374.
- This is the **full native labeled-flow population for source 5-1 only**. It does NOT verify independence of flows, bona fide absence of malicious traffic or any later model FPR.

Together with actual previous observed EOF-complete captures (not mere 512-row extrapolation):
| Native original benign device capture | Fully examined labeled flows | Native exact binary label | Notes |
|---|---:|---|---|
| `CTU-Honeypot-Capture-4-1` | **452** | `benign` | Prefix preflight reached EOF, 452 of 452 |
| `CTU-Honeypot-Capture-5-1` | **1,374** | `benign` | New complete I4 census |
| `CTU-Honeypot-Capture-7-1/Somfy-01` | **130** | `benign` | Prefix preflight reached EOF, 130 of 130; publisher Somfy-02..06 include PCAP-only sources and no native labeled logs |
| **Total three labeled device captures** | **1,956** | All native lowercase `benign` | Three source captures; flows are not independent experimental replicates |

All three have distinct capture IDs and collection dates, but a source ID or calendar date alone does not prove independence of traffic-generation devices or absence of upstream artifact overlap. The publisher also reports many more uppercase `Benign` flows embedded within 20 infected-source captures, which share those source contexts and must not be silently treated as independently collected clean benign-device runs.

## Existing VERA Protocol B support contract (no changes)

Read directly from revision-branch existing `config/protocol_b_loao.yml` and `config/protocol_b_support_ciciot2023.yml`, both consistent:
- `min_benign_rows.val = 200`; `min_benign_rows.test = 200`.
- `min_family_support.train = 200`, `min_family_support.val = 200`, `min_family_support.test = 200` for every represented known attack family.
- `unknown_support_for_holdout.val = 200`; `unknown_support_for_holdout.test = 200`; unknown validation support is required when tuning rejection `tau`.
- `min_known_families_after_holdout = 2`; `require_all_remaining_train_families_to_be_valid = true`.

**Necessary-but-not-sufficient benign result:** Capture 4-1 (452) and 5-1 (1,374) each can independently satisfy the numeric 200-flow benign validation or test minimum if assigned to different phases. Somfy-01 (130) cannot individually satisfy either. These are *possible support suppliers*, NOT an approved assignment. Captures 4 and 5 alone do not establish adequate source-diverse Stage 1 training, which may need benign from infected-source captures; using such rows requires a scientifically prospective group-preserving and label-validity justification. Observations across unrelated devices/time periods could also cause distribution shift and confound matched 1.5% benign-FPR, especially at small per-source counts. Do not relax support or interpret minimal row counts as reliable independent FPR estimates.

**Important non-implication:** The availability of at least 200 benign observations in two source groups **does not certify** that simultaneously admissible whole-capture train/validation/test partitions exist for all selected known families and the left-out unknown family. No matrix partition, LOAO holdout, family taxonomy, group-validation, feature exclusion, or model operating point was chosen.

## Combined evidence and remaining FINAL IoT-23 admissibility gates

Previously:
- I1 actual source prefix: 23/23 labeled scenario logs, 20 malware / 3 benign-device, one consistent 23-column Zeek schema, zero malformed or invalid sampled timestamps; publisher Somfy subcapture caveat preserved.
- I2 complete *publisher* per-capture exact-source label counts: `PartOfAHorizontalPortScan` 213,852,924 flows across 11 captures (10 with >=1,000), `DDoS` 19,538,713 across seven (six >=1,000), `C&C` 21,995 across ten (five >=1,000), from original v1 distribution; >=1,000 is descriptive, **not** the 200 support rule.
- I3 actual three selected complete local malware capture censuses (34-1, 44-1, 3-1) matched every published native label count, 179,485 records, no structural failures. Capture 33 local `Okiru` vs published `Okiru-Attack` discrepancy still open, and full source not scanned.

**Next gate is primarily SCIENTIFIC:** use the publisher's author-recorded independent malware-run/capture metadata and the exact label matrix to check whether ≥3 semantically coherent exact behavior labels can meet unchanged 200-row known/unknown/benign support for whole-capture held-out family experiments, and whether rule-based labeling yields unacceptably circular model targets under a precommitted leakage-safe feature policy. Distinct captures alone, large flow counts alone, and good sample-count arithmetic are not enough. No auto taxonomy merging such as `C&C` + `C&C-HeartBeat` or `Okiru` + `Okiru-Attack`.

**Stopping rule:** IoT-23 was the *final* third-primary candidate approved for investigation. If group independence/benign eligibility/label semantics/leakage cannot be defended to the user's current scientific standard without altering the task, **close the search**, discuss revising the mandatory third-primary requirement, and continue the already planned paired visible/blind CICIoT2023 experiment, repeated CICIDS2017 group-safe partitions, genuine literature method baselines, and machine-readable conclusion-stability evidence. Avoid another 47 GB full-label scan unless a specific, approved unresolved necessity would change an admission decision.

**Provenance and invariance:** User original data remains local under ignored `data/raw/IoT23/`, full report under ignored `outputs/17_revision_campaign/iot23/`. No raw source content, packet payloads, IP addresses, flows, model training, or original release files moved to GitHub. Draft PR #3 remains unmerged. This report records evidence, not protocol approval.
