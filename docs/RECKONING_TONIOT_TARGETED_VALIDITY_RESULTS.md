# TON_IoT targeted validity audit — Phase B7 results (2026-10-18)

Status: **user-local full processed-source diagnostic completed; TON_IoT dataset role NOT approved or decided**.

Evidence: user-uploaded `outputs/17_revision_campaign/toniot/targeted_validity_audit.json`, produced by revision-branch `scripts/17_audit_toniot_validity.py`. It contains aggregated full-source time-bin and attack-address-range counts and deterministic bounded per-class address samples. The report itself contains no source IPs or individual flows. These are observational diagnostics, not causal feature-leakage ablations, source-independent test partitions, or authenticated attack-operation records.

## Scan integrity and coverage

- 23 processed Network CSVs; **22,339,021** scanned records. Matches previously reviewed full census and official class totals. File 6 has 47 header columns due to `uid`; other 22 have 46.
- **Nine attack-family timestamp windows are strictly pairwise nonoverlapping**. All 99 one-hour bins containing attack records include at most one attack family. **98 of 99** also contain `normal`; the remaining scanning-only bin has 5,584 attacks and **zero benign records**. Of attack-bearing bins, **20** have fewer than 1,000 benign flows (including the zero-benign bin). There are **146** occupied hourly bins total, **145** with benign.
- Presence of benign in a time window does not prove independence or enough precision for rate-based metrics: DDoS bins can have only 166 benign rows despite hundreds of thousands of attacks. Per-hour matched-FPR diagnostics require explicit uncertainty and group design, not ad hoc thresholds.

## Class-conditional identity association (diagnostic reservoirs, not population prevalence)

Default deterministic reservoir: max 4,096 records **per type** (MITM fully sampled: 1,052). The records below reflect *only* the in-memory sample, not an exhaustive unique-IP census.

| Type | Source IPs in sample | Rows whose source IP also appears in sampled benign rows |
|---|---:|---:|
| scanning | 14 | 177 / 4096 |
| dos | 3 | 0 / 4096 |
| injection | 6 | 0 / 4096 |
| ddos | 5 | 0 / 4096 |
| password | 6 | 0 / 4096 |
| xss | 4 | 0 / 4096 |
| ransomware | 3 | 688 / 4096 |
| backdoor | 3 | 4094 / 4096 |
| mitm | 6 | 9 / 1052 |
| normal | 186 | N/A |

- For `dos`, `ddos`, `injection`, `password`, and `xss`, the 4,096 sampled attack rows share **no observed source IP** with the 4,096 sampled benign rows. This is a substantial **candidate binary label/source-identity shortcut** if raw addresses are used. It does NOT establish separability against the whole benign population or prove actual model reliance.
- Several sampled source IPs appear across multiple attack families, indicating a specific attack-family lookup from raw source IP alone is NOT established. The phenomenon is binary-label-versus-normal separation in these samples, and likely environmental identity confounding.
- Source `192.168.159.30` through `192.168.159.39` (provider README's documented attack-origin range) was found in **zero rows**, as `src_ip` **and** `dst_ip`, over the complete 22.34m processed rows. This is a real discrepancy between documentation and this processed representation. Possible causes (not verified): different address plane, transformation, different capture modality, or documentation applicability. It is NOT evidence of absence of address leakage. Do not invent mappings or drop addresses without a frozen protocol decision.

## Temporal and episode-proxy diagnostics

| Type | One-hour attack bins | With benign | Consecutive same-type gaps >=5 min | >=30 min | >=1 h | Backward timestamp transitions within type |
|---|---:|---:|---:|---:|---:|---:|
| scanning | 8 | 7 | 7 | 3 | 2 | 0 |
| dos | 9 | 9 | 0 | 0 | 0 | 0 |
| injection | 6 | 6 | 0 | 0 | 0 | 0 |
| ddos | 18 | 18 | 0 | 0 | 0 | 13 |
| password | 15 | 15 | 0 | 0 | 0 | 0 |
| xss | 9 | 9 | 1 | 1 | 0 | 4 |
| ransomware | 4 | 4 | 1 | 1 | 0 | 0 |
| backdoor | 27 | 27 | 1 | 0 | 0 | 0 |
| mitm | 3 | 3 | 5 | 0 | 0 | 0 |

**Note:** gap counters operate on consecutive same-type observations in CSV input order, NOT a fully sorted per-family event series. With 13 reversals in DDoS and 4 in XSS, they should not be interpreted as complete sorted chronological gap statistics. Even correctly ordered gaps do not identify separate independent attack campaigns. `ts` boundaries, chronological file chunks, IP changes and 1-hour bins are NOT validated session IDs.

The **full-source temporal confounding finding from earlier phases stands**: attacks occupy separate chronology windows. It is possible to make row-level train/validation/test splits under some definitions, but there is no independent-capture grouping shown; LOAO with shuffled flow-level partitions would be an *intra-corpus, potentially within-episode evaluation* and cannot be advertised as cross-source/time generalization. Removing `ts` alone does not eliminate time-correlated endpoint/protocol/traffic artifacts.

## Role recommendation, not decision

**Recommendation:** Do **not** promote TON_IoT to the third full **primary Protocol B LOAO** environment for the archival-value campaign on current evidence. The proposed source/time-independent family-supported partitions are not justified; attack families are perfectly time-separated, and there is considerable identity confounding. These shortcomings are about the intended scientific claim, not evidence that the raw data are corrupt. Previous full class counts and GT sample-flow matches remain valid and valuable.

Most defensible conditional uses, subject to explicit user approval and later frozen feature/split policy:
1. **Qualified Protocol A / external closed-set stress lane** across a distinct collection environment, with strict no-test-selection, declared within-corpus limitations, identity/timestamp handling, sample support, and validation-only configuration. Does not prove unseen-condition generalization or multiple attack repetitions.
2. **Further provenance acquisition** only if authoritative original capture/attack-operation manifests establish independent attack episodes and defensible support, after which re-review Protocol B. The current CSVs cannot establish this.
3. **Alternative third primary dataset** with documented independent collection units and defensible per-family support. This requires a separate feasibility check, not automatic promotion.

### Unresolved

- Exact number of independent attack executions per family, beyond observed timestamps/flows.
- Reason README's attacker IPs are absent from the processed IP columns.
- Feature leakage/proxy bias *causally* established through frozen split and ablation: not tested.
- Attack-to-benign identity overlap in **whole** dataset, as opposed to deterministic per-class 4,096-row sample.
- Feasible Protocol A sample cap/partition quality; no supervised experiments or metrics have been run.
- Actual full-repository `pytest` output after the newest audit changes has not been provided.

**Methodology changes must be approved by the user before coding preparation or running models.** Freeze historical `release/v2026.10-comnet/` and leave draft PR #3 unmerged while scope is decided.
