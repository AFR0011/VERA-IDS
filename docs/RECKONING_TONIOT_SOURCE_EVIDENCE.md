# TON_IoT Network source evidence — Phase B3

Status: **preliminary source metadata assessment, 2026-10-18**. No dataset promotion, feature set, labels, split, or experiment has been approved. Evidence: official `Network Features-Description.pdf` and `Statistics of Network Records.xlsx`, a local bounded source manifest, and separately provided user SHA-256 outputs. The raw CSVs were **not** accessible in this review environment; full-row counts are not independently audited until the local complete scan succeeds.

## Bounded processed source manifest

- User's full processed Network directory has `Network_dataset_1.csv` through `Network_dataset_23.csv`.
- Existing read-only preflight completed against all 23 files with at most **10,000 initial rows per file**, 230,000 observed rows. Total file size: **3,372,475,448 bytes**. Neither full row counts nor processed-file SHA-256 were produced.
- **22 files have 46 columns; `Network_dataset_6.csv` has 47 columns and adds `uid`.** All include `ts`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `label`, and `type`. No duplicate headers were detected.
- File 1 begins with normal; 2–8 scanning; 9–11 DoS; 12 injection; 13–18 DDoS; 19–20 password; 21–22 XSS; 23 backdoor. These are **initial 10k-row observations**, not proof that files contain only one class. The bounded sample does not expose MITM or ransomware, although both are reported in the provider's full summary. File numbers are **not** evidence of capture/temporal independence.

## Provider full Network class counts (not independently confirmed)

| `type` | Provider total rows |
|---|---:|
| scanning | 7,140,161 |
| ddos | 6,165,008 |
| dos | 3,375,328 |
| xss | 2,108,944 |
| password | 1,718,568 |
| backdoor | 508,116 |
| injection | 452,659 |
| ransomware | 72,805 |
| mitm | 1,052 |
| normal | 796,380 |
| **Total** | **22,339,021** |

The workbook separately reports **461,043 sampled training/testing rows**; those are NOT an equivalent replacement for the full corpus. Attack prevalence is approximately 96.4%; MITM has only 1,052 reported rows globally. Its Protocol B leave-one-attack-out feasibility is UNDECIDED under minimum held-out validation/test support constraints.

## Documented features and suspected threats to validity

The provider PDF defines `ts` as connection timestamp, source/destination IP and port as endpoints, Zeek/Bro connection/DNS/SSL/HTTP/violation data as connection features, `label` as binary normal/attack, and `type` as attack category. The 47th `uid` occurs in CSV #6 but is not described by this PDF. The CSVs include `http_referrer`; the PDF's HTTP feature numbering skips that row. Do not rely on documented types (e.g., `http_user_agent` is described as numeric) without actual field validation.

Candidate leakage/identity-sensitive predictors needing explicit decisions: `ts`, `uid`, IP addresses, ports, `dns_query`, `http_uri`, `http_referrer`, SSL certificate fields, and other environment-specific strings. The official README attributes labels to source IP addresses and event timestamps, so derivation leakage must be investigated. Field presence is not itself proof of leakage.

## Ground truth

The downloaded source has **18** `GroundTruth_Network_*.csv` files vs 23 processed CSVs. Index-based pairing has no demonstrated validity. The user verified `GroundTruth_Network_15.csv` and `_16.csv` have identical SHA-256 (`D1D4CB491CB6EDDA3523F7424D24EA1AFD4496A933E60B3EC357286F7F0F542F`). A grouped scan returned only that duplicate whole-file group (conditional on all 18 being included). Other files may still contain overlapping rows/events. Preserve all original files.

## Next read-only full-source metadata scan

The supplementary script `scripts/17_audit_toniot_metadata.py` streams every processed record, reports per-file and aggregate label/type counts and pair consistency, numeric timestamp min/max by file/family, invalid time values, malformed row widths, and inventories ground-truth **headers only**. It does NOT match flows against labeled events, identify disjoint captures, select any features, or declare leakage-free splits. Optional `--sha256` adds another full-file read and is off by default.

Run from VERA-IDS repository root:

```powershell
uv run python scripts/17_audit_toniot_metadata.py --processed-root "data/raw/ToN_IoT/Processed_Network_dataset" --groundtruth-root "data/raw/ToN_IoT/SecurityEvents_Network_datasets" --out "outputs/17_revision_campaign/toniot/full_metadata_audit.json"
```

This reads approximately 3.37 GB of CSV data (provider says 22.3 million rows) and may take substantial time locally. It only writes a small aggregate report under ignored `outputs/17_revision_campaign/`. Do not treat partial progress as an accepted audit. The new script passed two synthetic local tests during development; user-side full suite and real source scans remain **PENDING**.

**Further gates after report:** reconcile provider-vs-observed class counts; inspect true timestamp/label combinations and ground-truth headers; establish event-to-flow provenance; audit duplicates and identity leakage; approve candidate source/time partitioning and support; only then decide whether TON_IoT can join primary Protocol A/B evaluations. Never modify `release/v2026.10-comnet/`.
