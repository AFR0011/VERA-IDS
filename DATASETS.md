# Datasets

Raw benchmark data and row-level prepared derivatives are intentionally excluded.
Always review the provider's current terms and cite the official dataset paper.

## Official sources and project roles

| Dataset | Role | Acquisition | Scope used | Redistribution position |
| --- | --- | --- | --- | --- |
| CICIDS2017 (CIC-IDS2017) | primary Protocol A and recovered Protocol B | [UNB/CIC official page](https://www.unb.ca/cic/datasets/ids-2017.html) | full prepared corpus: 2,099,879 rows | official page describes research availability and requires citation; this repository still excludes the data |
| CICIoT2023 | primary Protocol A and Protocol B | [UNB/CIC official page](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | capped subset: 6,067,478 rows after `target_per_fine_label: 300000` | no sufficiently clear redistribution grant was established; exclude by default |
| NSL-KDD | external Protocol A stress test | [UNB/CIC official page](https://www.unb.ca/cic/datasets/nsl.html) | source-defined external prepared split | provider permits redistribution with citation, but no row data is needed here |
| UNSW-NB15 | external Protocol A and invalid Protocol B diagnostic | [UNSW official page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | source-defined external prepared split | academic-use terms require review, especially for commercial reuse; exclude data |

## Expected local layout

```text
data/
  raw/
    CICIDS2017/
    CICIoT2023/
    NSL-KDD/
    UNSW-NB15/
  processed/
    protocol_a/
    protocol_b/
```

`data/raw/`, `data/processed/`, and local path configuration are ignored by Git.
The original workspace used CSV/CSV.GZ split parts and metadata files such as
`SPLIT_REPORT.json`, taxonomy mappings, and family-support tables.

## Required schema and preprocessing contract

The canonical processed labels are:

- `y_stage1_attack`: binary benign/attack target;
- `y_stage2_family`: attack-family target;
- `y_stage2_fine`: fine-grained target where applicable;
- original `label` and `attempted_category` columns may be retained only as
  metadata and are never model features.

The engine explicitly excludes `label`, `attempted_category`, `y_stage1_attack`,
`y_stage2_family`, and `y_stage2_fine` from features. Preprocessing is fit on
training data only, numeric features may be scaled, and limited-cardinality
categorical features are one-hot encoded. Dataset-specific taxonomy mappings and
the exact input columns must be regenerated from the official files because a
portable schema/checksum manifest is not yet available.

## Exact split construction recorded by source

| Dataset | Protocol/split | Train | Validation | Test | Rule |
| --- | --- | ---: | ---: | ---: | --- |
| CICIDS2017 | A stratified | 1,469,531 | 315,174 | 315,174 | 70/15/15 family-stratified, seed 123 |
| CICIoT2023 | A stratified | 4,245,363 | 911,062 | 911,053 | 70/15/15 fine-label-stratified, seed 123, capped subset |
| CICIDS2017 | B original day/file | 361,994 | 919,178 | 818,707 | support-aware file/day split |
| CICIDS2017 | B recovered | 1,210,961 | 444,453 | 444,465 | contiguous within-day recovery split |
| CICIoT2023 | B day/file | 3,676,338 | 1,200,241 | 1,190,899 | support-aware day/file split, capped subset |

Protocol B's recorded support requirements are 200 benign validation/test rows,
200 rows per known family in train/validation/test, 200 held-out-family rows in
validation/test, and at least two known families after holdout. Both primary
support snapshots report six eligible holdouts at the selected 200-row threshold.

## Integrity and leakage checks

No authoritative official-download checksums were stored. Before use, record the
download URL, date, archive checksum, extracted-file checksums, row/column counts,
and provider citation in a local data manifest.

Split construction and integrity checks are dataset- and protocol-specific.
Users should run the packaged audit workflow on locally acquired inputs before
making split-validity claims about a new preparation.

## Known dataset limitations

- benchmark traffic is not a proxy for current production traffic;
- CICIoT2023 uses a capped subset rather than its full public corpus;
- family aggregation can hide fine-label imbalance and ambiguity;
- day/file construction can produce missing family or benign support;
- UNSW-NB15's external Protocol B snapshot has zero benign validation/test
  support and no valid holdouts, so it is diagnostic only;
- dataset licenses and web availability may change after this audit date.
