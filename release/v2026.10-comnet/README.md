# VERA-IDS v2026.10-comnet

This directory is the frozen reproducibility snapshot for the Computer Networks
submission:

**VERA-IDS: Testing Conclusion Stability in Network Intrusion Detection under
Operating, Open-Set, and Partition Constraints**

It extends the earlier `v2026.10` manuscript snapshot with the final
five-seed model-competence and external-profile comparison campaigns used in
the Computer Networks revision.

## Scope

The snapshot freezes compact aggregate evidence and provenance. It does not
redistribute benchmark datasets, prepared flow rows, raw prediction tables, or
trained models.

The comparison extension has two deliberately separate purposes:

1. **Model competence.** Extra Trees, LightGBM, and CatBoost are evaluated
   under the study's staged Protocol-A task to test whether the primary RF/XGB
   configurations are obviously weak. This is a competence sanity check, not a
   SOTA leaderboard.
2. **Configuration generality.** Independently documented classifier settings
   are transferred into VERA conditions. Protocol A is a transfer diagnostic;
   the standardized CICIoT2023 Protocol-B lane tests whether held-out-family
   dependence persists beyond the primary RF/XGB parameterizations.

Published source-study scores are provenance only and are not treated as
matched comparisons with VERA results.

## Frozen comparison evidence

- `raw/model_competence_summary.csv`
- `raw/primary_vera_five_seed_reference.csv`
- `raw/external_protocol_a_summary.csv`
- `raw/external_protocol_b_holdout_summary.csv`
- `raw/external_protocol_b_family_heterogeneity_summary.csv`
- `raw/final_campaign_gate.txt`

The earlier core manuscript evidence remains in the repository under
`outputs/summaries/` and `release/v2026.10/raw/`; the exact mapping is in
[PROVENANCE.md](PROVENANCE.md).

## Release gate

All final comparison surfaces use seeds 123-127. The final campaign audit
reports five-seed coverage for the competence reference, external Protocol A,
and external Protocol B.

Before creating the GitHub release, merge the comparison branch and create tag
`v2026.10-comnet` from the merged submission commit. The manuscript and Supporting
Information refer to that tag.
