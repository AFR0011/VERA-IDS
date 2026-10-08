# VERA-IDS v2026.10-comnet

Computer Networks submission snapshot for:

**VERA-IDS: Testing Conclusion Stability in Network Intrusion Detection under
Operating, Open-Set, and Partition Constraints**

## What changed after v2026.10

- reframed the manuscript around conclusion stability as the explicit validity target;
- added five-seed staged model-competence checks with Extra Trees, LightGBM, and
  CatBoost against the primary RF/XGB systems;
- added independently specified classifier-profile transfers derived from
  Adewole, Hung, Keskin, and Christy;
- completed a five-seed CICIoT2023 external-profile Protocol-B campaign showing
  persistent held-out-family dependence across Adewole-derived XGB,
  Hung-derived XGB, and Keskin-derived LightGBM configurations;
- retained Protocol-A profile transfer as a configuration-transfer diagnostic,
  not a matched leaderboard;
- added safeguards against reusing pre-safeguard single-class Protocol-B smoke
  artifacts;
- froze the compact aggregate evidence used by Tables 5-6 and Supporting
  Information Tables S7-S12.

## Scientific boundaries

This release does not claim state-of-the-art model performance or end-to-end
reproduction of the source studies. The competence lane addresses the
weak-primary-model alternative explanation. The external-profile lane tests
configuration generality for held-out-family dependence. Other
conclusion-stability effects remain bounded to the model families, datasets,
and comparisons evaluated in the manuscript.

The external Protocol-A lane uses its predeclared fixed transfer caps and must
not be interpreted as an apples-to-apples competition with the primary
dataset-specific competence lane.

## Verification

Before creating the GitHub tag/release:

```bash
python scripts/16_verify_comnet_release.py
python scripts/recompute_protocol_a_metrics.py --check
python scripts/verify_manifests.py
python scripts/check_links.py
```

The release verifier requires five-seed coverage for the competence reference,
external Protocol A, and external Protocol B and checks complete six-family
CICIoT2023 holdout coverage for all three external Protocol-B profiles.

## Release contents

The Computer Networks-specific compact evidence is under
`release/v2026.10-comnet/raw/`. Earlier core manuscript records remain under
`outputs/summaries/` and `release/v2026.10/raw/` and are mapped in
`release/v2026.10-comnet/PROVENANCE.md`.

Benchmark datasets, prepared rows, raw prediction tables, and trained models are
not redistributed.
