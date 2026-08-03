# VERA-IDS

VERA-IDS is a validity-aware evaluation framework for two-stage machine-learning
intrusion detection. Release `v2026.08` provides package-local experiment logic,
compact aggregate evidence, corrected Protocol A metrics, support-audited
Protocol B evaluation, tests, and a repository-edition thesis PDF.

Repository: <https://github.com/AFR0011/VERA-IDS>
Release: <https://github.com/AFR0011/VERA-IDS/releases/tag/v2026.08>
Thesis PDF: <https://github.com/AFR0011/VERA-IDS/releases/download/v2026.08/VERA-IDS-Thesis.pdf>
Protocol A correction: [PROTOCOL_A_CORRECTION.md](PROTOCOL_A_CORRECTION.md)
Asset hashes and page counts: [release/ASSET_MANIFEST.json](release/ASSET_MANIFEST.json)

## Scientific scope

- Protocol A: closed-set stratified evaluation. The primary system macro-F1
  averages only labels with genuine ground-truth support. Predictions of
  `Unknown` remain errors for their supported true classes.
- Protocol B: support-audited leave-one-attack-family-out evaluation. `Unknown`
  remains in the metric label set because valid folds contain true-Unknown rows.
- UNSW-NB15 Protocol B: retained only as an invalid diagnostic; zero benign
  validation/test support makes it ineligible for comparison or positive claims.
- Threshold experiments: the broad exploratory grid and the validation-selected
  rejection policy are separate lanes. Neither is universally canonical.
- Raw datasets, prepared rows, predictions, and approximately 63.9 GB of
  historical models are not distributed.

## Protocol A correction

The release contains 136 path-free aggregate confusion matrices with original
source hashes. Recompute the primary and historical metrics without data or
models:

```bash
python scripts/recompute_protocol_a_metrics.py --check
python scripts/verify_manifests.py
```

Two exact release gates are CICIDS2017 RF strict `0.818902771736530` and
CICIoT2023 RF strict `0.897154513910465`. See the correction supplement for the
complete definition, corrected tables, provenance, and interpretation.

## Installation

Python 3.12 is required. The lock targets portable CPU verification on Windows
and Linux.

```bash
uv sync --locked --python 3.12 --extra test
```

The historical experiment profile was Windows 11, Python 3.12.10, RTX 4090,
scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0, and CatBoost 1.2.10. It is
documented provenance, not a claim that the GPU experiments were rerun for this
release. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Numbered workflow

All commands are implemented inside the installed package and support
side-effect-free dry runs plus bounded deterministic `--synthetic` workflows.

```bash
python scripts/01_audit_and_prepare.py --dry-run
python scripts/02_run_protocol_a_two_stage.py --dry-run
python scripts/03_run_protocol_a_flat_baseline.py --dry-run
python scripts/04_audit_protocol_b_support.py --dry-run
python scripts/04b_analyze_protocol_b_support_sensitivity.py --dry-run
python scripts/05_run_protocol_b_loao.py --dry-run
python scripts/06_run_open_set_rejectors.py --lane validation-selected --dry-run
python scripts/06_run_open_set_rejectors.py --lane exploratory-grid --dry-run
python scripts/07_run_external_stress_tests.py --dry-run
python scripts/08_build_statistics.py --dry-run
python scripts/09_build_paper_pack.py --dry-run
python scripts/10_run_seed_reliability.py --dry-run
python scripts/11_run_reference_framework_eval.py --dry-run
python scripts/11b_build_reference_framework_comparison.py --dry-run
```

Run every advertised command in both modes with:

```bash
python scripts/verify_cli_workflows.py --all
```

Real experiment runs require benchmark data acquired from its providers. Models
are trained by the workflow and are intentionally excluded from Git and the
release; downloadable-model reproduction is not claimed.

## Verification

```bash
uv run python -m compileall -q src scripts tests
uv run python -m pytest -p no:cacheprovider
uv run python scripts/verify_cli_workflows.py --all
uv run python scripts/recompute_protocol_a_metrics.py --check
uv run python scripts/verify_manifests.py
uv run python scripts/check_links.py
uv run python scripts/smoke_test.py
uv run python scripts/release_audit.py scan .
```

GitHub Actions repeats locked installation, heavy imports, compilation, tests,
CLI dry runs, correction/manifest checks, link checks, and release scanning on
Windows and Ubuntu.

## Thesis

`VERA-IDS-Thesis.pdf` is attached to the GitHub release under CC BY 4.0. It has a
repository-edition cover stating that GitHub is not the institutional record and
linking to the Protocol A correction supplement. The source DOCX is not
published, tracked, or modified by this repository workflow.

## Licensing

Copyright © 2026 Ali Farrokhnejad.

- Original software, scripts, tests, and configuration logic: MIT (`LICENSE`).
- Original documentation, figures, aggregate evidence/results, correction
  supplement, and release manuscript PDF: CC BY 4.0 (`CONTENT_LICENSE.md`).
- Third-party datasets and dependencies: upstream terms; not redistributed.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [DATASETS.md](DATASETS.md).
