# Source and configuration map

The `v2026.10` tag should target the commit containing this directory. That tag pins the complete repository source tree, so the core experiment code does not need to be duplicated inside the release folder.

Principal workflow entry points:

- `scripts/01_audit_and_prepare.py`
- `scripts/02_run_protocol_a_two_stage.py`
- `scripts/03_run_protocol_a_flat_baseline.py`
- `scripts/04_audit_protocol_b_support.py`
- `scripts/04b_analyze_protocol_b_support_sensitivity.py`
- `scripts/05_run_protocol_b_loao.py`
- `scripts/06_run_open_set_rejectors.py`
- `scripts/07_run_external_stress_tests.py`
- `scripts/08_build_statistics.py`
- `scripts/10_run_seed_reliability.py`
- `scripts/11_run_reference_framework_eval.py`
- `scripts/11b_build_reference_framework_comparison.py`

Key implementation modules:

- `src/ids_eval_framework/src/two_stage_engine.py`
- `src/ids_eval_framework/_native/protocol_b_open_set.py`

Current public configuration files relevant to the manuscript:

- `config/datasets.yml`
- `config/protocol_a.yml`
- `config/protocol_b_loao.yml`
- `config/protocol_b_support_ciciot2023.yml`
- `config/open_set_validation_selected_rejection.yml`
- `config/open_set_exploratory_threshold_grid.yml`
- `config/seed_reliability.yml`
- `config/reference_framework_eval.yml`

Note that the older Supporting Information path `config/thresholds.yml` is obsolete in the public repository. The open-set settings are split between the validation-selected and exploratory-grid configuration files listed above.

The September comparison and diagnostic surfaces in `release/v2026.10/raw/` were assembled from frozen results rather than by rerunning model training.
