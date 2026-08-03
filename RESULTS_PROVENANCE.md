# Results Provenance

| Surface | Code and CLI | Configuration | Public evidence | Metric boundary |
| --- | --- | --- | --- | --- |
| Protocol A core | `scripts/02_run_protocol_a_two_stage.py`, native package modules | `config/protocol_a.yml` | `protocol_a_core_summary.csv`, 136-matrix evidence | supported labels primary; declared output historical |
| Protocol A flat | `scripts/03_run_protocol_a_flat_baseline.py` | `config/protocol_a.yml` | `protocol_a_flat_vs_two_stage.csv` | direct multiclass labels |
| Protocol B LOAO | `scripts/04*`, `scripts/05*` | `config/protocol_b_loao.yml` | best-per-holdout and support tables | includes supported true Unknown |
| Rejection, validation-selected | `scripts/06_run_open_set_rejectors.py` | `config/open_set_validation_selected_rejection.yml` | open-set/sink-aware summaries | support-eligible Protocol B folds |
| Rejection, exploratory | same CLI with `--lane exploratory-grid` | `config/open_set_exploratory_threshold_grid.yml` | exploratory output only | broad sensitivity lane |
| External Protocol A | `scripts/07_run_external_stress_tests.py` | `config/datasets.yml` | `external_protocol_a_summary.csv` | supported labels primary |
| Statistics | `scripts/08_build_statistics.py` | `config/protocol_b_loao.yml` | paired tests and confidence intervals | validation-selected cases |
| Five seeds | `scripts/10_run_seed_reliability.py` | `config/seed_reliability.yml` | `seed_reliability_summary.csv` | explicit primary/historical metric names |
| Reference profiles | `scripts/11*` | `config/reference_framework_eval.yml` | `reference_profile_metric_drop.csv` | descriptive reference values and full-framework profiles |

`outputs/summaries/SOURCE_MANIFEST.csv` records the byte size and SHA-256 of every
tracked summary. The Protocol A evidence table records original matrix hashes and
stable evidence IDs without filesystem paths.
