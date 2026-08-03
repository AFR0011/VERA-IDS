# Experiment Protocol

Protocol A is a closed-set stratified train/validation/test experiment. Protocol B
is support-audited day/file LOAO with a held-out family mapped to true `Unknown`.
Preprocessing fits on training data, threshold/tau/calibration/model selection is
validation-only, and the test split is evaluated after selection. Seeds, support
rules, row budgets, and grids are in `config/`. Distinct result surfaces must keep
their explicit ordered label sets and selection discipline.

The exact run order and limitations are in `README.md` and `REPRODUCIBILITY.md`.
