# Reproducibility

## Portable release environment

Release `v2026.08` is locked for Python 3.12 on Windows and Linux in `uv.lock`.
It is a CPU-portable verification profile. Install and verify with:

```bash
uv sync --locked --python 3.12 --extra test
uv run python -c "import numpy,pandas,scipy,sklearn,xgboost,lightgbm,catboost,shap,numba,pyarrow,matplotlib,imblearn,docx,pypdf,reportlab,psutil,joblib,yaml"
uv run python -m pytest -p no:cacheprovider
uv run python scripts/verify_cli_workflows.py --all
```

Direct runtime dependencies are exactly pinned in `pyproject.toml` and
`requirements.txt`; transitive dependencies and Windows/Linux artifacts are
frozen in `uv.lock`.

## Historical experiment environment

The manuscript records this historical profile:

| Component | Historical value |
| --- | --- |
| OS | Windows 11 Enterprise 25H2 |
| CPU | Intel Core i9-14900KF, 24 cores / 32 logical processors |
| Memory | 64 GB, 4800 MT/s |
| GPU | NVIDIA RTX 4090, 24 GB |
| Storage | Samsung 990 Pro, 2 TB |
| Python | 3.12.10 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.2.0 |
| LightGBM | 4.6.0 |
| CatBoost | 1.2.10 |

This release does not claim a historical GPU rerun. The portable lock is a
separate, current execution profile for compilation, imports, unit tests,
synthetic workflows, and future bounded runs.

## Reproduction levels

1. Corrected aggregate results: fully reproducible from the tracked 136-matrix
   evidence table; no data or models required.
2. Synthetic workflows and package behavior: reproducible from a clean clone
   with the locked environment.
3. Full benchmark experiments: require provider-acquired data and compute. The
   workflow trains models locally; historical model binaries are unavailable.
4. Historical byte-identical model reproduction: not claimed.

The numbered CLI workflow is package-local. Dataset paths are repository-relative
or supplied through ignored local configuration. The absent cross-dataset helper
is not advertised.

## Determinism and selection

Seeds, row caps, support rules, threshold grids, and output roots are declared in
`config/`. Model/threshold selection uses validation data. Test labels are not
used for selection. The exploratory broad rejection grid and validation-selected
policy are named separately and must not be conflated.

Protocol A and Protocol B use different supported label sets. See
[PROTOCOL_A_CORRECTION.md](PROTOCOL_A_CORRECTION.md) and
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md).

## Excluded artifacts

The repository and release exclude raw/prepared data, row-level scores and
predictions, caches, logs, serialized models, and private paths. Benchmark data
must be obtained under provider terms. The omitted historical model collection
is approximately 63.9 GB and is not offered for download.
