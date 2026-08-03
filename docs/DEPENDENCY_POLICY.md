# Dependency Policy

`uv.lock` is the authoritative portable CPU lock for CPython 3.12 on Windows and
Linux. `.python-version` selects Python 3.12.10, and the release gate installs the
locked environment and imports every heavy dependency. `requirements.txt`
mirrors the direct runtime pins for tools that do not consume `uv.lock`.

The historical GPU profile is Windows 11, Python 3.12.10, RTX 4090,
scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.6.0, and CatBoost 1.2.10. It is a
provenance record, not a claim that the historical experiments were rerun on the
release machine.
