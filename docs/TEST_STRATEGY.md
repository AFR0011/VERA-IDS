# Test Strategy

Release tests are deliberately dataset-free. They cover package/config path
resolution, explicit label-set metrics, Unknown-support guards, canonical support
rule mapping/conflict detection, tiny support filtering, summary provenance, and
absence of raw/model/local-path material. `scripts/smoke_test.py` writes a small
ignored JSON result. Heavy imports, training, dataset preparation, and result
recomputation require the unresolved full environment and legacy dependencies.
