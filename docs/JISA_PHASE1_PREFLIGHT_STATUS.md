# JISA Phase 1 Preflight Status

Status: READY FOR LOCAL PREFLIGHT
Date: 2026-09-07

Phase 0 is closed. The local evidence audit confirmed that the repeated-seed CICIoT2023 Protocol-B surface retains aggregate run artifacts and preprocessors but not the row-level probability outputs or trained classifier estimators needed for a blind operating-point replay. The blind lane will therefore require targeted reruns later.

Before any Phase-1 model fitting, the local Protocol-A prepared partitions must pass the non-destructive schema/row-count preflight implemented in `scripts/jisa_phase1_preflight.py`.

No model training should begin until the preflight reports `PASS=True` for both CICIDS2017 and CICIoT2023 across train/validation/test.
