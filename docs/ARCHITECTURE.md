# Architecture

Numbered scripts load YAML and call public orchestration modules under
`ids_eval_framework.src`. Compute-heavy author-owned implementations live under
`ids_eval_framework._native` and are imported statically. The merged
`two_stage_engine.py` supplies preprocessing, calibration, thresholding, and
system evaluation shared by Protocol A and Protocol B.

Protocol A, direct baselines, Protocol B LOAO, rejection, external stress,
statistics, seed reliability, and reference profiles are separate lanes with
distinct configuration and output roots. Full outputs are ignored; only path-free
aggregate evidence and curated summaries are tracked.

Protocol A system evaluation emits a supported-label primary macro-F1 and an
explicit declared-output historical value. Protocol B refuses Unknown metrics
without true-Unknown support.
