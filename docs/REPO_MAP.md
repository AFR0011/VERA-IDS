# Repository Map

- `src/ids_eval_framework/`: installable package, including private native lane modules.
- `scripts/`: numbered CLIs and deterministic verification/regeneration tools.
- `config/`: repository-relative experiment settings and explicitly named lanes.
- `tests/`: dataset-free metric, support, release-boundary, and provenance tests.
- `outputs/evidence/`: 136 path-free Protocol A confusion matrices.
- `outputs/summaries/`: compact aggregate result surfaces and hash manifest.
- `figures/`: aggregate release figures.
- `release/`: tracked metadata for the untracked GitHub release asset.
- `docs/`: architecture, protocols, claims, environment, and governance support.
- `.github/workflows/`: locked Windows/Linux verification.

Raw/prepared data, row-level predictions, models, private configuration, release
staging, and audit evidence are excluded by `.gitignore` and release scans.
