# Shared Errors

- 2026-08-03: One optional heavy import test skipped because SciPy was not
  installed. Full ML execution remains unverified and is tracked as a risk.
- 2026-08-03: `PUBLIC-RELEASE-002` resolved the earlier local environment gap:
  the locked Python 3.12.10 environment installed and all retained heavy imports
  passed. Historical GPU training was not rerun and is not claimed.
