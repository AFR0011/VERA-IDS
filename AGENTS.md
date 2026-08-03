# Repository Guidance

Workflow schema: `agentic-workflow/v2`
Project: VERA-IDS
Repository profile: mixed research / data-ML / software

## Purpose and authority

This repository is the public VERA-IDS release for trustworthy two-stage IDS
evaluation. `BLUEPRINT.md` defines the release batch, `DEV_STATE.md`
defines its state, actual code/config/result artifacts outrank summaries, and
`QA_REPORT.md`/`RISK_REGISTER.md` preserve verification and unresolved risk.

## Authority order

1. `AGENTS.md` and the accepted `BLUEPRINT.md`.
2. Actual code, configuration, protected source artifacts, and compact result bytes.
3. `DEV_STATE.md`, `QA_REPORT.md`, and `RISK_REGISTER.md`.
4. Public summaries and explanatory documentation.

## Operating rules

- Never commit raw/prepared benchmark data, row-level predictions, credentials,
  private identifiers, local paths, or untrusted serialized models.
- Do not alter model parameters, thresholds, seeds, splits, labels, metrics, or
  stored scientific values without a new approved experiment surface.
- Treat Protocol A, Protocol B, Stage-2, direct baseline, external, seed, and
  reference-profile metrics as distinct surfaces.
- Unknown-detection metrics require genuine true-Unknown support.
- Preserve historical evidence and record discrepancies in `ERRATA.md`.
- Advertised workflows use package-native implementations. Full scientific
  training remains expensive and requires users to obtain benchmark datasets;
  historical trained models are not distributed.

## Protected paths

- `../Codes/**` is the read-only protected research snapshot.
- `outputs/summaries/**` and `figures/**` are claim-bearing copied artifacts;
  modify only by a documented regeneration with source hashes.
- `.release-audit/**` is local internal audit evidence and must stay ignored.

## Required commands

```bash
uv sync --locked --extra test
uv run python -m compileall -q src scripts tests
uv run python -m pytest -p no:cacheprovider
uv run python scripts/verify_cli_workflows.py --all
uv run python scripts/recompute_protocol_a_metrics.py --check
uv run python scripts/verify_manifests.py
uv run python scripts/check_links.py
uv run python scripts/smoke_test.py
uv run python scripts/release_audit.py scan .
```
