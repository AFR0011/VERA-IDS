# Run and Verification Protocol

Workflow schema: `agentic-workflow/v2`
Project: VERA-IDS

## Environment

- The package declares Python 3.10 or newer; release checks used the local Python
  interpreter recorded by the command output rather than a fully locked ML
  environment.
- Install public dependencies with `python -m pip install -r requirements.txt`.
- Raw datasets are never installed or downloaded by tests.

## Verification ladder

1. `python -m compileall src scripts tests`
2. `python -m pytest`
3. `python scripts/smoke_test.py`
4. staged-content secret/path/data scan and tracked-file size review
5. repeat steps 1-3 from a temporary clean clone

## Expensive execution boundary

Dataset preparation and model training are not part of release validation. Full
experiment commands are documented only where evidenced by source, together with
their dataset, compute, and legacy-dependency requirements.

## Evidence recording

Record actual command output and limitations in `QA_REPORT.md` and
`PUBLICATION_CHECKLIST.md`. Unavailable checks remain explicit risks.
