# Risk Register

Workflow schema: `agentic-workflow/v2`
Project: VERA-IDS

## Active release risks

### RISK-101 — Native migration behavior drift

- Severity: High
- Status: Closed for release candidate
- Evidence: author-owned phase logic was migrated with byte-identical initial
  hashes, then statically wired to the package-local engine. All numbered CLI
  dry runs and imports pass under the locked Python 3.12 environment.
- Closure evidence: every CLI passed help/dry-run/synthetic checks; zero forbidden
  loader matches; independent tester verdict `PASS`.

### RISK-102 — Protocol A correction propagation

- Severity: High
- Status: Closed for release candidate
- Evidence: exactly 136 matrices/34 runs validate; exact RF strict gates pass;
  core, external, seed, reference, comparison, and figure surfaces regenerated.
- Closure evidence: exact checks, corrected artifacts, manifests, and independent
  recomputation passed.

### RISK-103 — Manuscript transformation

- Severity: High
- Status: Closed for release candidate
- Evidence: source hash unchanged; cleaned OOXML has zero revisions/comments and
  sanitized metadata; text projection matches; the 191-page PDF passed automated
  scans and every-page visual inspection.

### RISK-104 — Portable environment

- Severity: Medium
- Status: Mitigated locally; public CI pending
- Evidence: Python 3.12.10 locked installation and all heavy imports pass on
  Windows. The lock is limited to supported Windows/Linux environments.
- Closure gate: green Windows and Ubuntu GitHub Actions.

### RISK-105 — External publication

- Severity: High
- Status: Open until publication
- Evidence: no remote has been created or pushed in this active batch.
- Closure gate: one-root main, green CI, annotated tag, public release, and
  unauthenticated PDF download with matching SHA-256.

## Resolved decisions

- MIT software and CC BY 4.0 content boundaries approved by Ali Farrokhnejad.
- Protocol A supported-label metric approved with explicit historical field.
- UNSW-NB15 Protocol B restricted to invalid diagnostic status.
- Historical models and raw/prepared data excluded; no download claim.
- Threshold/rejection experiments separated into named lanes.
- Repository-edition PDF only; source DOCX excluded.
