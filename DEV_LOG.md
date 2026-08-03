# Development Log

Workflow schema: `agentic-workflow/v2`
Project: VERA-IDS
Repository profile: mixed
Initialized: 2026-08-03

## 2026-08-03 - Repository bootstrap

- Classified as `mixed` with traits: research, data-ml, software.
- Created missing governance files without modifying product/source artifacts.
- Classification evidence is recorded in `docs/REPO_PROFILE.md`.

## 2026-08-03 - PUBLIC-RELEASE-001

- Inventoried the protected source tree: 5,063 files and 1,118,742,194 bytes.
- Repeated the inventory after preparation; the inventory SHA-256 remained
  `234167c2087e2126322dd45bffee8e1e05a7a27b36338b9f503137ae831b6aba`.
- Curated public code, configuration, documentation, figures, and 14 compact
  result summaries into this separate repository.
- Excluded raw/prepared data, serialized models, caches, row-level outputs,
  local-path-bearing run records, and unsupported bulk generated artifacts.
- Added repository-relative path handling, lightweight metric definitions,
  manifest verification, release scanning, synthetic tests, and a smoke path.
- Verified 21 tests passed and one optional heavy import was skipped because
  SciPy was unavailable. Compilation, smoke, content scan, manifest checks, and
  temporary clean-copy validation passed.
- Independent tester verdict: `PASS_WITH_RISKS`.
- Publication verdict remains `NOT READY FOR PUBLICATION` pending the human
  decisions recorded in `PUBLICATION_CHECKLIST.md` and `RISK_REGISTER.md`.

## 2026-08-03 - PUBLIC-RELEASE-002

- Established MIT software and CC BY 4.0 content/manuscript boundaries.
- Migrated advertised workflows to package-native implementations; removed the
  dynamic legacy loader, unavailable cross-dataset helper, and superseded
  paper-reproduction lane; renumbered reference commands to 11/11b.
- Added separate rejection lanes and per-CLI dry-run/synthetic verification.
- Recomputed Protocol A primary supported-label macro-F1 from 136 path-free
  matrices and regenerated summaries, comparisons, manifests, and figures.
- Locked the portable Python 3.12 environment and added Windows/Ubuntu CI.
- Built and inspected the 191-page repository-edition manuscript PDF without
  modifying the authoritative DOCX.
- Protected inventory hashes match; independent tester verdict: `PASS`.
- A single-branch clone of the one-root candidate passed locked installation,
  heavy imports, compilation, 26 tests, every CLI in dry-run/synthetic modes,
  exact correction, manifests, links, smoke, and the public scan.
