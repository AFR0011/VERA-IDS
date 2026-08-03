# QA Report

Workflow schema: `agentic-workflow/v2`
Project: VERA-IDS
Batch: `PUBLIC-RELEASE-002`
Tested: 2026-08-03

## Verdict

- Independent tester verdict: `PASS`.
- Local candidate recommendation: `READY FOR PUBLICATION`.
- Full historical GPU training was not rerun and is not claimed.

## Recorded evidence

- Locked CPython 3.12.10 environment: 47 packages; all retained heavy imports
  passed.
- Compilation passed; pytest: 26 passed.
- All 14 advertised CLI surfaces passed `--help`, `--dry-run`, and bounded
  deterministic `--synthetic` execution.
- Protocol A: 136 matrices, 34 runs, four policies per run; evidence SHA-256
  `1C34102C3BDC593896E8C20ED1579631AC44B57AE3E7FFFA46DC3F5C249B17B5`.
- Exact RF strict gates: CICIDS2017 `0.818902771736530`; CICIoT2023
  `0.897154513910465`.
- Protocol B tests retain genuinely supported `Unknown` and reject zero-support
  Unknown metrics.
- Manifests, Markdown links, smoke workflow, static runtime scan, and public
  release scan passed with zero findings.
- Protected `../Codes/**` before/after inventories match at 5,102 files and SHA-256
  `EC91903C45FB950EA62796DC7B3A9AA79AAE61FB220DD9AF562E89B30BFD1046`.
- Authoritative DOCX SHA-256 remains
  `534D1346FC6AF42A93FA9893C9D488AB665CEE4863752B41647C134F569C00CC`.
- Cleaned DOCX has zero revisions/comments and sanitized Office metadata.
- Final PDF: 191 pages, 3,803,300 bytes, SHA-256
  `14DF7A720CC989F1DD5BE515B5F919A11CE25654C0C91B4167A60C7C48720306`;
  every page was visually inspected after correcting the cover layout.

## Clean-clone evidence

- A single-branch clone of the one-root candidate installed the 47-package lock,
  passed heavy imports, compilation, 26 tests, all CLI dry-run/synthetic checks,
  exact correction checks, manifests, links, smoke, and the zero-finding scan.

## Remaining external gates

- Windows and Ubuntu GitHub Actions.
- Annotated tag, release asset, and unauthenticated public download/hash check.
