# Blueprint

Workflow schema: `agentic-workflow/v2`
Project: VERA-IDS
Repository profile: mixed research / data-ML / software

## Active batch

- Status: ACCEPTED
- Batch ID: `PUBLIC-RELEASE-002`
- Owner and sole product writer: root
- Target: public GitHub release `AFR0011/VERA-IDS` tag `v2026.08`

### Objective

Resolve the approved publication blockers and produce a standalone,
scientifically corrected, dual-licensed release. Publish only after local,
clean-clone, manuscript, environment, scan, GitHub CI, and public-download gates
all pass.

### Authoritative decisions

- Ali Farrokhnejad is the copyright holder.
- Original software is MIT licensed.
- Original documentation, figures, aggregate results, correction supplement, and
  release manuscript PDF are CC BY 4.0 licensed.
- Protocol A primary system macro-F1 averages supported labels only. Predictions
  of `Unknown` still count against the supported true class. The prior declared-
  output-label value remains explicitly historical.
- Protocol B retains genuinely supported `Unknown`; the invalid UNSW-NB15
  Protocol B experiment is diagnostic only.
- The manuscript-only split-disclosure boundary is preserved.
- The superseded zero-result lane is removed. Reference-profile commands occupy
  steps 11/11b.
- Raw data, row-level outputs, and historical models are not published.
- The authoritative DOCX remains unchanged. Only a cleaned, repository-edition
  PDF named `VERA-IDS-Thesis.pdf` is attached to the GitHub release.

### Intended files

All tracked publication artifacts inside this repository: governance, licenses,
package and CLI implementations, configurations, compact aggregate evidence,
corrected summaries and figures, tests, CI, documentation, citation metadata,
and the tracked release-asset manifest. Ignored audit evidence may be written
under `.release-audit/`.

### Allowed adjacent files

- Read-only access to `../Codes/**` and the authoritative workspace DOCX.
- Temporary manuscript copies, renders, environments, and clean clones under
  `.release-audit/` or an operating-system temporary directory.
- The public GitHub repository, Actions runs, tag, release, and PDF asset only
  after all pre-publication gates pass.

### Out of scope

- Mutating protected research files or the authoritative DOCX.
- Retraining or claiming a rerun of the historical GPU experiments.
- Publishing data, predictions, models, the DOCX, private paths, credentials, or
  pre-release Git history.
- Restoring the missing cross-dataset helper or the superseded paper lane.
- Presenting invalid UNSW Protocol B as comparative evidence.
- Publishing while any mandatory gate is failed or unavailable.

### Acceptance criteria

1. Protected before/after hashes match exactly.
2. License boundaries and third-party exclusions are explicit.
3. Advertised CLIs use package-local implementations; no dynamic legacy loader,
   protected phase path, or machine-specific path remains.
4. Exactly 136 path-free Protocol A matrices deterministically reproduce all
   corrected surfaces, including CICIDS2017 RF strict
   `0.818902771736530` and CICIoT2023 RF strict `0.897154513910465`.
5. Protocol A emits primary and historical metric columns; Protocol B tests keep
   supported `Unknown` in its label set.
6. All affected summaries, comparisons, provenance, and figures are regenerated;
   `PROTOCOL_A_CORRECTION.md` documents the correction without changing the thesis.
7. A Python 3.12 portable lock installs and imports every retained dependency;
   compile, unit, synthetic CLI, manifest, link, and release-scan checks pass in
   the worktree and a clean clone.
8. The cleaned manuscript contains no revisions, comments, stale metadata,
   placeholders, local paths, or stale `V5`; every PDF page is visually checked.
9. The asset manifest exactly records PDF/DOCX hashes, size, pages, license, and tag.
10. Public history has one root commit on `main`; CI is green; annotated tag and
    release exist; an unauthenticated asset download matches the tracked hash.

### Stop policy

Any failed or unavailable scientific, standalone-CLI, Python 3.12, manuscript,
licensing, scan, clean-clone, GitHub CI, history, or public-download gate sets the
cycle to `STOP_NEEDS_HUMAN`. No failed gate may be reclassified as completion.

### Rollback

Before publication, restore or inspect the previous candidate from the verified
ignored Git bundle and abandon the new root. After public disclosure, do not
delete, retag, withdraw, or rewrite the remote without new explicit direction.

### Evidence required for done

Protected inventories; native-migration scan and provenance; 136-matrix manifest
and recomputation transcript; summary/figure hashes; Python 3.12 lock/install and
heavy imports; worktree and clean-clone verification; manuscript cleanup, text,
metadata, and page-inspection evidence; verified pre-public history bundle;
single-root Git evidence; green Actions; public URLs; and downloaded PDF hash.
