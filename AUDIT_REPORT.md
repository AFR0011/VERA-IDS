# Public Release Audit

Release candidate: `v2026.08`
Copyright holder: Ali Farrokhnejad

## Scope

The public tree contains original software, configurations, tests, documentation,
compact aggregate summaries, 136 path-free Protocol A confusion matrices, and
four aggregate figures. It excludes raw/prepared benchmark data, row-level
predictions, serialized models, caches, logs, private paths, and the source DOCX.

## Rights

- Software: MIT.
- Original documentation, figures, aggregate results, correction supplement,
  and release manuscript PDF: CC BY 4.0.
- Third-party datasets and dependencies: not redistributed; upstream terms apply.

## Standalone implementation

Advertised CLIs call statically imported package-local modules. The public tree
contains no runtime phase-script loader or dependency on the protected research
layout. The missing cross-dataset helper and superseded zero-result lane are not
advertised. Reference-profile commands are steps 11/11b.

## Scientific correction

The Protocol A aggregate evidence contains 136 matrices across 34 runs and four
policies per run. It contains ordered labels, integer counts, stable identifiers,
and source hashes without paths. Exact corrected gates pass at
`0.818902771736530` and `0.897154513910465`. Protocol B retains supported
`Unknown`; invalid external Protocol B evidence is diagnostic only.

## Environment

The current release uses a locked Python 3.12 Windows/Linux CPU-portable profile.
The historical RTX 4090 environment is separately documented and was not rerun.
Historical models (approximately 63.9 GB) are unavailable.

## Gate status

Licensing, scientific correction, standalone CLIs, locked environment,
manuscript preparation, protected-input integrity, and independent verification
have passed locally. The exact one-root clean clone also passed the full locked
suite. GitHub Actions, annotated tag, release publication, and unauthenticated
download remain external publication gates.
