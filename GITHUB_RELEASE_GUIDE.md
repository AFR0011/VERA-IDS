# GitHub Release Guide

The repository must remain local until every blocking item in
`PUBLICATION_CHECKLIST.md` is resolved. The approved public destination is
`https://github.com/AFR0011/VERA-IDS`.

## Recommended metadata

- Name: `VERA-IDS`
- Description: `Validity-aware, support-audited evaluation of two-stage ML intrusion-detection experiments.`
- Topics: `intrusion-detection`, `machine-learning`, `open-set-recognition`,
  `reproducibility`, `cybersecurity`, `cicids2017`, `ciciot2023`

## Create and inspect the GitHub repository

Create an empty repository in the intended account, with no generated README,
license, or `.gitignore`. Confirm the visibility setting before adding a remote.

```bash
git status --short
git log --oneline --decorate -1
git ls-files
git remote -v
git remote add origin https://github.com/AFR0011/VERA-IDS.git
git remote -v
git push -u origin main
```

Do not run the remote/push commands until license, ownership, scientific, and
privacy review is complete. Push only the reviewed one-commit `main` branch.

## Repository features

- Enable Issues after a public professional contact and security-reporting route exist.
- Enable Discussions only if the author wants a research Q&A venue and can moderate it.
- Add branch protection for `main` and require tests/secret scanning for pull requests.
- Configure GitHub's secret scanning and dependency alerts.

## First tagged release

After resolving every gate and observing green CI on `main`:

```bash
git tag -a v2026.08 -m "VERA-IDS v2026.08"
git push origin v2026.08
```

Publish the release and attach only `VERA-IDS-Thesis.pdf`. Do not attach the
source DOCX, datasets, models, or historical archives. Verify the public download
against `release/ASSET_MANIFEST.json` from an unauthenticated URL.

## Optional Zenodo DOI

Connect the GitHub repository to Zenodo, enable the repository, then create the
GitHub release. Zenodo can archive that release and mint a version DOI plus a
concept DOI. After receipt, add the DOI and repository URL to `CITATION.cff`,
README citation instructions, and release notes, then tag a metadata correction
if needed. Verify that the archived bundle does not include ignored/private data.
