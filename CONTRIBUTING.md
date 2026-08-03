# Contributing

Open an issue before changing a protocol, split, metric, label set, threshold,
seed, or claim-bearing result. Scientific changes require a new configuration,
new run identifier, preserved historical artifact, and an update to
`RESULTS_PROVENANCE.md` and `ERRATA.md` where applicable.

For ordinary code/documentation changes:

1. work on a branch;
2. keep data and machine-specific configuration untracked;
3. run `python -m pytest` and `python scripts/smoke_test.py`;
4. scan the candidate tree with `python scripts/release_audit.py scan .`;
5. update documentation and provenance with evidence, not assumptions;
6. never commit secrets, personal data, raw benchmark data, row-level scores, or
   untrusted serialized models.

No contribution can resolve licensing or institutional ownership by code change;
those decisions require the rights holder.
