# Security Policy

Report a suspected credential, personal-data exposure, unsafe artifact, or code
execution issue privately to the future repository maintainer. No public security
contact address has been authorized yet; configure one before publication.

## Trust boundaries

- Raw datasets and generated CSV/JSON content are untrusted inputs.
- Pickle/joblib model files can execute code during deserialization. Do not load
  untrusted artifacts; this release includes none.
- Legacy phase scripts are dynamically imported only for full runs and are not
  bundled. Dry-run mode does not import them.
- Configuration files must not contain credentials or private machine paths.
- Dataset network-address features are research data, not operational targets;
  do not confuse benchmark values with live infrastructure.

Use `python scripts/release_audit.py scan .` before each release, then run a
dedicated secret scanner and review Git history, not only the working tree.
