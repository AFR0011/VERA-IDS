# Reckoning P1: provisional claim registry and evaluation contracts

Status: **implemented core contracts; scientific protocol NOT frozen**. This
implementation is bounded to the eight claim families C01–C08 approved on
2026-10-23. It does **not** approve new datasets, label mappings, feature
blacklists, source-group assignments, methods, operating thresholds, inference
rules, or full model runs. The original COMNET evidence is unchanged.

## Components

- `config/claim_registry.json`: eight **provisional families**, not yet atomic,
  estimable claims. Composite C04/C07/C08 must split into separately defined
  atomic claims before experiment freeze.
- `claim_registry.py`: validates types, identifiers and registry shape; blocks
  claims lacking exact metrics, denominators, fixed anchors, evidence, audit
  gates, and uncertainty definitions from being treated as frozen.
- `claim_eligibility.py`: aggregates **explicitly assessed**, evidence-linked
  claim-specific gates. Missing, restricted, failed and passed evidence remain
  distinct. No row-count threshold can establish source independence alone.
- `claim_comparability.py`: checks exact metric, denominator and label-set
  semantics, *one* authorized changed experimental factor, and fixed anchors.
  Test identity is separately recorded. The default test-population policy
  **requires identical population IDs and record fingerprints**, preventing a
  policy-only or method-only comparison from quietly changing test rows.
  Different test populations are explicitly permitted only when a caller
  selects `PopulationPolicy.ALLOW_DIFFERENT`, as appropriate for a
  justified partition, held-out-family, or cross-environment sensitivity
  study, and supplies complete test provenance. This never proves a causal
  one-factor effect. Observation-paired inference is
  not authorized by this helper. Explicitly requested pairing and matching
  population ID plus record fingerprint produce a **pairing metadata match**,
  not verified row-by-row label/identity alignment. Subsequent runners must
  verify alignment before any paired inference. Changing test populations
  can still be a comparable sensitivity analysis, but cannot be treated as paired.
- `claim_relations.py`: descriptive ordinal preservation/reversal/ties,
  categorical preservation/transitions/undefined cases, and signed magnitude
  deltas. Statistical significance is always **not assessed** at this stage.
  No practical-equivalence width is chosen here. Missing and undefined results
  are never encoded as zero performance or automatically unstable.

These modules deliberately do not reconstruct experiments from aggregate tables,
assert new scientific results, define independence from filenames or timestamps,
or turn a numerical sign reversal into an inferentially supported reversal.

## Test commands (repository root)

```powershell
uv run python -m pytest tests/test_reckoning_claim_framework.py -p no:cacheprovider
uv run python -m pytest -p no:cacheprovider
uv run python -m compileall -q src scripts tests
```

## Required next scientific specifications

Before full campaign execution: register metric and denominator definitions,
label-set semantics, precise comparison anchors and pairing policy per atomic
claim; audit source/label/proxy/group/split/support/selection evidence for the
relevant dataset and claim; specify uncertainty and multiplicity rules and
justify any prospective equivalence widths. Validate these in the P1 five-dataset
eligibility matrix, execute only approved P2 feasibility pilots, then perform
an explicit P3 configuration freeze. This module is not an experiment runner.

## Evidence boundary

Prior COMNET submission provenance explicitly states that its validation-visible
and validation-blind results are **not seed-matched causal estimates**, and its
CICIDS2017 partition comparison is **two constructions**, not five independently
regenerated partitions. Both are valid historical evidence only within their
original comparison semantics. New paired and repeated-partition claims require
new experiments. Historical Protocol A supported-label and Protocol B
true-Unknown metric semantics remain authoritative.
