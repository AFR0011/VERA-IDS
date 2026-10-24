# Reckoning C06 — historical budget-feasibility derivation

**Status:** P1 historical-source derivation implemented; eligibility and full experimental protocol **not frozen**. No new model training, threshold selection, data preparation or experimental results.

## Scientific contract

**Question.** For each existing support-eligible Protocol B LOAO held-out family, does a rejector's historical validation-feasibility category change when the allowed **overall validation rejection budget** is raised from 1% to 3%, 3% to 5%, or 5% to 10%?

**Observed category.** `feasible` if and only if that family is present in the frozen row's `feasible_holdouts` set for its dataset, rejector and budget; otherwise `infeasible`. This is a category of the original **validation-constrained selection** result, not a dataset validity category. It is not an estimate of test rejection rate, test accuracy, or unknown detection.

**Factor and anchors.** The sole declared changed factor is `validation_rejection_budget`. The required fixed anchors are dataset, Protocol B LOAO, validation-visible information, held-out family, rejector, prepared partition, model/profile and candidate grid, and validation population. The released aggregate proves only dataset, method, family membership, and budget fields. **Prepared partition, model configuration/candidate grid, and validation population identities remain unverified at this source level.** The derivation must therefore not set scientific eligibility to `admissible` or call existing formal relation utilities with fabricated test/validation identity hashes.

**Evidence.** The source is `release/v2026.10/raw/visible_budget_feasibility_coverage.csv`, Git blob SHA `3a6c388a544f8d028b5c1223cae58d0b5ef1dc51`. The script validates the Git blob content, accommodating only Git line-ending conversion, and checks 32 original aggregate cells, dataset-specific held-out family names, membership/count/fraction consistency, uniqueness, complete budget coverage, and monotone feasibility sets as budgets relax. Nonmonotone source summaries cause a failure requiring manual source/selection audit, not automatic repairs.

**Unit and denominator.** There are two historical datasets × four rejectors × six holdouts × four budgets = **192 derived descriptive condition categories** from **32 source aggregate rows**, not 192 independent model evaluations. Three adjacent-budget comparisons for each dataset × rejector × holdout yield **144 descriptive transitions**, not 144 statistical tests. A summary has eight dataset/rejector rows. Across all 144 historical adjacent-budget descriptive transitions: **108 remained feasible, 15 remained infeasible, and 21 became feasible**. These are set-membership transitions, not statistical tests. `feasible` denominators are six original row-count-supported LOAO holdouts **per dataset**; no third dataset, new source-validity assessment or independence claim is inferred.

**Interpretation.** A `became_feasible` transition describes an existing aggregate set-membership change. `remained_feasible`/`remained_infeasible` are descriptive preservation. A disappearing feasible holdout under a relaxed budget violates the fixed-grid nesting expectation; the implementation rejects that source condition. The original comparisons supply neither independently sampled source environments nor a justified seed-based confidence interval for category transitions. Do not attach p-values, effect-size margins or inferential stability labels.

## Existing restrictions

CICIDS2017 recovered Protocol B: 20 disjoint raw-row units, but four source-day files contribute to train/validation/test and the legacy audit found nonzero cross-split exact processed feature hashes. Its historical feasibility cells are preserved, while source/overlap interpretation remains restricted. The group-safe alternative's generating implementation remains unverified. CICIoT2023 has not yet passed the new symmetric five-dataset source, label, shortcut, identity and collection-group audit. For **both** datasets, the resulting CSVs therefore say `eligibility_status=not_assessed` and `formal_relation_status=not_assessed`; CICIDS additionally records its known restriction.

These categories concern the validation-visible budget-coverage results reported in COMNET §4.4 Figure 4, not the separately validation-blind 3% comparison in Figure 3 or the joint-feasible 3% performance comparison in Table 4. In particular, a feasible validation condition need not remain within budget on test traffic.

## Run and verify (repository root)

```powershell
uv run python scripts/18_derive_c06_budget_feasibility.py --check-only
uv run python -m pytest tests/test_c06_feasibility.py -p no:cacheprovider
uv run python -m pytest -p no:cacheprovider
uv run python -m compileall -q src scripts tests
uv run python scripts/18_derive_c06_budget_feasibility.py
```

Only the last command writes outputs, into the existing ignored `outputs/17_revision_campaign/stability/` directory:

- `c06_validation_feasibility_observations.csv` (192 historical categories)
- `c06_validation_feasibility_adjacent_transitions.csv` (144 descriptive, not inferential, transitions)
- `c06_validation_feasibility_summary.csv` (8 source-consistent method/dataset summaries)
- `c06_evidence_manifest.json` (pinned source, limitation/eligibility metadata)

No frozen evidence file, `config/claim_registry.json`, preparation code, or model run is modified. These files are **staging evidence derivatives**, not the campaign-wide final `conclusion_stability_records.csv`. Inclusion into the latter requires explicit scientific gate assessments and complete comparison anchors.
