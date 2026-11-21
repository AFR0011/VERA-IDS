# C03 fixed paired pilot — verified results

Completed 2026-11-18. Scope: CICIoT2023, prepared Protocol B day/file partition, Botnet held out, model seed 123 (Stage-2 seed 133). This completes one P2 feasibility pilot, not the full campaign.

Both validation arms selected `rf_class_weight_balanced` and Stage-1 threshold `0.9166666666666666`. The fitted models and ordered test population are shared. Only held-out-family information available to supervised validation selection differs: blind K versus visible same K plus U. The observed differences here arise in rejection-policy selection.

## Primary 3% validation budget

The budget applies to known validation rows in both arms. It is not a ceiling on total test rejection, which also includes genuine Unknown rows. All four policies were validation-feasible in both arms.

| Method | Macro-F1 blind | Macro-F1 visible | Visible − blind | Unknown detection blind → visible | False Unknown among known blind → visible |
| --- | ---: | ---: | ---: | ---: | ---: |
| max_confidence | 0.746844 | 0.888332 | +0.141488 | 0.0057% → 94.4434% | 0.0008% → 1.3453% |
| margin | 0.746951 | 0.885631 | +0.138680 | 0.1275% → 95.0150% | 0.0683% → 1.7644% |
| entropy | 0.747088 | 0.900668 | +0.153580 | 0.1122% → 97.6976% | 0.0034% → 0.2213% |
| family_conditional | 0.746825 | 0.906155 | +0.159330 | 0.0000% → 99.4612% | 0.0000% → 0.3237% |

Test support: 1,190,899 total rows; 58,858 benign; 955,530 represented attacks; 176,511 genuine Unknown (Botnet) attacks. Known-all FUR denominator is 1,014,388, including benign. Macro-F1 uses the fixed seven-class universe. Complete matrices, numerator/denominator pairs, benign error/rejection and raw Stage-1 AUROC/AP definitions are retained in the machine-readable receipt.

## All predeclared budget/method cells

| Known-validation budget | Method | Blind macro-F1 | Visible macro-F1 | Difference | Feasibility |
| --- | --- | ---: | ---: | ---: | --- |
| 0.01 | entropy | 0.747088 | 0.900668 | +0.153580 | feasible->feasible |
| 0.01 | family_conditional | 0.746825 | 0.906155 | +0.159330 | feasible->feasible |
| 0.01 | margin | 0.746951 | 0.756862 | +0.009910 | feasible->feasible |
| 0.01 | max_confidence | 0.746844 | 0.826802 | +0.079958 | feasible->feasible |
| 0.03 | entropy | 0.747088 | 0.900668 | +0.153580 | feasible->feasible |
| 0.03 | family_conditional | 0.746825 | 0.906155 | +0.159330 | feasible->feasible |
| 0.03 | margin | 0.746951 | 0.885631 | +0.138680 | feasible->feasible |
| 0.03 | max_confidence | 0.746844 | 0.888332 | +0.141488 | feasible->feasible |
| 0.05 | entropy | 0.747088 | 0.900668 | +0.153580 | feasible->feasible |
| 0.05 | family_conditional | 0.746825 | 0.906155 | +0.159330 | feasible->feasible |
| 0.05 | margin | 0.746951 | 0.885631 | +0.138680 | feasible->feasible |
| 0.05 | max_confidence | 0.746844 | 0.888332 | +0.141488 | feasible->feasible |
| 0.10 | entropy | 0.747088 | 0.900668 | +0.153580 | feasible->feasible |
| 0.10 | family_conditional | 0.746825 | 0.906155 | +0.159330 | feasible->feasible |
| 0.10 | margin | 0.746951 | 0.885631 | +0.138680 | feasible->feasible |
| 0.10 | max_confidence | 0.746844 | 0.888332 | +0.141488 | feasible->feasible |

Visible-minus-blind macro-F1 is positive in all 16 predeclared cells, with effects ranging from +0.009910 to +0.159330. These cells share data and fitted models; they are not 16 independent replications. No best method was selected from these test outcomes.

## Interpretation boundary

This is a conditional descriptive comparison on one fixed prepared-corpus holdout/seed. It demonstrates that validation-information availability changes this fixed evaluation result, while also changing false-unknown error. It does not establish population causality, source-independent generalization, robustness across seeds/holdouts, or a universal stability score. No confidence intervals, significance tests or practical-equivalence margin were introduced after seeing outcomes. Historical unsupervised preprocessing exposure to Botnet training features is retained; only supervised selection information differs between arms.

## Verification and resources

All 433 repository tests passed in 224.73 s; all nine required commands exited zero. The 22 focused tests include RF/XGB streaming equivalence, real isolated scoring, pre-seal access rejection, corruption, deadline/resource guards and recovery. Completion verification passed against the frozen contract, all producer dependencies and selected score checkpoints. All 32 arm/policy confusion matrices independently reconcile to their reported accuracy, macro-F1 and exact rate denominators.

Post-seal runtime: 330.005 s (5 min 30 s); peak sampled worker RSS 2.870 GiB; new artifacts 57.12 MiB. All sampled retained limits held. Available-RAM telemetry briefly reached 4,096 bytes, with five Stage-2 samples below 256 MiB across 1.83 seconds; that gate was removed by the user and was not silently reinstated. This run is not a claim that the host always has comfortable memory headroom.

The release scan retains existing content findings and adds a filename-only `supervisor.py` match, reviewed as process-supervision code. This is not a clean public-release certification.

## Evidence

- [Complete compact result and provenance](../experiments/c03/postseal/PILOT_SUCCESS.json)
- [Pre-execution verification](../experiments/c03/postseal/PROVENANCE.json)
- [Implementation and operational contract](RECKONING_C03_SCORE_PRODUCER_CONTRACT_DRAFT.md)
- Compact-result SHA-256: `cf8cb971f8c8b5588796c0e6aba6e219074d26bdf4b2e39c9f6a8c3f5c01dc2c`.
- Post-seal contract SHA-256: `1491a4b1215c68b0427cfda9ec493b9bc41ff0e140c4c6b429290c493e635319`.
- Frozen selection SHA-256: `72e8d4f8b41c84a639e56f21b6582cecec2a7841e3c50b4da9d90c34dacd871b`.
- Evaluation SHA-256: `2db0f4f382ba3d83db5ec5c9125fd59865490b890b5e5952f91b11d86ccda2d9`.
- Producer plan SHA-256: `6b54cfd10448ff4f4cf9cc601e3269debe14f95e78999b0378ffba955d057bdd`.

Raw/prepared data, local models, score arrays and resource traces remain outside the public repository. Historical release evidence and scientific values were not overwritten. The next master-plan work is the remaining P2 partition/method feasibility pilots and P1 claim-specific audit completion, followed by a reviewed P3 campaign freeze before expansion or article conclusions.
