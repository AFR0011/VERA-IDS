# C03 paired pilot: integrated validation-selector runner core

**Status:** Source-bound no-fit integration implementation, **not** an authorized model-fitting runner. Prepared 2026-10-25.

## What was established by the user's latest verification

`c03_rf_stage2_sharing_gate.py` returned **PASS** using the actual historical model helper and training derivation. The two Random Forest profiles share Stage-2 hyperparameters, class weighting and seed 133 by design; the synthetic 60-row Stage-2 CSR feature array, labels and weights matched exactly. Altered features, labels, weights, random seed, and row identity were each detected by five negative controls. This is **not** proof of the real 1,201,422-row Stage-2 feature tensor or fitted RF numerical identity.

Across the previous four user-local component suites, **37/37** tests had passed. None of those tests or this gate trains actual models.

## Added integrated code

`c03_paired_runner_core.py` is a prospective integrated *selection* component which:

- Validates the pinned Git commit, user-approved scientific-policy SHA-256, historical seed123/Botnet aggregate containing exactly three candidate profiles, and saved prospective source-inventory manifest. `--verify-bytes` also recomputes current SHA-256 for all 369 processed files and eight originally inventoried code/policy files.
- Imports the actual historical Stage-1 family-aware chooser, with the frozen 2 × 2 search combinations and search parameters.
- Accepts **one common ordered validation truth vector** and 3 candidate score pairs (Stage-1 attack probabilities and Stage-2 five-class probabilities), refusing incomplete, malformed, nonfinite, mis-normalized or mis-supported bundles.
- Computes candidate selection on shared represented-family Stage-2 K macro-F1 with accessible-view Stage-1 AUROC as approved tie-breaker, Stage-1 threshold separately on K and K+U, and all four approved rejectors at the four frozen rejection budgets. Constraints always use K, while fixed-seven-label macro-F1 selection uses the arm's accessible validation view.
- Produces a content-digested two-arm selection record and fails closed on incomplete arms, missing methods/budgets or post-seal changes. The included **synthetic** guarded-reader interface refuses reads before seal. There is no real test reader in this module.

`test_c03_paired_runner_core.py` is the integrated synthetic acceptance suite: 10 tests. On the author's isolated container **9 passed and 1 was skipped** because the actual VERA-IDS repository is not present. The skipped test runs the integrated selection path with the actual pinned Stage-1 source on the user's machine, so the user's local run is required to establish it.

## User-local commands (PowerShell at VERA-IDS root)

Put `c03_paired_runner_core.py` and `test_c03_paired_runner_core.py` beside existing C03 scripts. Do **not** commit yet.

```powershell
python -B c03_paired_runner_core.py --preflight --verify-bytes
python -B -m unittest -v test_c03_paired_runner_core.py
```

Both commands are read-only. `--preflight --verify-bytes` hashes current source file bytes without loading model observations, and the unit suite generates only in-memory synthetic data. Neither command fits a preprocessor, trains a model, opens the real test Parquet rows, computes test metrics, writes result artifacts, or mutates Git.

## Remaining components, before any real fit

1. **Preprocessing/model-fit worker:** implement historical full-train preprocessor, exact train cap and validation cap, strict supervised LOAO, exactly five fitted components with one shared Stage-2 RF model, validation score generation/checkpoints and per-tensor/row-index hashes. The two RF references must literally reuse one verified stored Stage-2 tensor and fitted component. Do not import the historical `run_one_combo` as a runner because it reads test before selection.
2. **Production selection seal:** persist signed content/selection/runner hashes atomically in a separate output namespace; ensure the scorer passes all three sets of validation-only scores to this core; disallow any production test-loader import/access until both arms are sealed.
3. **Post-seal test evaluator:** separate stage, with fresh verification of selection digest/policy/score-provenance and a single full ordered test population for both arms. Fail closed when infeasible, do not impute scores; export per-method and per-budget traces, exact row/label denominators and signed visible-minus-blind effects.
4. **Memory feasibility:** 15.4 GiB physical RAM and previously observed 4.3 GiB available are **not cleared** for training the large unbounded 300-tree RF models. Need a separately approved bounded feasibility probe with resource abort limits or a larger-RAM machine. Do not change estimators, depth, sample cap, weighting, family universe or model kind to fit hardware without formal protocol amendment.
5. **Execution approval:** The user explicitly approved the *scientific selector policy*, not model fitting. An actual C03 experiment must be separately authorized once end-to-end runner and resource conditions are reviewed. Rehash the newly introduced runner and source files as additional prospective provenance; the earlier input manifest does not magically include these new files.

**Interpretation limit:** Fixed prepared CICIoT2023 environment, verified disjoint legacy raw CSV file-path assignments. The 169 raw part files share a middle UUID; independence of physical captures/attack executions/flows and freedom from numeric-feature proxies remain unestablished. Neither 37 prior synthetic tests nor the new core warrants new scientific performance claims.
