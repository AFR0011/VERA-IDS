# C03 symmetric selector: approval and reference acceptance audit

**Decision date:** 2026-10-25  
**Decision:** User approved the symmetric scientific selector policy from the C03 v3 freeze-readiness review. This fixes the *scientific selection rules* prospectively. It does **not** authorize fitting models, modifying repository files, executing a full campaign, or viewing new test scores before selection is sealed.

## Verified scientific design

The paired condition is the availability of held-out-family validation observations to one deterministic selector. The two arms share the same trained candidate models, preprocessing, sampled training/validation population, known observations K, test population, and search/feasibility rules. The blind arm gets K; the visible arm gets exactly K plus held-out U. Both arms use the same known-only rejection constraints. The 3% budget is primary; 1%, 5%, and 10% are sensitivity conditions. Global rejectors search all unique finite gated-score thresholds with unsplit ties plus no-reject; family conditional searches 21 alpha values, with per-family quantiles estimated only on K. The objective is composed-system macro-F1 over the fixed seven-class universe (zero F1 for absent class).

This is a *new* symmetric selector. It is not the historical visible UDR-first tuner or Phase-3B maximum-rejection selector, and is not directly interchangeable with their numeric outcomes.

## Model-free acceptance tests performed

13 `unittest` cases passed on standalone synthetic data. Coverage includes:

- Fixed seven-class F1 and zero contribution for absent Unknown.
- Correct tie-aware Stage-1 AUROC; undefined AUC rejects the case.
- Stage-2 K-only F1 unaffected by adding U, with profile-selection reversal possible through Stage-1 AUC tie-breaking.
- Exact tied-uncertainty handling and inclusive global rejection threshold.
- No gated observations and explicit infeasibility without fallback.
- K-only feasibility budgets despite a large U population.
- Cumulative-confusion global search compared with a deliberately slow brute-force full-confusion oracle on 14 fixtures × 4 budgets = 56 scenarios.
- Family-conditional empirical quantiles and strict rejection comparison.
- Deterministic repeated selection, missing support, and unfrozen-budget failures.

On a disposable synthetic dataset with 700,000 validation observations and approximately 465,657 distinct gated uncertainty scores, `select_global()` took 1.76 seconds; 21-point family-conditional search took 1.56 seconds on the present container. These are developer-environment benchmarks, **not** user machine estimates or training runtime estimates. Test scores and historical traffic data were not read.

## Still unverified before execution freeze

1. Complete binding of candidate IDs to historical parameter dictionaries, preprocessing and Stage-1 threshold helper, including exact score transformations and seed offsets.
2. Cross-arm K row IDs/score digests, U exclusion in blind selector data access, deterministic seal of both decisions before any test-score access, and test-row identities.
3. Source file SHA-256, feature/proxy policy, independence limitations, software versions, and resource envelope.
4. Exact integration-test behavior for no feasible method, missing Stage-1 AUC, identical scores, different selected profiles, and tie-boundary parser semantics on real pipeline scoring arrays.
5. Explicit separately scoped model-fitting authorization.

**Disposition:** scientific selector policy approved; standalone model-free kernel acceptance passed; end-to-end pilot **not execution-frozen**. No repository write, experiment execution, or new test-result claim.
