# C03 paired validation-information pilot: freeze-readiness review v3

**Status:** reviewed proposal; **NOT FROZEN**, **NOT APPROVED FOR TRAINING**  
**Date:** 2026-10-24  
**One-cell feasibility pilot:** CICIoT2023, Protocol B day-file split, held-out Botnet, model seed 123  
**Basis:** canonical Reckoning master campaign plan, C03 claim matrix, historical Phase 3A/3B source and local read-only preflights.

## 1. Evidence and disposition

The pilot's three historic Stage-1 strict-LOAO candidates are `xgb_inv_family_clipped`, `rf_inv_family_clipped`, and `rf_class_weight_balanced`; their parameter dictionaries are recorded in the historical seed-123 candidate aggregate, not to be manually retyped. The old Stage-2 winner for this cell was tied (`stage2_margin=0`, `unique_stage2_winner=False`), and the historical blind archive fitted only the top two candidates. Historical-winner reuse is **not permitted** in the new paired pilot.

Current processed train/validation/test splits have 3,676,338 / 1,200,241 / 1,190,899 rows, respectively. The historical bounded sample had 1,500,000 training rows, 700,000 validation rows, 599,278 known (K) and 100,722 held-out Botnet (U). A read-only audit confirmed zero Stage-1/Stage-2 label mismatches over all 1,190,899 ordered current test rows and the 599,278 ordered known-validation rows reconstructed with the original sampling procedure. These do **not** show historical feature-byte equality, raw-flow independence, or source-group-safe partitioning.

The JISA Phase-3 historical global preflight `PASS=False` followed from nonunique Stage-2 winners; its other candidate matrix, strict LOAO and processed-source gates were true. That failure prohibits the winner-reuse shortcut, not the new paired experiment.

**Review decision:** a one-cell controlled *feasibility* pilot is methodologically defensible **if** the constraints below are frozen before fitting, post-sampling support and source provenance checks pass, and model fitting is separately authorized. A one-cell result is not a population-level causal estimate or a generalization claim.

## 2. Intervention and controlled condition

Define, for rejector method *m*, known-traffic validation rejection budget *b*, and test metric *M*:

\[
\Delta_M(m,b)=M_{\mathrm{test}}(S(K\cup U;m,b))-M_{\mathrm{test}}(S(K;m,b)),
\]

where **S is one common deterministic selection procedure** and **K** is the exact same row-ordered benign-plus-represented-family subset in both arms. **U** contains held-out Botnet validation observations. Their labels become system `Unknown` only in the visible arm. The blind arm does not receive U features, labels, scores or aggregate statistics during selection. This is *validation-observation information availability*, not merely a label-mask intervention.

All other inputs are shared: source byte snapshot; full three-candidate definitions; split and sampled training rows; preprocessor; supervised model fitting and RNG seeds; scored candidate tensors; evaluation test rows; labels/taxonomy; rejection method and numerical operations. Results estimate the performance contrast **under S** for this fixed seed, holdout and observed test surface. Do not portray the contrast as the historical visible-minus-blind experiment or as full inductive unknown-family blindness.

## 3. Fit and score once, select twice

1. **Before any model fit:** lock full file-content hashes (not path/size inventories), loader order, feature set, environment/library versions, exact canonical candidate IDs/parameters, split identities and support ledger. Reconstruct train 1,500,000 and val 700,000 samples once with historical per-file sequential-mixing RNG (seed 123 for train, 124 for val); keep stable local source file/row provenance where obtainable. Check K/U counts and per-family support on the capped samples.
2. **Preprocessing:** fit one shared historical transformation from the original full training-part stream, using that helper's separate statistics cap and feature inference procedure. The historical preprocessing was exposed to features of the held-out family's training rows; record this as an unsupervised-transductive limitation. It is not an H-excluded representation.
3. **Supervised fitting:** for all THREE profiles, fit Stage 1 after dropping Botnet attack training rows, Stage 2 using represented attack families only. Preserve the original seed offsets and weight semantics. Fit each unique model component once, with exact-feature/label/parameter/RNG identity checks before any caching across profiles. Neither selection arm triggers model re-fitting.
4. **Validation scoring:** score all 700,000 validation observations once per fitted candidate. Form K and U by one preserved mask and construct two read-only selector views. Save candidate-level score/sampling digests; verify that K predictions/probabilities are identical in both views.
5. **Select and seal:** run the *same versioned selector implementation* on the two views, then write two immutable selection traces, eligibility/feasibility outcomes, selected candidate profiles and Stage-1/rejector parameters. No test score, test truth, result, or diagnostics accessible to any selection operation; test evaluation code runs only after both choices are sealed and hashes verified.
6. **Test once per selected policy:** score the same ordered 1,190,899 test rows for each distinct selected profile, cache scores by candidate identity, compute final predictions separately for both arm policies and each method/budget; record signed visible-minus-blind differences where both arms are feasible. If selected profiles differ, selected-profile score arrays should differ, **but test source row identities must not**.

## 4. Unified selector proposal, corrected for historical search spaces

**A. Candidate profile.** For each candidate, compute represented-family Stage-2 validation macro-F1 using the same K known-attack rows (not U). Select highest value, breaking exact or tolerance-level ties (proposed absolute `1e-12`) on Stage-1 AUROC using the arm's accessible benign/attack validation labels, then ascending canonical candidate ID. The visible arm may use H attacks for its Stage-1 AUROC tie-break; this is the controlled information intervention. AUC undefined ⇒ explicit eligibility failure, not conversion to zero. Important: the historical `stage2_margin=0` makes this tie-break active for the pilot.

**B. Stage-1 threshold.** Apply the same historical family-aware threshold selector over each arm's accessible validation rows: FPR target candidates 0.01 and 0.02; minimum family support 100; objectives minimum family recall and 10th-percentile family recall; 40-point sweep; best objective, then attack TPR. Record per-class support for the sweep, selected threshold and all tie details. A visible unknown/held-out H is treated as an attack for the Stage-1 selection, but never admitted to supervised training. Same mathematical selector in both arms.

**C. Rejection methods and operating policies.** Four methods: max-confidence, margin, normalized entropy and family-conditional rejection. Apply only on Stage-1 attack-gated observations. For global methods, enumerate **all distinct finite uncertainty scores on the arm's accessible gated validation rows**, including the `k=0` no-rejection point. Sort descending, maintain stable tie groups and evaluate only group ends: *never split equal uncertainty scores*. The historical rejection tests are `uncertainty >= threshold`, including equality. Family-conditional candidates use `alpha ∈ {0, 0.005, ..., 0.10}` (21 values), with class-conditional empirical quantiles calculated using *K represented-family truth labels only* in **both arms**. Apply family rejection with `fam_pmax < class_threshold`, a strict inequality.

**D. Feasibility (fixed known-only control).** For each candidate operating point, compute the following limits only over exactly the same K rows in both arms: overall rejection on K ≤ b (primary 0.03; descriptive sensitivity b=0.01/0.05/0.10); benign-to-represented-family FP rate ≤ 0.02; benign rejection ≤ 0.10; false-unknown among K ≤ 0.05; false-unknown among represented attacks ≤ 0.10. Stage-1 gate is arm-specific; K denominator rows are identical. A visible arm is **not** granted a larger known-traffic error budget just because U is visible. Test overall rejection is a separate reported **realized** metric, not the validation budget.

**E. Unified objective.** Among feasible points for each arm, profile, method and budget, maximize *final composed-system macro-F1* over the arm's accessible validation population: K for blind, K∪U for visible. Use the common fixed class universe (`Benign`, five represented attacks, `Unknown`), with absent-class F1 set to zero. This rule is identical algebraically but its empirical objective value changes because U is visible. This is an **intentionally new symmetric experimental policy**, not the historical Phase 3B maximum-feasible-rejection selector, nor the historical visible UDR-first tau tuner. The choice of macro-F1 objective is a **proposal** requiring explicit scientific acceptance before outcomes are observed; do not tune it to historical performance.

**F. Deterministic ties and infeasibility.** Proposed ordering: larger accessible-label macro-F1 (differences below `1e-12` treated as tie), lower FUR on K, lower rejection fraction on K, then a canonical parameter ordering (for global thresholds prefer the **higher** uncertainty threshold; for family-conditional candidates prefer **smaller** alpha; methods are never pooled in one selector). Ties and float operations must be implemented consistently and independently unit-tested with exact-mass threshold fixtures. `No feasible candidate` is a categorical arm/method/budget outcome; do not impute a fallback point or a zero-valued test score. Where only one arm is feasible, report feasibility transition and leave numeric paired delta **NA**.

**Implementation note:** the original global rejector selector optimized maximum feasible rejection using sorted-score cumulative counts. Maximizing composed macro-F1 over *all* unique thresholds requires a **new** efficient cumulative-confusion-matrix search over tied score groups, otherwise repeated full-array F1 calculation can become infeasible for hundreds of thousands of observations. Must benchmark on synthetic data before fitting candidates. The implementation and its unit tests are new work and must not be described as the historical replay algorithm unchanged.

## 5. Prespecified reporting and acceptance tests

**Required outputs:** version/config/code/source SHA-256 identifiers, score-array identity, K/U indices and masks, candidate/profile selection ledgers, held-out exclusion at supervised fit, common-preprocessor fit provenance, selected Stage-1 threshold, rejector parameters/feasibility on K, per-arm test confusion matrices, UDR, FUR known-all/known-attacks, benign-to-family FPR, benign rejection, final macro-F1, system accuracy, realized overall rejection, and score-defined AUROC/AUPR only where semantically applicable. All primary signed differences defined as visible minus blind. Undefined outcomes remain NA with reason; do not silently drop infeasible arms. Full records remain local; any later release is a separate approved action.

**Unit and integration gates (pass before training):**
- Fully identical selector inputs ⇒ identical profile, threshold, rejection and zero paired delta. H-excluded no-visible-U branch must reproduce K-only selection behavior.
- Candidate Stage-2 known-only F1 invariant to U mask; Stage-1 tie AUC permitted to change when U added. Verify fixed candidate universe and canonical tie IDs.
- Demonstrate a synthetic U-only perturbation changes selection but does **not** change fitted model or K scores. Verify exact K row identity and labels in both views and test-row alignment.
- Rejector global uncertainty ties are never split; exact comparisons `>=` and family-conditional `<` match declared semantics. No hidden pandas/Python float parsing changes, including high-mass boundary scores. Benchmark cumulative search versus brute-force enumeration on small synthetic fixtures.
- Feasibility denominators always K, not accessible validation union; known-budget ceiling invariant to adding U. Feasibility may differ due to arm-specific gate/threshold, not altered denominators.
- Empty/undefined class, no attacks through Stage-1, missing candidate/AUC, no feasible point, and unequal selected profiles produce correct explicit statuses.
- Test labels and scores cannot be accessed by any selector; seal both policies before invoking evaluator. Pilot smoke tests must verify deterministic repeated-run signatures.

**Study design limits:** one seed/holdout pilot tests feasibility and pairing only. Fitting-seed variability and formal C03 evidence require the planned five-seed×six-holdout matched matrix under a prospectively frozen protocol. Network-flow test rows can be strongly dependent; do not compute row-independent inferential claims or represent 5 model seeds as independent traffic sources. Do not propose significance/practical-equivalence claims until dependence model and effect-margin/multiplicity family are specified prospectively.

## 6. Gates still open (not frozen)

1. **Post-cap support:** read-only deterministic sample-support counts for benign, all five represented attack families and H, on 1.5m sampled train/700k sampled val; confirm 200-row support floors and labels after strict LOAO filtering. Separate full test support established from archived score labels, subject to row provenance restrictions.
2. **Source identity and leakage:** full data-file SHA-256 and transformation/feature schema hashes to bind the **new** input snapshot. Existing label/order equality does not establish historical feature identity. Inspect source group, near duplicates and proxy features separately; do not label day/file boundaries independent captures without evidence.
3. **Selector freeze:** explicitly accept objective E, tie ordering F, K-only feasibility, and precise candidate/threshold search spaces; test edge cases and runtime on synthetic scores.
4. **Resource plan and explicit execution permission:** pilot includes XGB (Stage-1 800, Stage-2 1,400 estimators) and RF models and may need substantial memory/GPU time. No training until pilot resource envelope and output namespace are approved. Do not overwrite historical scores, files, code or releases.

**End state:** freeze-ready methodological proposal, not a frozen experiment. No model fitting, new result, manuscript claim, repo edit, PR/commit, or release modification is authorized by this review.
