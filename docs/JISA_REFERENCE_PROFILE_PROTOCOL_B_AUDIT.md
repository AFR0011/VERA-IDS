# JISA reference-profile Protocol-B audit

Status: TRACKED EVIDENCE AUDITED; LOCAL FEASIBILITY GATE REQUIRED  
Branch: `jisa-q1-revision`

## Purpose

Determine whether the historical Adewole-derived XGB and Neto-derived RF Protocol-B results can be restored as supporting evidence that VERA-IDS evaluation effects are not confined to the primary model configurations.

## What the tracked repository already establishes

The manifested compact surface `outputs/summaries/reference_profile_metric_drop.csv` contains complete six-holdout Protocol-B matrices for:

- Adewole-derived XGB / CICIDS2017;
- Adewole-derived XGB / CICIoT2023;
- Neto-derived RF / CICIoT2023.

The rows were produced by the reference-framework path, which injects fixed literature-derived model profiles into the Protocol-B runner. Each profile has a single model/parameter/weighting configuration per dataset/holdout and uses seed 123. Protocol-B threshold/tau selection is validation-visible.

Frozen descriptive summaries from the tracked surface are:

| Profile | Dataset | Protocol-A system macro-F1 | Protocol-B mean macro-F1 | Protocol-B range |
|---|---|---:|---:|---:|
| Adewole-derived XGB | CICIDS2017 | 0.753401 | 0.587074 | 0.381347–0.712236 |
| Adewole-derived XGB | CICIoT2023 | 0.671437 | 0.707310 | 0.609409–0.805744 |
| Neto-derived RF | CICIoT2023 | 0.828879 | 0.708573 | 0.614249–0.796934 |

These results do **not** support a universal degradation claim. One mean rises slightly under Protocol B. They do support the narrower observation that changing the evaluation condition materially changes the system-level result and exposes holdout-specific variation in literature-derived configurations.

## Important distinction

The profiles are literature-derived/framework-compatible configurations, not exact reproductions of the source studies. Published source metrics remain contextual only because source task, taxonomy, sampling, preprocessing, split, and metric definitions are not matched.

The Protocol-B reference-profile runs are also not the manuscript's five-seed validation-blind primary result. They are a separate, single-seed, validation-visible supporting analysis.

## Remaining local feasibility gate

The native Protocol-B runner can fall back to `tau=0` when no validation candidate satisfies the configured rejection constraints. The tracked compact summary does not retain the `tau_best.json -> ok` flag. Therefore the main-text use of the historical Protocol-B reference rows should be authorized only after confirming locally that every selected reference-profile run has `tau_best.json` with `ok=true`.

Run:

```powershell
python -m py_compile scripts/jisa_reference_profile_protocol_b_audit.py
python scripts/jisa_reference_profile_protocol_b_audit.py
```

Main-text use is authorized only if the final line reports:

`MAIN_TEXT_REFERENCE_PROTOCOL_B_AUTHORIZED=True`

## If the gate passes

Use the analysis to support the sentence-level claim:

> The dependence of the conclusion on evaluation condition was also observed in literature-derived model configurations and was therefore not confined to the primary VERA-IDS parameterizations.

Do not say:

- the source papers were exactly reproduced;
- VERA caused a performance drop relative to the published values;
- the profiles validate live zero-day detection;
- the reference-profile Protocol-B results are equivalent to the five-seed validation-blind analysis.
