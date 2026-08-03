#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ids_eval_framework._native.full_scope_utils import load_module


CFG = {
    "source_root": Path("protocolB_grid_runs step 5 - open-set baselines"),
    "out_root": Path("q2_statistical_refresh"),
    "bootstrap_resamples": 1_000,
    "random_seed": 123,
}


def bh_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(p))
    if valid_idx.size == 0:
        return pd.Series(out, index=values.index)
    ordered = valid_idx[np.argsort(p[valid_idx])]
    adjusted = np.empty(len(ordered), dtype=float)
    running = 1.0
    m = len(ordered)
    for reverse_rank, idx in enumerate(ordered[::-1], start=1):
        rank = m - reverse_rank + 1
        running = min(running, p[idx] * m / rank)
        adjusted[rank - 1] = min(1.0, running)
    for rank, idx in enumerate(ordered):
        out[idx] = adjusted[rank]
    return pd.Series(out, index=values.index)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    source_root = CFG["source_root"].resolve()
    out_root = CFG["out_root"].resolve()
    summary_root = out_root / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    open_mod = load_module("14.ProtocolB_OpenSetBaselines.py", "ids_q2_open_set_stats")
    grid_mod = load_module("5.ProtocolB_GridRunner_V1.2.py", "ids_q2_grid_stats")
    resamples = int(CFG["bootstrap_resamples"])
    seed = int(CFG["random_seed"])

    plan_path = source_root / "scenario_plan.csv"
    plan = pd.read_csv(plan_path)
    ci_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for case_idx, record in enumerate(plan.to_dict("records"), start=1):
        dataset = str(record["dataset"])
        holdout = str(record["holdout_family"])
        run_dir = Path(open_mod.case_run_dir(str(source_root), dataset, holdout))
        manifest = load_json(run_dir / "scenario_manifest.json")
        families = list(manifest["valid_known_families"])
        unknown_label = str(manifest.get("unknown_label", "Unknown"))
        selected = pd.read_csv(run_dir / "selected_methods.csv")
        selected_test = selected.loc[selected["split"] == "test"].copy()
        test_scores = pd.read_csv(run_dir / "test_scores.csv.gz")
        y_true = test_scores["y_true_sys"].astype(str).to_numpy(dtype=object)

        predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        metrics_by_method: dict[str, dict[str, Any]] = {}
        for row in selected_test.to_dict("records"):
            method = str(row["method"])
            params = open_mod.parse_json_dict(row["selection_param_json"])
            pred, reject, uncertainty = open_mod.predict_for_method(
                test_scores,
                method,
                float(row["thr_high"]),
                families,
                unknown_label,
                params,
            )
            metrics = open_mod.evaluate_predictions(
                grid_mod,
                test_scores,
                pred,
                reject,
                uncertainty,
                families,
                unknown_label,
            )
            predictions[method] = (pred, reject)
            metrics_by_method[method] = metrics
            ci_rows.append(
                {
                    "dataset": dataset,
                    "holdout_family": holdout,
                    "split_variant": str(row["split_variant"]),
                    "model_family": str(row["model_family"]),
                    "stage1_weight_mode": str(row["stage1_weight_mode"]),
                    "method": method,
                    "bootstrap_resamples": resamples,
                    "random_seed": seed + case_idx,
                    **metrics,
                    **open_mod.bootstrap_ci(
                        grid_mod,
                        y_true,
                        pred,
                        reject,
                        families,
                        unknown_label,
                        resamples,
                        seed + case_idx,
                    ),
                }
            )

        control_pred, control_reject = predictions["control_tau"]
        for method, (pred, reject) in predictions.items():
            if method == "control_tau":
                continue
            paired_rows.append(
                {
                    "dataset": dataset,
                    "holdout_family": holdout,
                    "split_variant": str(selected_test.iloc[0]["split_variant"]),
                    "method_a": method,
                    "method_b": "control_tau",
                    "bootstrap_resamples": resamples,
                    "random_seed": seed + 1000 + case_idx,
                    **open_mod.paired_bootstrap_diff(
                        grid_mod,
                        y_true,
                        pred,
                        reject,
                        control_pred,
                        control_reject,
                        families,
                        unknown_label,
                        resamples,
                        seed + 1000 + case_idx,
                    ),
                }
            )

        audit_rows.append(
            {
                "dataset": dataset,
                "holdout_family": holdout,
                "source_run_dir": str(run_dir.relative_to(Path.cwd())),
                "n_test_rows": len(test_scores),
                "n_true_unknown": int((y_true == unknown_label).sum()),
                "methods": "|".join(sorted(predictions)),
                "selection_source": "selected_methods.csv validation-selected parameters",
                "test_reselection_performed": False,
            }
        )

    ci = pd.DataFrame(ci_rows)
    paired = pd.DataFrame(paired_rows)
    paired["macro_f1_pvalue_bh"] = bh_adjust(paired["delta_macro_f1_pvalue_bootstrap"])
    paired["unknown_detection_pvalue_bh"] = bh_adjust(
        paired["delta_unknown_detection_pvalue_bootstrap"]
    )
    paired["macro_f1_pvalue_bonferroni"] = np.minimum(
        1.0, paired["delta_macro_f1_pvalue_bootstrap"] * len(paired)
    )
    paired["unknown_detection_pvalue_bonferroni"] = np.minimum(
        1.0, paired["delta_unknown_detection_pvalue_bootstrap"] * len(paired)
    )

    ci.to_csv(summary_root / "protocol_b_holdout_confidence_intervals_1000.csv", index=False)
    paired.to_csv(summary_root / "protocol_b_paired_method_tests_1000.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(summary_root / "selection_discipline_audit.csv", index=False)
    with (out_root / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_root": str(source_root),
                "bootstrap_resamples": resamples,
                "random_seed": seed,
                "n_holdouts": len(plan),
                "n_ci_rows": len(ci),
                "n_paired_rows": len(paired),
                "models_retrained": False,
                "parameters_reselected": False,
                "test_predictions_reused": True,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
