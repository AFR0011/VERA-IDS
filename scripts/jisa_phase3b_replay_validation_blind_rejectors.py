#!/usr/bin/env python3
"""Replay Phase-3B rejectors on frozen heldout-validation-blind score files.

No model fitting occurs here. Rejector operating points are selected only from the
Phase-3A heldout-free validation score surface, then evaluated once on the frozen full
test score surface.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "jisa_phase3_validation_blind.yml"
DATASET = "CICIoT2023"
METHODS = ["max_confidence", "margin", "entropy", "family_conditional"]
SCORE_USECOLS = [
    "y_true_sys",
    "is_true_unknown",
    "p_attack",
    "fam_pred_family",
    "fam_pmax",
    "top2_margin",
    "stage2_entropy",
    "true_known_family_prob",
]
TOL = 1e-12


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected YAML mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def sanitize(value: str) -> str:
    return str(value).replace("/", "_")


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=SCORE_USECOLS)
    for col in ["is_true_unknown", "p_attack", "fam_pmax", "top2_margin", "stage2_entropy", "true_known_family_prob"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["p_attack", "fam_pmax", "top2_margin", "stage2_entropy"]].isna().any().any():
        raise RuntimeError(f"Non-finite required score in {path}")
    return df


def prepare(df: pd.DataFrame, families: list[str], stage1_threshold: float) -> dict[str, Any]:
    labels = ["Benign"] + list(families) + ["Unknown"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_true = df["y_true_sys"].astype(str)
    unknown_values = sorted(set(y_true) - set(labels))
    if unknown_values:
        raise RuntimeError(f"Unexpected truth labels: {unknown_values}")
    y_idx = y_true.map(label_to_idx).to_numpy(dtype=np.int16)

    fam_pred = df["fam_pred_family"].astype(str)
    bad_pred = sorted(set(fam_pred) - set(families))
    if bad_pred:
        raise RuntimeError(f"Unexpected predicted families: {bad_pred[:10]}")
    fam_idx = fam_pred.map({f: i + 1 for i, f in enumerate(families)}).to_numpy(dtype=np.int16)

    p_attack = df["p_attack"].to_numpy(dtype=np.float64)
    gate = p_attack >= float(stage1_threshold)
    base_pred = np.zeros(len(df), dtype=np.int16)
    base_pred[gate] = fam_idx[gate]

    return {
        "n": len(df),
        "labels": labels,
        "y_idx": y_idx,
        "base_pred": base_pred,
        "gate": gate,
        "benign_mask": y_idx == 0,
        "unknown_mask": y_idx == len(labels) - 1,
        "known_attack_mask": (y_idx > 0) & (y_idx < len(labels) - 1),
        "fam_pred_idx0": fam_idx.astype(np.int16) - 1,
        "fam_pmax": df["fam_pmax"].to_numpy(dtype=np.float64),
        "margin": df["top2_margin"].to_numpy(dtype=np.float64),
        "entropy": df["stage2_entropy"].to_numpy(dtype=np.float64),
        "true_known_family_prob": df["true_known_family_prob"].to_numpy(dtype=np.float64),
    }


def metrics(state: dict[str, Any], reject: np.ndarray) -> dict[str, float]:
    y = state["y_idx"]
    pred = state["base_pred"].copy()
    unknown_idx = len(state["labels"]) - 1
    pred[reject] = unknown_idx
    n_labels = len(state["labels"])
    cm = np.bincount(y.astype(np.int64) * n_labels + pred.astype(np.int64), minlength=n_labels * n_labels).reshape(n_labels, n_labels)
    support = cm.sum(axis=1).astype(float)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = support - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    supported = support > 0

    benign = state["benign_mask"]
    unknown = state["unknown_mask"]
    known_att = state["known_attack_mask"]
    known_all = ~unknown
    benign_n = int(benign.sum())
    unknown_n = int(unknown.sum())
    known_att_n = int(known_att.sum())
    known_all_n = int(known_all.sum())

    benign_family = benign & (pred > 0) & (pred < unknown_idx)
    return {
        "macro_f1_supported": float(np.mean(f1[supported])) if supported.any() else float("nan"),
        "accuracy": float(tp.sum() / len(y)) if len(y) else float("nan"),
        "overall_reject_rate": float(reject.mean()) if len(reject) else 0.0,
        "benign_reject_rate": float(reject[benign].mean()) if benign_n else 0.0,
        "benign_family_fp_rate": float(benign_family.sum() / benign_n) if benign_n else 0.0,
        "false_unknown_rate_all_known": float(reject[known_all].sum() / known_all_n) if known_all_n else 0.0,
        "false_unknown_rate_known_attacks": float(reject[known_att].sum() / known_att_n) if known_att_n else 0.0,
        "unknown_detection_rate": float(reject[unknown].sum() / unknown_n) if unknown_n else float("nan"),
    }


def feasible(m: dict[str, float], budget: float, cfg: dict[str, Any]) -> bool:
    return bool(
        m["overall_reject_rate"] <= budget + TOL
        and m["benign_family_fp_rate"] <= float(cfg["max_benign_family_fp_rate"]) + TOL
        and m["benign_reject_rate"] <= float(cfg["max_benign_reject_rate"]) + TOL
        and m["false_unknown_rate_all_known"] <= float(cfg["max_false_unknown_rate_all_known"]) + TOL
        and m["false_unknown_rate_known_attacks"] <= float(cfg["max_false_unknown_rate_known_attacks"]) + TOL
    )


def uncertainty(state: dict[str, Any], method: str) -> np.ndarray:
    out = np.zeros(state["n"], dtype=np.float64)
    gate = state["gate"]
    if method == "max_confidence":
        out[gate] = 1.0 - state["fam_pmax"][gate]
    elif method == "margin":
        out[gate] = 1.0 - state["margin"][gate]
    elif method == "entropy":
        out[gate] = state["entropy"][gate]
    else:
        raise ValueError(method)
    return out


def select_global(state: dict[str, Any], method: str, budget: float, cfg: dict[str, Any]) -> dict[str, Any] | None:
    gate_idx = np.flatnonzero(state["gate"])
    if gate_idx.size == 0:
        rej = np.zeros(state["n"], dtype=bool)
        m = metrics(state, rej)
        return {"uncertainty_threshold": math.inf, "validation_metrics": m, "reject": rej} if feasible(m, budget, cfg) else None

    u_all = uncertainty(state, method)
    u = u_all[gate_idx]
    if not np.isfinite(u).all():
        raise RuntimeError(f"Non-finite {method} uncertainty")
    order_local = np.argsort(-u, kind="mergesort")
    idx_sorted = gate_idx[order_local]
    u_sorted = u[order_local]
    # End positions of equal-score groups. Equal uncertainty scores are never split.
    ends = np.flatnonzero(np.r_[u_sorted[:-1] != u_sorted[1:], True]) + 1

    benign_sorted = state["benign_mask"][idx_sorted].astype(np.int64)
    known_att_sorted = state["known_attack_mask"][idx_sorted].astype(np.int64)
    c_benign = np.cumsum(benign_sorted)
    c_known_att = np.cumsum(known_att_sorted)

    n = state["n"]
    benign_n = int(state["benign_mask"].sum())
    known_att_n = int(state["known_attack_mask"].sum())
    base_benign_fp = int((state["benign_mask"] & state["gate"]).sum())

    best_k: int | None = None
    best_thr = math.inf
    # k=0 first.
    candidates = [0] + [int(x) for x in ends]
    for k in candidates:
        ben_rej = 0 if k == 0 else int(c_benign[k - 1])
        att_rej = 0 if k == 0 else int(c_known_att[k - 1])
        overall = k / n
        ben_rej_rate = ben_rej / benign_n if benign_n else 0.0
        att_fur = att_rej / known_att_n if known_att_n else 0.0
        benign_fpr = (base_benign_fp - ben_rej) / benign_n if benign_n else 0.0
        # Validation is all-known, so all-known FUR equals overall rejection rate.
        if (
            overall <= budget + TOL
            and benign_fpr <= float(cfg["max_benign_family_fp_rate"]) + TOL
            and ben_rej_rate <= float(cfg["max_benign_reject_rate"]) + TOL
            and overall <= float(cfg["max_false_unknown_rate_all_known"]) + TOL
            and att_fur <= float(cfg["max_false_unknown_rate_known_attacks"]) + TOL
        ):
            if best_k is None or k > best_k:
                best_k = k
                best_thr = math.inf if k == 0 else float(u_sorted[k - 1])

    if best_k is None:
        return None
    reject = np.zeros(n, dtype=bool) if best_k == 0 else (state["gate"] & (u_all >= best_thr))
    m = metrics(state, reject)
    if not feasible(m, budget, cfg):
        raise RuntimeError(f"Internal feasibility mismatch for {method}, budget={budget}")
    return {"uncertainty_threshold": best_thr, "validation_metrics": m, "reject": reject}


def family_thresholds(state: dict[str, Any], families: list[str], alpha: float) -> list[float]:
    out: list[float] = []
    probs = state["true_known_family_prob"]
    y = state["y_idx"]
    for fam_i, _fam in enumerate(families, start=1):
        vals = probs[y == fam_i]
        vals = vals[np.isfinite(vals)]
        out.append(0.0 if vals.size == 0 else float(np.quantile(vals, alpha)))
    return out


def family_reject(state: dict[str, Any], thresholds: list[float]) -> np.ndarray:
    gate = state["gate"]
    thr = np.asarray(thresholds, dtype=np.float64)
    pred_fam0 = state["fam_pred_idx0"]
    reject = np.zeros(state["n"], dtype=bool)
    idx = np.flatnonzero(gate)
    reject[idx] = state["fam_pmax"][idx] < thr[pred_fam0[idx]]
    return reject


def select_family_conditional(state: dict[str, Any], families: list[str], budget: float, cfg: dict[str, Any], alpha_grid: list[float]) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        thresholds = family_thresholds(state, families, alpha)
        reject = family_reject(state, thresholds)
        m = metrics(state, reject)
        if feasible(m, budget, cfg):
            rows.append({"alpha": alpha, "thresholds": thresholds, "validation_metrics": m, "reject": reject})
    if not rows:
        return None
    rows.sort(
        key=lambda r: (
            -float(r["validation_metrics"]["overall_reject_rate"]),
            -float(r["validation_metrics"]["macro_f1_supported"]),
            float(r["validation_metrics"]["false_unknown_rate_all_known"]),
            float(r["alpha"]),
        )
    )
    return rows[0]


def apply_global(state: dict[str, Any], method: str, threshold: float) -> np.ndarray:
    if math.isinf(float(threshold)):
        return np.zeros(state["n"], dtype=bool)
    return state["gate"] & (uncertainty(state, method) >= float(threshold))


def unknown_curve(state: dict[str, Any], method: str) -> tuple[float, float]:
    target = state["unknown_mask"].astype(int)
    if np.unique(target).size < 2:
        return float("nan"), float("nan")
    score = uncertainty(state, method)
    return float(roc_auc_score(target, score)), float(average_precision_score(target, score))


def alpha_grid(cfg: dict[str, Any]) -> list[float]:
    spec = cfg["family_conditional_alpha_grid"]
    start, stop, step = float(spec["start"]), float(spec["stop"]), float(spec["step"])
    count = int(round((stop - start) / step))
    return [round(start + i * step, 12) for i in range(count + 1)]


def case_paths(cfg: dict[str, Any], seed: int, holdout: str) -> tuple[Path, Path]:
    root = REPO_ROOT / str(cfg["experiment"]["output_root"])
    score = root / "score_generation" / f"seed_{seed}" / sanitize(holdout)
    out = root / "rejector_replay" / f"seed_{seed}" / sanitize(holdout)
    return score, out


def run_case(cfg: dict[str, Any], seed: int, holdout: str, force: bool = False) -> int:
    score_dir, out_dir = case_paths(cfg, seed, holdout)
    selected_path = score_dir / "selected_profile.json"
    val_path = score_dir / "val_known_scores.csv.gz"
    test_path = score_dir / "test_scores.csv.gz"
    if not (selected_path.exists() and val_path.exists() and test_path.exists()):
        raise RuntimeError(f"Incomplete Phase-3A case: {score_dir}")
    out_csv = out_dir / "selected_points.csv"
    if out_csv.exists() and not force:
        print(f"SKIP complete seed={seed} holdout={holdout}")
        return 0

    selected = load_json(selected_path)
    families = [str(x) for x in selected["known_families"]]
    s1_thr = float(selected["stage1_threshold"])
    rcfg = cfg["rejector_replay"]
    budgets = [float(x) for x in rcfg["budgets"]]
    alphas = alpha_grid(rcfg)

    val_df = load_scores(val_path)
    val_state = prepare(val_df, families, s1_thr)
    if int(val_state["unknown_mask"].sum()) != 0:
        raise RuntimeError("Blind validation unexpectedly contains true Unknown rows")

    selections: list[dict[str, Any]] = []
    for method in METHODS:
        for budget in budgets:
            if method == "family_conditional":
                sel = select_family_conditional(val_state, families, budget, rcfg, alphas)
            else:
                sel = select_global(val_state, method, budget, rcfg)
            if sel is None:
                selections.append({"method": method, "budget": budget, "feasible": False})
                continue
            vm = dict(sel["validation_metrics"])
            row: dict[str, Any] = {"method": method, "budget": budget, "feasible": True}
            row.update({f"val_{k}": v for k, v in vm.items()})
            if method == "family_conditional":
                row["alpha"] = float(sel["alpha"])
                row["family_thresholds_json"] = json.dumps({f: float(t) for f, t in zip(families, sel["thresholds"])}, sort_keys=True)
            else:
                row["uncertainty_threshold"] = float(sel["uncertainty_threshold"])
            selections.append(row)

    del val_df, val_state
    test_df = load_scores(test_path)
    test_state = prepare(test_df, families, s1_thr)
    curves = {m: unknown_curve(test_state, m) for m in ["max_confidence", "margin", "entropy"]}

    out_rows: list[dict[str, Any]] = []
    for row in selections:
        rec = {
            "dataset": DATASET,
            "seed": seed,
            "holdout_family": holdout,
            "selected_profile": str(selected["selected_candidate_label"]),
            "stage1_threshold": s1_thr,
            **row,
        }
        if not bool(row["feasible"]):
            out_rows.append(rec)
            continue
        method = str(row["method"])
        if method == "family_conditional":
            thresholds_map = json.loads(str(row["family_thresholds_json"]))
            reject = family_reject(test_state, [float(thresholds_map[f]) for f in families])
            auroc, aupr = float("nan"), float("nan")
        else:
            reject = apply_global(test_state, method, float(row["uncertainty_threshold"]))
            auroc, aupr = curves[method]
        tm = metrics(test_state, reject)
        rec.update({f"test_{k}": v for k, v in tm.items()})
        rec["test_unknown_known_auroc"] = auroc
        rec["test_unknown_known_aupr"] = aupr
        out_rows.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(out_rows).sort_values(["budget", "method"])
    result.to_csv(out_csv, index=False)
    manifest = {
        "dataset": DATASET,
        "seed": seed,
        "holdout_family": holdout,
        "phase3a_score_dir": str(score_dir),
        "selected_profile": selected["selected_candidate_label"],
        "stage1_threshold": s1_thr,
        "heldout_validation_used_for_rejector_selection": False,
        "test_used_for_rejector_selection": False,
        "selection_objective": "max_feasible_validation_rejection",
        "methods": METHODS,
        "budgets": budgets,
        "primary_budget": float(rcfg["primary_budget"]),
    }
    (out_dir / "case_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    view_cols = ["budget", "method", "feasible", "val_overall_reject_rate", "test_unknown_detection_rate", "test_false_unknown_rate_all_known", "test_macro_f1_supported"]
    for col in view_cols:
        if col not in result.columns:
            result[col] = np.nan
    print(result[view_cols].to_string(index=False))
    print(f"COMPLETE seed={seed} holdout={holdout} rows={len(result)} feasible={int(result['feasible'].fillna(False).sum())}")
    return 0


def summarize(cfg: dict[str, Any]) -> int:
    root = REPO_ROOT / str(cfg["experiment"]["output_root"]) / "rejector_replay"
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("seed_*/*/selected_points.csv")):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        print("No Phase-3B results found.")
        return 1
    all_df = pd.concat(frames, ignore_index=True)
    summary_root = root.parent / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(summary_root / "phase3b_case_results.csv", index=False)

    metric_cols = [
        "test_unknown_detection_rate",
        "test_false_unknown_rate_all_known",
        "test_false_unknown_rate_known_attacks",
        "test_overall_reject_rate",
        "test_benign_reject_rate",
        "test_benign_family_fp_rate",
        "test_macro_f1_supported",
        "test_accuracy",
        "test_unknown_known_auroc",
        "test_unknown_known_aupr",
    ]
    rows: list[dict[str, Any]] = []
    for keys, g in all_df.groupby(["holdout_family", "method", "budget"], sort=True):
        holdout, method, budget = keys
        feasible_g = g[g["feasible"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
        rec: dict[str, Any] = {
            "holdout_family": holdout,
            "method": method,
            "budget": float(budget),
            "n_total": int(len(g)),
            "n_feasible": int(len(feasible_g)),
        }
        for col in metric_cols:
            vals = pd.to_numeric(feasible_g.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
            rec[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            rec[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        rows.append(rec)
    summary = pd.DataFrame(rows).sort_values(["budget", "holdout_family", "method"])
    summary.to_csv(summary_root / "phase3b_five_seed_summary.csv", index=False)

    expected_cases = len(cfg["experiment"]["seeds"]) * len(cfg["experiment"]["holdouts"])
    found_cases = len(list(root.glob("seed_*/*/selected_points.csv")))
    print(f"completed cases: {found_cases}/{expected_cases}")
    primary = float(cfg["rejector_replay"]["primary_budget"])
    view = summary[np.isclose(summary["budget"].astype(float), primary)].copy()
    cols = ["holdout_family", "method", "n_feasible", "test_unknown_detection_rate_mean", "test_false_unknown_rate_all_known_mean", "test_macro_f1_supported_mean"]
    print(view[cols].to_string(index=False))
    print(f"summary: {summary_root}")
    return 0 if found_cases == expected_cases else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--seed", type=int)
    p.add_argument("--holdout")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    args = p.parse_args()
    cfg = load_yaml(args.config.resolve())
    if args.summarize_only:
        return summarize(cfg)
    if args.seed is None or not args.holdout:
        raise SystemExit("--seed and --holdout are required unless --summarize-only is used")
    if args.seed not in [int(x) for x in cfg["experiment"]["seeds"]]:
        raise SystemExit(f"Seed {args.seed} is not configured")
    if args.holdout not in [str(x) for x in cfg["experiment"]["holdouts"]]:
        raise SystemExit(f"Holdout {args.holdout!r} is not configured")
    return run_case(cfg, args.seed, args.holdout, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
