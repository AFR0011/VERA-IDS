#!/usr/bin/env python3
"""Canonical JISA Phase-1 controlled direct runner after pilot audit.

This wrapper preserves the Phase-1 training/evaluation implementation in
`jisa_phase1_run_controlled_direct.py` but replaces the pilot's single-boundary
FPR gate with the predeclared family-aware threshold sweep from the frozen
experiment specification.

The pilot output remains local and is non-canonical. New canonical Phase-1 runs
should use this script.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import jisa_phase1_run_controlled_direct as base  # noqa: E402

_GATE_MIN_SUPPORT = 50
_GATE_SWEEP_POINTS = 200


def _family_aware_threshold(
    y_idx: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    target_fpr: float,
) -> dict[str, float | int | str]:
    """Select a validation-only gate under the frozen family-aware rule.

    Feasible thresholds must satisfy benign FPR <= target_fpr. Selection is
    lexicographic: maximize minimum eligible-family gate recall, then total
    attack recall, then choose the highest threshold. The threshold set contains
    an exact benign-FPR boundary plus a 200-point quantile sweep over validation
    attack scores, so the FPR boundary is never accidentally missed by the
    numerical sweep.
    """
    benign_idx = labels.index("Benign")
    true_benign = y_idx == benign_idx
    if not np.any(true_benign):
        raise RuntimeError("Validation split contains no Benign rows")

    attack_score = 1.0 - proba[:, benign_idx]
    benign_scores = np.asarray(attack_score[true_benign], dtype=np.float64)
    n_benign = int(len(benign_scores))
    allowed = int(math.floor(float(target_fpr) * n_benign))

    # Exact most-permissive boundary satisfying the benign-FPR constraint,
    # accounting for tied/quantized probability values.
    desc = np.sort(benign_scores)[::-1]
    if allowed <= 0:
        boundary = float(np.nextafter(desc[0], np.inf))
    else:
        raw_boundary = float(desc[min(allowed - 1, n_benign - 1)])
        count_ge = int(np.sum(benign_scores >= raw_boundary))
        boundary = raw_boundary if count_ge <= allowed else float(np.nextafter(raw_boundary, np.inf))

    sweep_points = max(2, int(_GATE_SWEEP_POINTS))
    quantiles = np.linspace(0.0, 1.0, sweep_points)
    candidates = np.quantile(attack_score, quantiles)
    candidate_thresholds = {float(boundary), float(np.nextafter(np.max(attack_score), np.inf))}
    for value in np.asarray(candidates, dtype=np.float64):
        candidate_thresholds.add(float(value))
        candidate_thresholds.add(float(np.nextafter(value, np.inf)))

    attack_indices = [i for i, label in enumerate(labels) if label != "Benign"]
    family_masks = []
    for idx in attack_indices:
        mask = y_idx == idx
        support = int(mask.sum())
        if support >= int(_GATE_MIN_SUPPORT):
            family_masks.append(mask)
    if not family_masks:
        raise RuntimeError("No attack family meets the frozen minimum validation support")

    true_attack = ~true_benign
    best = None
    feasible_count = 0
    for threshold in sorted(candidate_thresholds):
        pred_attack = attack_score >= float(threshold)
        fpr = float(np.mean(pred_attack[true_benign]))
        if fpr > float(target_fpr) + 1e-12:
            continue
        feasible_count += 1
        family_recalls = [float(np.mean(pred_attack[m])) for m in family_masks]
        min_family_recall = float(min(family_recalls))
        attack_tpr = float(np.mean(pred_attack[true_attack])) if np.any(true_attack) else float("nan")
        row = {
            "threshold": float(threshold),
            "validation_benign_fpr": fpr,
            "validation_attack_recall": attack_tpr,
            "validation_min_family_gate_recall": min_family_recall,
        }
        if best is None:
            best = row
            continue
        key = (row["validation_min_family_gate_recall"], row["validation_attack_recall"], row["threshold"])
        best_key = (best["validation_min_family_gate_recall"], best["validation_attack_recall"], best["threshold"])
        if key > best_key:
            best = row

    if best is None:
        raise RuntimeError("Family-aware threshold sweep produced no feasible operating point")

    return {
        **best,
        "target_benign_fpr": float(target_fpr),
        "n_validation_benign": n_benign,
        "allowed_validation_false_positives": allowed,
        "min_family_support": int(_GATE_MIN_SUPPORT),
        "sweep_points": int(_GATE_SWEEP_POINTS),
        "candidate_thresholds": int(len(candidate_thresholds)),
        "feasible_thresholds": int(feasible_count),
        "selection_objective": "max_min_family_recall_then_attack_tpr_then_highest_threshold",
        "selection_scope": "validation_only",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=base.DEFAULT_CONFIG)
    p.add_argument("--dataset", choices=["CICIDS2017", "CICIoT2023"])
    p.add_argument("--model", choices=["rf", "xgb"])
    p.add_argument("--seed", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p


def main() -> int:
    global _GATE_MIN_SUPPORT, _GATE_SWEEP_POINTS
    args = build_parser().parse_args()
    if args.summarize_only:
        base.summarize(args.config)
        return 0

    if args.dataset is None or args.model is None or args.seed is None:
        raise SystemExit("--dataset, --model, and --seed are required unless --summarize-only is used")

    cfg = base.load_config(args.config)
    gate_cfg = cfg.get("fpr_matched_gate", {}) or {}
    _GATE_MIN_SUPPORT = int(gate_cfg.get("min_family_support", 50))
    _GATE_SWEEP_POINTS = int(gate_cfg.get("sweep_points", 200))

    base.choose_fpr_threshold = _family_aware_threshold
    base.run_item(args.config, args.dataset, args.model, args.seed, dry_run=args.dry_run, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
