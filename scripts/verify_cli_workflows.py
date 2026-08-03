#!/usr/bin/env python3
"""Verify every advertised CLI and bounded synthetic package behavior."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ids_eval_framework.metrics import (  # noqa: E402
    macro_f1,
    protocol_a_supported_labels,
    protocol_b_labels,
)
from ids_eval_framework.src.open_set_rejection import (  # noqa: E402
    max_softmax_confidence,
    normalized_entropy,
    top2_margin,
)
from ids_eval_framework.src.statistics import bootstrap_mean_ci  # noqa: E402
from ids_eval_framework.src.support_sensitivity import evaluate_support_thresholds  # noqa: E402


COMMANDS = [
    ("01_audit_and_prepare.py", "--dry-run"),
    ("02_run_protocol_a_two_stage.py", "--dry-run"),
    ("03_run_protocol_a_flat_baseline.py", "--dry-run"),
    ("04_audit_protocol_b_support.py", "--dry-run"),
    ("04b_analyze_protocol_b_support_sensitivity.py", "--dry-run"),
    ("05_run_protocol_b_loao.py", "--dry-run"),
    ("06_run_open_set_rejectors.py", "--lane", "validation-selected", "--dry-run"),
    ("06_run_open_set_rejectors.py", "--lane", "exploratory-grid", "--dry-run"),
    ("07_run_external_stress_tests.py", "--dry-run"),
    ("08_build_statistics.py", "--dry-run"),
    ("09_build_paper_pack.py", "--dry-run"),
    ("10_run_seed_reliability.py", "--dry-run"),
    ("11_run_reference_framework_eval.py", "--dry-run"),
    ("11b_build_reference_framework_comparison.py", "--dry-run"),
]

SYNTHETIC_COMMANDS = [
    ("01_audit_and_prepare.py", "--synthetic"),
    ("02_run_protocol_a_two_stage.py", "--synthetic"),
    ("03_run_protocol_a_flat_baseline.py", "--synthetic"),
    ("04_audit_protocol_b_support.py", "--synthetic"),
    ("04b_analyze_protocol_b_support_sensitivity.py", "--synthetic"),
    ("05_run_protocol_b_loao.py", "--synthetic"),
    ("06_run_open_set_rejectors.py", "--lane", "validation-selected", "--synthetic"),
    ("06_run_open_set_rejectors.py", "--lane", "exploratory-grid", "--synthetic"),
    ("07_run_external_stress_tests.py", "--synthetic"),
    ("08_build_statistics.py", "--synthetic"),
    ("09_build_paper_pack.py", "--synthetic"),
    ("10_run_seed_reliability.py", "--synthetic"),
    ("11_run_reference_framework_eval.py", "--synthetic"),
    ("11b_build_reference_framework_comparison.py", "--synthetic"),
]


def cli_checks() -> None:
    for command in COMMANDS:
        help_command = [sys.executable, str(ROOT / "scripts" / command[0]), "--help"]
        subprocess.run(help_command, cwd=ROOT, check=True, capture_output=True, text=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def synthetic_checks() -> None:
    for command in SYNTHETIC_COMMANDS:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    true = ["Benign", "Botnet", "DDoS"]
    pred = ["Benign", "Unknown", "DDoS"]
    value = macro_f1(true, pred, protocol_a_supported_labels(["Botnet", "DDoS"]))
    assert 0.0 < value < 1.0
    assert protocol_b_labels(["Botnet", "DDoS"], unknown_support=2)[-1] == "Unknown"

    probs = np.asarray([[0.8, 0.2], [0.55, 0.45]], dtype=float)
    assert np.allclose(max_softmax_confidence(probs), [0.8, 0.55])
    assert np.all(top2_margin(probs) >= 0)
    assert np.all(normalized_entropy(probs) >= 0)
    mean, low, high = bootstrap_mean_ci([0.7, 0.8, 0.9], resamples=100, seed=123)
    assert low <= mean <= high

    import pandas as pd

    support = pd.DataFrame(
        [
            {"dataset": "synthetic", "family": family, "split": split, "count": count}
            for family, counts in {"Benign": (20, 20, 20), "A": (20, 20, 20), "B": (20, 20, 20)}.items()
            for split, count in zip(("train", "val", "test"), counts)
        ]
    )
    holdouts, summary = evaluate_support_thresholds(
        {"synthetic": support}, [10], benign_label="Benign", min_known_families_after_holdout=1
    )
    assert not holdouts.empty and bool(summary.iloc[0]["all_holdouts_eligible"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run CLI and synthetic checks.")
    parser.add_argument("--dry-run", action="store_true", help="Run CLI help/dry-run checks.")
    parser.add_argument("--synthetic", action="store_true", help="Run bounded synthetic checks.")
    args = parser.parse_args()
    if args.all or args.dry_run or not (args.dry_run or args.synthetic):
        cli_checks()
    if args.all or args.synthetic or not (args.dry_run or args.synthetic):
        synthetic_checks()
    print("Advertised CLI and synthetic workflow checks OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
