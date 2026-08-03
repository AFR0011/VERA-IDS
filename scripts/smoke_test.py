#!/usr/bin/env python3
"""Dataset-free smoke test for the public release."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ids_eval_framework.metrics import (  # noqa: E402
    macro_f1,
    protocol_a_supported_labels,
    protocol_b_labels,
    unknown_detection_rate,
)
from ids_eval_framework.src.paths import load_config  # noqa: E402


def main() -> None:
    config = load_config("config/protocol_b_loao.yml")
    rules = config["support_audit_rules"]
    assert rules["unknown_support_for_holdout"]["test"] == 200

    families = ["Botnet", "DDoS"]
    labels_a = protocol_a_supported_labels(families)
    labels_b = protocol_b_labels(families, unknown_support=2)
    y_true = ["Benign", "Botnet", "Unknown", "Unknown"]
    y_pred = ["Benign", "Botnet", "Unknown", "DDoS"]
    assert labels_a == ["Benign", "Botnet", "DDoS"]
    assert labels_b == ["Benign", "Botnet", "DDoS", "Unknown"]
    assert unknown_detection_rate(y_true, y_pred) == 0.5
    score = macro_f1(y_true, y_pred, labels_b)
    assert 0.0 <= score <= 1.0

    summary = REPO_ROOT / "outputs" / "summaries" / "protocol_a_core_summary.csv"
    with summary.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and {
        "stage1_roc_auc",
        "stage2_macro_f1_fixedK",
        "system_macro_f1_supported_labels",
        "system_macro_f1_declared_output_labels_historical",
    } <= set(rows[0])

    smoke_dir = REPO_ROOT / "outputs" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pass",
        "dataset_rows_used": 0,
        "protocol_b_unknown_support_guard": True,
        "synthetic_macro_f1": score,
    }
    output = smoke_dir / "smoke_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: wrote {output.relative_to(REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
