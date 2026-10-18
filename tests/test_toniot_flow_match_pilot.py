"""Synthetic ground-truth-to-processed flow-key sample consistency checks."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_pilot_toniot_flow_match.py"
FIELDS = "ts,src_ip,src_port,dst_ip,dst_port,proto"


def run(gt: Path, processed: Path, out: Path, sample: int = 20):
    return subprocess.run([
        sys.executable, str(SCRIPT), "--groundtruth-root", str(gt),
        "--processed-root", str(processed), "--out", str(out),
        "--sample-per-file-type", str(sample),
    ], capture_output=True, text=True)


def test_matches_identity_only_and_absence_with_no_flow_data_in_report(tmp_path: Path) -> None:
    gt = tmp_path / "gt"; processed = tmp_path / "processed"
    gt.mkdir(); processed.mkdir()
    (gt / "GroundTruth_Network_1.csv").write_text(
        FIELDS + ",type\n"
        "1,192.0.2.1,10,192.0.2.2,80,tcp,scanning\n"
        "2,192.0.2.1,10,192.0.2.2,80,tcp,scanning\n"
        "3,192.0.2.1,10,192.0.2.2,80,tcp,scanning\n"
        "4,192.0.2.1,10,192.0.2.2,80,tcp,dos\n", encoding="utf-8"
    )
    (processed / "Network_dataset_1.csv").write_text(
        FIELDS + ",label,type\n"
        "1,192.0.2.1,10,192.0.2.2,80,tcp,1,scanning\n"
        "2,192.0.2.1,10,192.0.2.2,80,tcp,1,dos\n"
        "4,192.0.2.1,10,192.0.2.2,80,tcp,1,dos\n", encoding="utf-8"
    )
    out = tmp_path / "results" / "audit.json"
    p = run(gt, processed, out)
    assert p.returncode == 0, p.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["groundtruth_rows_scanned_as_stored"] == 4
    assert result["processed_rows_scanned"] == 3
    assert result["totals_as_stored_including_duplicate_gt_files"] == {
        "sampled_rows_as_stored": 4,
        "exact_flow_and_attack_label_matches": 2,
        "identity_matches_without_same_attack_label": 1,
        "unmatched_exact_six_field_keys": 1,
    }
    assert "192.0.2.1" not in out.read_text(encoding="utf-8")
    assert result["unique_sampled_six_field_keys"] == 4


def test_reject_missing_input_and_zero_sample(tmp_path: Path) -> None:
    p = run(tmp_path / "none", tmp_path / "missing", tmp_path / "audit.json")
    assert p.returncode != 0
    p2 = run(tmp_path / "none", tmp_path / "missing", tmp_path / "audit.json", sample=0)
    assert p2.returncode != 0
    assert not (tmp_path / "audit.json").exists()
