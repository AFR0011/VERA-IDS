"""Synthetic verification of complete, read-only TON_IoT ground-truth census."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_audit_toniot_groundtruth.py"


def invoke(source: Path, target: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--groundtruth-root", str(source),
         "--out", str(target), *extra],
        capture_output=True, text=True,
    )


def test_counts_ranges_duplicates_and_order_issues(tmp_path: Path) -> None:
    source = tmp_path / "groundtruth"
    source.mkdir()
    data = ("ts,src_ip,src_port,dst_ip,dst_port,proto,type\n"
            "10,192.0.2.1,5,192.0.2.2,80,tcp,scanning\n"
            "9,192.0.2.3,6,192.0.2.2,80,tcp,dos\n")
    (source / "GroundTruth_Network_1.csv").write_text(data, encoding="utf-8")
    (source / "GroundTruth_Network_2.csv").write_text(data, encoding="utf-8")
    result_path = tmp_path / "results" / "full.json"
    process = invoke(source, result_path, "--sha256")
    assert process.returncode == 0, process.stderr
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["groundtruth_files"] == 2
    assert report["total_groundtruth_rows_as_stored"] == 4
    assert report["aggregate_type_counts_as_stored"] == {"dos": 2, "scanning": 2}
    assert report["groundtruth"][0]["backwards_timestamp_transitions"] == 1
    assert report["groundtruth"][0]["numeric_timestamp_range"]["min"] == 9.0
    assert len(report["duplicate_whole_files_by_sha256"]) == 1
    assert "192.0.2.1" not in result_path.read_text(encoding="utf-8")


def test_required_headers_and_no_partial_report(tmp_path: Path) -> None:
    source = tmp_path / "groundtruth"
    source.mkdir()
    (source / "GroundTruth_Network_1.csv").write_text(
        "ts,type\n100,normal\n", encoding="utf-8"
    )
    out = tmp_path / "report.json"
    process = invoke(source, out)
    assert process.returncode != 0
    assert not out.exists()


def test_missing_and_nonfinite_times(tmp_path: Path) -> None:
    source = tmp_path / "groundtruth"
    source.mkdir()
    (source / "GroundTruth_Network_1.csv").write_text(
        "ts,src_ip,src_port,dst_ip,dst_port,proto,type\n"
        ",192.0.2.1,1,192.0.2.2,80,tcp,normal\n"
        "NaN,192.0.2.1,1,192.0.2.2,80,tcp,normal\n"
        "15,192.0.2.1,1,192.0.2.2,80,tcp,backdoor\n", encoding="utf-8"
    )
    out = tmp_path / "report.json"
    process = invoke(source, out)
    assert process.returncode == 0, process.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    row = report["groundtruth"][0]
    assert row["invalid_or_missing_timestamps"] == 2
    assert row["missing_required_field_counts"]["ts"] == 1
    assert row["numeric_timestamp_range"]["count"] == 1
