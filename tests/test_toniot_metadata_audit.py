"""Synthetic checks for safe streaming TON_IoT source metadata auditing."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_audit_toniot_metadata.py"


def test_scan_counts_timestamps_and_source_headers(tmp_path: Path) -> None:
    p = tmp_path / "processed"
    g = tmp_path / "groundtruth"
    p.mkdir(); g.mkdir()
    (p / "Network_dataset_1.csv").write_text(
        "ts,label,type,feature\n10,0,normal,a\n11.5,1,dos,b\n,1,dos,c\n", encoding="utf-8"
    )
    (p / "Network_dataset_2.csv").write_text(
        "ts,uid,label,type\n12,1,1,mitm\n13,2,0,normal\n", encoding="utf-8"
    )
    (g / "GroundTruth_Network_1.csv").write_text("attack_start,src_ip\n15,1.1.1.1\n", encoding="utf-8")
    (g / "GroundTruth_Network_2.csv").write_text("attack_start,src_ip\n15,1.1.1.1\n", encoding="utf-8")
    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--processed-root", str(p),
                        "--groundtruth-root", str(g), "--out", str(out), "--sha256"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["total_processed_rows"] == 5
    assert result["aggregate_type_counts"] == {"dos": 2, "mitm": 1, "normal": 2}
    assert result["processed"][0]["invalid_or_missing_timestamps"] == 1
    assert result["processed"][0]["numeric_timestamp_range"]["max"] == 11.5
    assert "uid" in result["processed"][1]["columns"]
    assert len(result["duplicate_groundtruth_files_by_sha256"]) == 1
    assert result["groundtruth_headers"][0]["columns"] == ["attack_start", "src_ip"]


def test_fails_on_missing_essential_columns(tmp_path: Path) -> None:
    p = tmp_path / "processed"; g = tmp_path / "groundtruth"; p.mkdir(); g.mkdir()
    (p / "Network_dataset_1.csv").write_text("ts,label\n1,0\n", encoding="utf-8")
    (g / "GroundTruth_Network_1.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    out = tmp_path / "report.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--processed-root", str(p),
                        "--groundtruth-root", str(g), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert not out.exists()
