"""Behavioral tests for the read-only TON_IoT feasibility inventory."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_preflight_toniot_source.py"


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_bounded_sample_and_candidate_tags(tmp_path: Path) -> None:
    source = tmp_path / "input"
    source.mkdir()
    (source / "subset.csv").write_text("ts,src_ip,proto,label,type,packets\n1,1.2.3.4,tcp,0,normal,2\n2,1.2.3.5,tcp,1,dos,5\n3,1.2.3.6,udp,1,scan,3\n", encoding="utf-8")
    out = tmp_path / "out" / "inventory.json"
    result = invoke("--input-root", str(source), "--out", str(out), "--max-rows-per-file", "2")
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["row_count_complete"] is False
    assert record["file_hashes_complete"] is False
    f = record["files"][0]
    assert f["observed_rows"] == 2
    assert f["column_candidates"]["ts"] == ["time_candidate"]
    assert f["column_candidates"]["src_ip"] == ["identity_candidate"]
    assert f["label_value_counts"]["type"] == {"normal": 1, "dos": 1}
    assert f["sha256"] is None


def test_full_scan_hash_and_dry_run(tmp_path: Path) -> None:
    source = tmp_path / "rows.csv"
    source.write_text("label,feature\n0,a\n1,b\n", encoding="utf-8")
    out = tmp_path / "manifest.json"
    dry = invoke("--input-root", str(source), "--out", str(out), "--dry-run")
    assert dry.returncode == 0 and not out.exists()
    full = invoke("--input-root", str(source), "--out", str(out), "--max-rows-per-file", "0", "--sha256")
    assert full.returncode == 0, full.stderr
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["row_count_complete"] is True
    assert len(record["files"][0]["sha256"]) == 64
    assert record["files"][0]["observed_rows"] == 2


def test_missing_source_is_error(tmp_path: Path) -> None:
    result = invoke("--input-root", str(tmp_path / "not_there"))
    assert result.returncode != 0
