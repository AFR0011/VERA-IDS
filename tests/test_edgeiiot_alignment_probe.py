"""Synthetic original Edge-IIoTset CSV field width/position privacy checks."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_probe_edgeiiot_alignment.py"


def invoke(root: Path, out: Path, n: int = 20):
    return subprocess.run([
        sys.executable, str(SCRIPT), "--input-root", str(root),
        "--out", str(out), "--sample-per-file", str(n),
        "--expected-original-files", "0",
    ], capture_output=True, text=True)


def test_recognizes_year_fragments_and_ip_in_time_column(tmp_path: Path) -> None:
    root = tmp_path / "data"
    attack = root / "Attack traffic"
    normal = root / "Normal traffic" / "Modbus"
    attack.mkdir(parents=True)
    normal.mkdir(parents=True)
    header = "frame.time,ip.src_host,Attack_label,Attack_type\n"
    (attack / "Backdoor_attack.csv").write_text(
        header + " 2021 17:58:23.150209000 ,203.0.113.1,1,Backdoor\n",
        encoding="utf-8",
    )
    (normal / "Modbus.csv").write_text(
        header + "192.168.0.128,192.168.0.1,0,Normal\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    p = invoke(root, out)
    assert p.returncode == 0, p.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["sample_rows_total"] == 2
    assert data["sample_rows_with_field_count_mismatch"] == 0
    rows = {r["source_file"]: r for r in data["source_records"]}
    a = rows["Attack traffic/Backdoor_attack.csv"]
    m = rows["Normal traffic/Modbus/Modbus.csv"]
    assert a["first_column_shape_counts"]["0:frame.time"] == {"year_time_fragment": 1}
    assert m["first_column_shape_counts"]["0:frame.time"] == {"ip_address": 1}
    assert a["binary_label_field_counts"] == {"1": 1}
    assert m["attack_type_field_counts"] == {"Normal": 1}
    output = out.read_text(encoding="utf-8")
    assert "192.168.0.128" not in output
    assert "203.0.113.1" not in output
    assert "17:58:23" not in output


def test_unquoted_date_comma_creates_positional_width_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "data"
    attack = root / "Attack traffic"
    attack.mkdir(parents=True)
    (attack / "DDoS_UDP_Flood_attack.csv").write_text(
        "frame.time,ip.src_host,Attack_label,Attack_type\n"
        "Nov 21, 2021 17:58:23.150209000,203.0.113.9,1,DDoS_UDP\n"
        "\"Nov 21, 2021 17:58:23.150209000\",203.0.113.9,1,DDoS_UDP\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    p = invoke(root, out)
    assert p.returncode == 0, p.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    rec = data["source_records"][0]
    assert rec["row_field_count_distribution"] == {"4": 1, "5": 1}
    assert rec["rows_with_field_count_mismatch"] == 1
    assert rec["first_column_shape_counts"]["0:frame.time"] == {
        "date_time_text": 1, "month_day_fragment": 1,
    }
    assert rec["binary_label_field_counts"] == {"1": 1, "other": 1}


def test_missing_source_and_essential_header_fail_without_report(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    p = invoke(tmp_path / "missing", out)
    assert p.returncode != 0
    attack = tmp_path / "Attack traffic"
    attack.mkdir()
    (attack / "XSS_attack.csv").write_text("frame.time,foo\n0,1\n", encoding="utf-8")
    p = invoke(tmp_path, out)
    assert p.returncode != 0 and not out.exists()


def test_sampling_limit(tmp_path: Path) -> None:
    root = tmp_path / "data"
    attack = root / "Attack traffic"
    attack.mkdir(parents=True)
    (attack / "MITM_attack.csv").write_text(
        "frame.time,Attack_label,Attack_type\n"
        + "0.0,1,MITM\n" * 100,
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    p = invoke(root, out, n=3)
    assert p.returncode == 0, p.stderr
    rec = json.loads(out.read_text(encoding="utf-8"))["source_records"][0]
    assert rec["rows_sampled_from_start"] == 3
    assert rec["first_column_shape_counts"]["0:frame.time"] == {"number": 3}
