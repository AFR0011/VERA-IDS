"""Synthetic timestamp format/label probe checks for Edge-IIoTset originals."""
from pathlib import Path
import json
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_probe_edgeiiot_timestamps.py"


def run(root, out, n=10):
    return subprocess.run([
        sys.executable, str(SCRIPT), "--input-root", str(root), "--out", str(out),
        "--sample-per-original-file", str(n), "--expected-original-files", "0"
    ], capture_output=True, text=True)


def test_format_timestamp_and_privacy(tmp_path):
    root = tmp_path / "src"
    normal = root / "Edge-IIoTset dataset" / "Normal traffic" / "Distance"
    attack = root / "Edge-IIoTset dataset" / "Attack traffic"
    normal.mkdir(parents=True)
    attack.mkdir(parents=True)
    (normal / "Distance.csv").write_text(
        'frame.time,ip.src_host,Attack_label,Attack_type\n'
        '"Dec 19, 2021 13:01:15.123456789 UTC",192.0.2.1,0,Normal\n',
        encoding="utf8"
    )
    (attack / "MITM_attack.csv").write_text(
        "frame.time,ip.src_host,Attack_label,Attack_type\n"
        "2021-12-20 12:00:04,203.0.113.4,1,MITM\n"
        "1639981820,203.0.113.3,1,MITM\n"
        "INVALID,203.0.113.3,1,MITM\n", encoding="utf8"
    )
    out = tmp_path / "summary.json"
    r = run(root, out)
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["source_kind_counts"] == {"attack": 1, "normal": 1}
    assert data["all_samples"][0]["timestamp_format_counts"] == {
        "iso_no_zone": 1, "other_unparsed": 1, "plausible_unix_seconds_utc": 1
    }
    assert data["all_samples"][1]["timestamp_format_counts"] == {"month_text_with_zone": 1}
    assert "192.0.2.1" not in out.read_text()
    assert "203.0.113.4" not in out.read_text()
    assert "Dec 19," not in out.read_text()


def test_missing_and_bad_headers(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    out = tmp_path / "report.json"
    r = run(root, out)
    assert r.returncode != 0 and not out.exists()
    attack = root / "Attack traffic"
    attack.mkdir()
    (attack / "bad.csv").write_text("frame.time,x\ntest,1\n")
    r = run(root, out)
    assert r.returncode != 0 and not out.exists()


def test_sampling_caps(tmp_path):
    attack = tmp_path / "src" / "Attack traffic"
    attack.mkdir(parents=True)
    (attack / "A.csv").write_text(
        "frame.time,Attack_label,Attack_type\n" +
        "2021-12-01 11:00:00,1,A\n" * 8
    )
    out = tmp_path / "report.json"
    r = run(attack.parent, out, n=3)
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["all_samples"][0]["rows_sampled_from_file_start"] == 3
