"""Synthetic tests for the read-only Edge-IIoTset full PCAP chronology census."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_census_edgeiiot_pcap_timeline.py"


def make_pcap(path, seconds, endian="<", resolution=1_000_000):
    if endian == "<":
        magic = bytes.fromhex("d4c3b2a1" if resolution == 1_000_000 else "4d3cb2a1")
    else:
        magic = bytes.fromhex("a1b2c3d4" if resolution == 1_000_000 else "a1b23c4d")
    blob = magic + struct.pack(endian + "HHiIII", 2, 4, 0, 0, 65535, 1)
    for sec in seconds:
        fraction = 250000 if resolution == 1_000_000 else 250000000
        blob += struct.pack(endian + "IIII", sec, fraction, 3, 3) + b"XYZ"
    path.write_bytes(blob)


def command(root, manifest, out, extra=()):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--input-root", str(root),
         "--pcap-report", str(manifest), "--out", str(out),
         "--expected-files", "0", "--progress-every", "0", *extra],
        capture_output=True, text=True
    )


def fixture(tmp_path):
    root = tmp_path / "data"
    attack = root / "Attack traffic"
    attack.mkdir(parents=True)
    normal = root / "Normal traffic"
    normal.mkdir()
    start = int(datetime(2021, 11, 21, 23, 59, tzinfo=timezone.utc).timestamp())
    make_pcap(attack / "Backdoor_attack.pcap", [
        start, start + 360, start + 3660, start + 3659, start + 86400
    ])
    make_pcap(normal / "Distance.pcap", [start, start + 1], endian=">", resolution=1_000_000_000)
    manifest = tmp_path / "pcap_probe.json"
    manifest.write_text(json.dumps({
        "schema_version": "edgeiiot-original-pcap-timestamp-probe-v1",
        "files": [
            {"original_file": "Attack traffic/Backdoor_attack.pcap", "source_kind": "attack"},
            {"original_file": "Normal traffic/Distance.pcap", "source_kind": "normal"},
        ]
    }), encoding="utf-8")
    return root, manifest


def test_complete_census_day_gaps_and_order(tmp_path):
    root, manifest = fixture(tmp_path)
    out = tmp_path / "result.json"
    p = command(root, manifest, out)
    assert p.returncode == 0, p.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["all_files_scanned_to_eof"]
    assert report["total_packet_headers_scanned"] == 7
    assert report["status_counts"] == {"full_file_scanned": 2}
    attack = report["files"][0]
    assert attack["packets_scanned"] == 5
    assert attack["occupied_utc_days"] == 2
    assert attack["backward_adjacent_transitions"] == 1
    assert attack["forward_gap_counts_at_least_seconds"]["300"] == 3
    assert attack["forward_gap_counts_at_least_seconds"]["3600"] == 1
    assert len(attack["ten_largest_forward_adjacent_gaps"]) == 3
    assert "XYZ" not in out.read_text(encoding="utf-8")
    assert "Attack_label" not in out.read_text(encoding="utf-8")
    assert report["files"][1]["format"] == "pcap_nanosecond"


def test_packet_limit_disqualifies_full_census(tmp_path):
    root, manifest = fixture(tmp_path)
    out = tmp_path / "report.json"
    p = command(root, manifest, out, extra=("--max-packets-per-file", "2"))
    assert p.returncode == 0, p.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert not report["all_files_scanned_to_eof"]
    assert report["total_packet_headers_scanned"] == 4
    assert report["status_counts"] == {"packet_limit_reached": 2}


def test_bad_record_size_reports_incomplete(tmp_path):
    root, manifest = fixture(tmp_path)
    path = root / "Attack traffic" / "Backdoor_attack.pcap"
    blob = bytearray(path.read_bytes())
    struct.pack_into("<I", blob, 32, 999999)
    path.write_bytes(blob)
    out = tmp_path / "report.json"
    p = command(root, manifest, out)
    assert p.returncode == 0, p.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert not data["all_files_scanned_to_eof"]
    assert data["status_counts"]["invalid_record_size_or_truncation"] == 1


def test_manifest_traversal_refused_without_output(tmp_path):
    root, manifest = fixture(tmp_path)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["files"][0]["original_file"] = "../another.pcap"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "result.json"
    p = command(root, manifest, out)
    assert p.returncode != 0
    assert not out.exists()
