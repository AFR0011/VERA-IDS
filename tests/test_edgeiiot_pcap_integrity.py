"""Synthetic tests of bounded PCAP record failure and shared metadata evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_probe_edgeiiot_pcap_integrity.py"


def make_pcap(path, records, malformed=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = bytes.fromhex("d4c3b2a1") + struct.pack("<HHiIII", 2, 4, 0, 0, 65535, 1)
    for sec, frac, captured, original in records:
        blob += struct.pack("<IIII", sec, frac, captured, original) + b"A" * captured
    if malformed is not None:
        blob += struct.pack("<IIII", *malformed)
    path.write_bytes(blob)


def setup(tmp_path, changed_second=False):
    root = tmp_path / "sources"
    scanner = root / "Attack traffic" / "Vulnerability scanner attack.pcap"
    modbus = root / "Normal traffic" / "Modbus.pcap"
    temperature = root / "Normal traffic" / "Temperature_and_Humidity.pcap"
    sec = int(datetime(2021, 12, 20, tzinfo=timezone.utc).timestamp()) + 1400
    expected = [(sec + i, i, 3, 3) for i in range(3)]
    make_pcap(scanner, expected[:2], malformed=(sec + 10, 0, 200000, 3))
    make_pcap(modbus, expected)
    different = expected[:1] + [(sec + 1, 0, 3, 3)] + expected[2:] if changed_second else expected
    make_pcap(temperature, different)
    entries = [
        {"source_file": "Attack traffic/Vulnerability scanner attack.pcap",
         "source_kind": "attack", "status": "invalid_record_size_or_truncation", "packets_scanned": 2},
        {"source_file": "Normal traffic/Modbus.pcap", "source_kind": "normal",
         "status": "full_file_scanned", "packets_scanned": 3},
        {"source_file": "Normal traffic/Temperature_and_Humidity.pcap", "source_kind": "normal",
         "status": "full_file_scanned", "packets_scanned": 3},
    ]
    entries += [{"source_file": f"placeholder-{i}.pcap", "source_kind": "attack",
                 "status": "full_file_scanned", "packets_scanned": 0} for i in range(21)]
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "schema_version": "edgeiiot-full-pcap-header-timeline-census-v1",
        "file_count": 24, "max_packets_per_file": 0, "files": entries,
    }), encoding="utf8")
    return root, report


def run(root, report, out):
    return subprocess.run([
        sys.executable, str(SCRIPT), "--input-root", str(root),
        "--timeline-report", str(report), "--out", str(out),
    ], capture_output=True, text=True)


def test_structural_reason_and_equal_timestamp_length_sequence(tmp_path):
    root, report = setup(tmp_path)
    out = tmp_path / "out.json"
    p = run(root, report, out)
    assert p.returncode == 0, p.stderr
    result = json.loads(out.read_text())
    scanner = result["vulnerability_scanner"]
    assert scanner["headers_valid"] == 2
    assert scanner["agrees_with_prior_record_count"]
    assert scanner["status"] == "invalid_record_header"
    assert set(scanner["failed_constraints"]) == {
        "captured_length_exceeds_original_length",
        "captured_length_exceeds_snapshot_length",
        "captured_length_exceeds_remaining_file_bytes",
    }
    same = result["normal_source_day_comparison"]
    assert same["ordered_packet_header_sequences_identical_for_day"]
    assert same["modbus"]["packets_on_requested_day"] == 3
    assert same["temperature_and_humidity"]["packets_on_requested_day"] == 3
    assert "AAAA" not in out.read_text()


def test_different_packet_timestamp_breaks_fingerprint(tmp_path):
    root, report = setup(tmp_path, changed_second=True)
    out = tmp_path / "out.json"
    p = run(root, report, out)
    assert p.returncode == 0, p.stderr
    result = json.loads(out.read_text())
    assert not result["normal_source_day_comparison"]["ordered_packet_header_sequences_identical_for_day"]


def test_rejects_source_traversal(tmp_path):
    root, report = setup(tmp_path)
    doc = json.loads(report.read_text())
    doc["files"][0]["source_file"] = "../untrusted/Vulnerability scanner attack.pcap"
    report.write_text(json.dumps(doc))
    out = tmp_path / "out.json"
    p = run(root, report, out)
    assert p.returncode != 0
    assert not out.exists()


def test_refuses_capped_prior_scan(tmp_path):
    root, report = setup(tmp_path)
    doc = json.loads(report.read_text())
    doc["max_packets_per_file"] = 5
    report.write_text(json.dumps(doc))
    out = tmp_path / "out.json"
    p = run(root, report, out)
    assert p.returncode != 0
    assert not out.exists()
