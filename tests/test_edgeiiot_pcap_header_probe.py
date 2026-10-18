"""Synthetic original Edge-IIoTset PCAP timestamp and file pairing tests."""
import json
from pathlib import Path
import struct
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_probe_edgeiiot_pcap_headers.py"


def build_pcap(path, *, endian="<", resolution=1_000_000, seconds=1640000000, fraction=500000):
    if endian == "<":
        magic = bytes.fromhex("d4c3b2a1") if resolution == 1_000_000 else bytes.fromhex("4d3cb2a1")
    else:
        magic = bytes.fromhex("a1b2c3d4") if resolution == 1_000_000 else bytes.fromhex("a1b23c4d")
    path.write_bytes(
        magic + struct.pack(endian + "HHiIII", 2, 4, 0, 0, 65535, 1)
        + struct.pack(endian + "IIII", seconds, fraction, 3, 3) + b"ABC"
        + struct.pack(endian + "IIII", seconds + 1, fraction, 3, 3) + b"XYZ"
    )


def run(root, out):
    return subprocess.run([
        sys.executable, str(SCRIPT), "--input-root", str(root),
        "--out", str(out), "--expected-original-files", "0",
    ], capture_output=True, text=True)


def test_endianness_resolutions_and_no_packet_payload_export(tmp_path):
    root = tmp_path / "data"
    cases = [
        ("Attack traffic", "Backdoor_attack", {"endian": "<", "resolution": 1_000_000}),
        ("Normal traffic", "Distance", {"endian": ">", "resolution": 1_000_000_000, "fraction": 500_000_000}),
    ]
    for kind, base, args in cases:
        target = root / kind
        target.mkdir(parents=True)
        build_pcap(target / (base + ".pcap"), **args)
        (target / (base + ".csv")).write_text("frame.time,Attack_label,Attack_type\nabc,0,Normal\n")
    out = tmp_path / "result.json"
    proc = run(root, out)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["original_pcap_files"] == 2
    assert result["first_packet_plausible_epoch_files"] == 2
    assert result["status_counts"] == {"packet_sample_read": 2}
    assert all(f["sampled_packet_headers"] == 2 for f in result["files"])
    assert not any(x in out.read_text() for x in ("ABC", "XYZ", "192.0.2.1"))


def test_pcapng_recognized_but_not_misparsed(tmp_path):
    root = tmp_path / "data" / "Attack traffic"
    root.mkdir(parents=True)
    (root / "MITM_attack.csv").write_text("frame.time\n1\n")
    (root / "MITM_attack.pcap").write_bytes(bytes.fromhex("0a0d0d0a") + b"12345678")
    out = tmp_path / "result.json"
    proc = run(root.parent, out)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["status_counts"] == {"pcapng_not_parsed": 1}
    assert result["first_packet_plausible_epoch_files"] == 0


def test_invalid_packet_length_flagged(tmp_path):
    root = tmp_path / "data" / "Attack traffic"
    root.mkdir(parents=True)
    build_pcap(root / "Ransomware_attack.pcap")
    blob = bytearray((root / "Ransomware_attack.pcap").read_bytes())
    struct.pack_into("<I", blob, 32, 999_999)
    (root / "Ransomware_attack.pcap").write_bytes(blob)
    (root / "Ransomware_attack.csv").write_text("frame.time\n1\n")
    out = tmp_path / "result.json"
    proc = run(root.parent, out)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["status_counts"] == {"invalid_packet_length": 1}


def test_missing_csv_pair_refuses_report(tmp_path):
    root = tmp_path / "data" / "Attack traffic"
    root.mkdir(parents=True)
    build_pcap(root / "Backdoor_attack.pcap")
    out = tmp_path / "result.json"
    proc = run(root.parent, out)
    assert proc.returncode != 0
    assert not out.exists()
