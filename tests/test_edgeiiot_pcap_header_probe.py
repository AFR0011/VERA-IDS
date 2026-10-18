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


def test_author_distribution_aliases_14_attacks_10_normal(tmp_path):
    """Enforce all 24 source-manifest pairs, not fuzzy string matching."""
    attack_aliases = {
        "Backdoor_attack": "Backdoor_attack",
        "DDoS HTTP Flood Attacks": "DDoS_HTTP_Flood_attack",
        "DDoS ICMP Flood Attacks": "DDoS_ICMP_Flood_attack",
        "DDoS TCP SYN Flood Attacks": "DDoS_TCP_SYN_Flood_attack",
        "DDoS UDP Flood Attacks": "DDoS_UDP_Flood_attack",
        "MITM (ARP spoofing + DNS) Attack": "MITM_attack",
        "OS Fingerprinting attack": "OS_Fingerprinting_attack",
        "Password attacks": "Password_attack",
        "Port Scanning attack": "Port_Scanning_attack",
        "Ransomware attack": "Ransomware_attack",
        "SQL injection attack": "SQL_injection_attack",
        "Uploading attack": "Uploading_attack",
        "Vulnerability scanner attack": "Vulnerability_scanner_attack",
        "XSS attacks": "XSS_attack",
    }
    normal_stems = [
        "Distance", "Flame_Sensor", "Heart_Rate", "IR_Receiver", "Modbus",
        "phValue", "Soil_Moisture", "Sound_Sensor",
        "Temperature_and_Humidity", "Water_Level",
    ]
    root = tmp_path / "data"
    attack = root / "Attack traffic"
    attack.mkdir(parents=True)
    for pcap_stem, csv_stem in attack_aliases.items():
        build_pcap(attack / (pcap_stem + ".pcap"))
        (attack / (csv_stem + ".csv")).write_text(
            "frame.time,Attack_label,Attack_type\n", encoding="utf-8"
        )
    for stem in normal_stems:
        normal = root / "Normal traffic" / stem
        normal.mkdir(parents=True)
        build_pcap(normal / (stem + ".pcap"))
        (normal / (stem + ".csv")).write_text(
            "frame.time,Attack_label,Attack_type\n", encoding="utf-8"
        )
    out = tmp_path / "report.json"
    proc = subprocess.run([
        sys.executable, str(SCRIPT), "--input-root", str(root),
        "--out", str(out),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["original_pcap_files"] == 24
    assert data["corresponding_original_csv_files"] == 24
    assert sum(f["pairing_basis"] == "documented_filename_alias" for f in data["files"]) == 13
    assert sum(f["pairing_basis"] == "same_stem" for f in data["files"]) == 11
    assert all(f["matching_csv_present"] for f in data["files"])
    assert data["first_packet_plausible_epoch_files"] == 24


def test_unknown_capture_name_not_fuzzy_matched(tmp_path):
    root = tmp_path / "data" / "Attack traffic"
    root.mkdir(parents=True)
    build_pcap(root / "Some Unknown Attack.pcap")
    (root / "XSS_attack.csv").write_text("frame.time\n", encoding="utf-8")
    out = tmp_path / "out.json"
    proc = run(root.parent, out)
    assert proc.returncode != 0
    assert "Unmatched named source files" in proc.stderr
    assert not out.exists()
