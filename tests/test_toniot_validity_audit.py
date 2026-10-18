"""Synthetic checks for streaming TON_IoT source validity diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_audit_toniot_validity.py"


def call(root: Path, out: Path, expected: int = 0, bin_seconds: int = 60):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--processed-root", str(root), "--out", str(out),
         "--expected-rows", str(expected), "--expected-files", "0",
         "--bin-seconds", str(bin_seconds), "--sample-per-type", "100"],
        capture_output=True, text=True,
    )


def test_disjoint_periods_attack_address_and_sampled_overlap(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    header = "ts,src_ip,src_port,dst_ip,dst_port,proto,label,type\n"
    rows1 = [
        "0,192.168.159.30,4,203.0.113.1,53,udp,0,normal",
        "10,203.0.113.10,4,203.0.113.1,53,udp,0,normal",
        "20,192.168.159.30,4,203.0.113.1,53,udp,1,scanning",
        "30,192.168.159.31,4,203.0.113.2,53,udp,1,scanning",
    ]
    rows2 = [
        "120,192.168.159.31,4,203.0.113.1,53,udp,1,dos",
        "200,203.0.113.10,4,203.0.113.3,53,udp,1,dos",
    ]
    (root / "Network_dataset_1.csv").write_text(header + "\n".join(rows1) + "\n", encoding="utf-8")
    (root / "Network_dataset_2.csv").write_text(header + "\n".join(rows2) + "\n", encoding="utf-8")
    out = tmp_path / "result.json"
    p = call(root, out, expected=6)
    assert p.returncode == 0, p.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["processed_rows_scanned"] == 6
    assert result["attack_family_intervals_strictly_disjoint"] is True
    assert result["per_type"]["scanning"]["rows_with_documented_attack_src_ip"] == 2
    assert result["per_type"]["dos"]["rows_with_documented_attack_src_ip"] == 1
    assert result["per_type"]["normal"]["rows_with_documented_attack_src_ip"] == 1
    assert result["per_type"]["dos"]["time_gaps_at_least_seconds"]["300"] == 0
    assert result["per_type"]["scanning"]["bins_with_benign"] == 1
    assert result["address_sample_associations"]["dos"]["sample_rows_src_also_in_benign_sample"] == 1
    assert result["address_sample_associations"]["dos"]["sample_rows_src_also_in_other_attack_family_samples"] == 1
    assert result["address_sample_associations"]["dos"]["distinct_five_tuple_identities_in_sample"] == 2
    assert "192.168.159.31" not in out.read_text(encoding="utf-8")
    assert "203.0.113" not in out.read_text(encoding="utf-8")


def test_detect_overlapping_family_intervals_and_time_gaps(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    (root / "Network_dataset_1.csv").write_text(
        "ts,src_ip,dst_ip,src_port,dst_port,proto,label,type\n"
        "0,a,b,1,2,tcp,1,scanning\n"
        "2,a,b,1,2,tcp,1,dos\n"
        "500,a,b,1,2,tcp,1,scanning\n", encoding="utf-8"
    )
    out = tmp_path / "report.json"
    p = call(root, out, expected=3)
    assert p.returncode == 0, p.stderr
    j = json.loads(out.read_text(encoding="utf-8"))
    assert j["attack_family_intervals_strictly_disjoint"] is False
    assert j["per_type"]["scanning"]["time_gaps_at_least_seconds"]["300"] == 1
    assert j["per_type"]["scanning"]["backwards_timestamp_transitions_within_type"] == 0


def test_reject_incomplete_data_without_report(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    (root / "Network_dataset_1.csv").write_text(
        "ts,label,type,src_ip,dst_ip,src_port,dst_port,proto\n"
        "1,0,normal,a,b,1,2,tcp\n", encoding="utf-8"
    )
    out = tmp_path / "report.json"
    p = call(root, out, expected=999)
    assert p.returncode != 0
    assert not out.exists()


def test_reject_bad_labels_and_malformed_width(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    target = root / "Network_dataset_1.csv"
    target.write_text(
        "ts,label,type,src_ip,dst_ip,src_port,dst_port,proto\n"
        "1,1,normal,a,b,1,2,tcp\n", encoding="utf-8"
    )
    out = tmp_path / "report.json"
    p = call(root, out, expected=1)
    assert p.returncode != 0
    assert not out.exists()
    target.write_text(
        "ts,label,type,src_ip,dst_ip,src_port,dst_port,proto\n"
        "1,0,normal,a,b,1,2\n", encoding="utf-8"
    )
    p = call(root, out, expected=1)
    assert p.returncode != 0
    assert not out.exists()
