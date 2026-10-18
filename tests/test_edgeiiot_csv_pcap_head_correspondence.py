"""Tests for bounded original CSV/PCAP head-time correspondence, no inferred sessions."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_compare_edgeiiot_csv_pcap_heads.py"


def pcap(path, times):
    contents = bytes.fromhex("d4c3b2a1") + struct.pack("<HHiIII", 2, 4, 0, 0, 65535, 1)
    for timestamp in times:
        contents += (
            struct.pack("<IIII", int(timestamp.timestamp()), timestamp.microsecond, 4, 4)
            + b"DATA"
        )
    path.write_bytes(contents)


def invoke(root, manifest, out):
    return subprocess.run([
        sys.executable, str(SCRIPT), "--input-root", str(root),
        "--pcap-report", str(manifest), "--out", str(out),
        "--expected-pairs", "0"
    ], capture_output=True, text=True)


def populate(root):
    folder = root / "Attack traffic"
    folder.mkdir(parents=True)
    back_csv = folder / "Backdoor_attack.csv"
    back_csv.write_text(
        "frame.time,Attack_label,Attack_type\n"
        " 2021 17:58:23.150209000 ,1,Backdoor\n"
        " 2021 17:58:23.150231000 ,1,Backdoor\n",
        encoding="utf-8",
    )
    base = datetime(2021, 12, 5, 16, 58, 23, tzinfo=timezone.utc)
    back_pcap = folder / "Backdoor_attack.pcap"
    pcap(back_pcap, [base.replace(microsecond=150209), base.replace(microsecond=150231)])
    mitm_csv = folder / "MITM_attack.csv"
    mitm_csv.write_text(
        "frame.time,Attack_label,Attack_type\n0.0,1,MITM\n6.0,1,MITM\n",
        encoding="utf-8",
    )
    mitm_pcap = folder / "MITM (ARP spoofing + DNS) Attack.pcap"
    pcap(mitm_pcap, [base.replace(microsecond=123456), base.replace(microsecond=456789)])
    pairs = []
    for pc, cs in [(back_pcap, back_csv), (mitm_pcap, mitm_csv)]:
        pairs.append({
            "original_file": pc.relative_to(root).as_posix(),
            "paired_csv_filename": cs.relative_to(root).as_posix(),
            "pairing_basis": "documented_filename_alias" if "MITM" in pc.name else "same_stem",
        })
    manifest = root / "pcap_report.json"
    manifest.write_text(json.dumps({
        "schema_version": "edgeiiot-original-pcap-timestamp-probe-v1",
        "files": pairs
    }), encoding="utf-8")
    return manifest


def test_exact_first_rows_match_plus_one_hour_without_recovery(tmp_path):
    root = tmp_path / "src"
    manifest = populate(root)
    out = tmp_path / "aggregate.json"
    proc = invoke(root, manifest, out)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["pair_count"] == 2
    assert data["total_ordinal_positions_compared"] == 4
    assert data["total_parsed_year_time_fragments"] == 2
    assert data["pairs_with_at_least_one_exact_match"] == 1
    back, mitm = data["files"]
    assert back["exact_ordinal_matches_by_candidate_hour_offset"] == {"1": 2}
    assert back["max_matched_ordinals"] == 2
    assert mitm["max_matched_ordinals"] == 0
    rendered = out.read_text(encoding="utf-8")
    assert "17:58" not in rendered and "16:58" not in rendered
    assert "DATA" not in rendered


def test_untrusted_manifest_traversal_rejected(tmp_path):
    root = tmp_path / "src"
    manifest = populate(root)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["files"][0]["paired_csv_filename"] = "../../outside.csv"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "no_report.json"
    proc = invoke(root, manifest, out)
    assert proc.returncode != 0
    assert not out.exists()


def test_wrong_manifest_version_refused(tmp_path):
    root = tmp_path / "src"
    manifest = populate(root)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["schema_version"] = "unknown"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "no_report.json"
    proc = invoke(root, manifest, out)
    assert proc.returncode != 0 and not out.exists()


def test_duplicate_pair_refused(tmp_path):
    root = tmp_path / "src"
    manifest = populate(root)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["files"].append(doc["files"][0])
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "no_report.json"
    proc = invoke(root, manifest, out)
    assert proc.returncode != 0 and not out.exists()
