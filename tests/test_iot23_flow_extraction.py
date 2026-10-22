"""Synthetic extraction tests for the IoT-23 read-only source preflight."""
from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_extract_iot23_flow_sources.py"


def make_tar(path, *, attack_count=20, benign_count=3, extra_bad=None):
    base = "opt/Malware-Project/BigDataset/IoTScenarios"
    with tarfile.open(path, "w:gz") as archive:
        for kind, n in (("CTU-IoT-Malware-Capture", attack_count),
                        ("CTU-Honeypot-Capture", benign_count)):
            for i in range(n):
                ident = f"{kind}-{i}-1"
                entries = {
                    f"{base}/{ident}/README.md": b"# Sample native source\n",
                    f"{base}/{ident}/bro/conn.log.labeled": (
                        b"#fields\tts\tuid\tid.orig_h\tid.resp_h\tlabel   detailed-label\n"
                        b"1540000000.0\tu0\t192.0.2.1\t192.0.2.2\tMalicious DDoS\n"
                    ),
                    f"{base}/{ident}/malware_binary.bin": b"MALWARE_NOT_TO_EXTRACT",
                    f"{base}/{ident}/capture.pcap": b"PCAP_NOT_TO_EXTRACT",
                }
                for member_name, data in entries.items():
                    info = tarfile.TarInfo(member_name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        if extra_bad is not None:
            info = tarfile.TarInfo(extra_bad)
            info.size = 4
            archive.addfile(info, io.BytesIO(b"evil"))


def run(archive, root, out, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--archive", str(archive),
         "--output-root", str(root), "--out", str(out), *extra],
        capture_output=True, text=True,
    )


def test_selects_23_readmes_and_labels_not_pcap_or_binary(tmp_path):
    source = tmp_path / "iot_23_datasets_full.tar.gz"
    make_tar(source)
    root = tmp_path / "extract"
    report = tmp_path / "manifest.json"
    result = run(source, root, report)
    assert result.returncode == 0, result.stderr
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert doc["extracted_scenarios"] == 23
    assert doc["extracted_files"] == 46
    assert len(list(root.rglob("conn.log.labeled"))) == 23
    assert len(list(root.rglob("README.md"))) == 23
    assert not list(root.rglob("*.pcap"))
    assert not list(root.rglob("*.bin"))
    assert "MALWARE_NOT_TO_EXTRACT" not in report.read_text()


def test_refuses_existing_files_without_overwriting(tmp_path):
    source = tmp_path / "archive.tar.gz"
    make_tar(source, attack_count=1, benign_count=0)
    root = tmp_path / "root"
    out = tmp_path / "manifest.json"
    first = run(source, root, out, "--expected-scenarios", "1")
    assert first.returncode == 0, first.stderr
    existing = next(root.rglob("conn.log.labeled"))
    old_data = existing.read_bytes()
    second = run(source, root, tmp_path / "retry.json", "--expected-scenarios", "1")
    assert second.returncode != 0
    assert existing.read_bytes() == old_data


def test_rejects_selected_traversal_member(tmp_path):
    source = tmp_path / "bad.tar.gz"
    malicious = "../CTU-IoT-Malware-Capture-1-1/README.md"
    make_tar(source, attack_count=0, benign_count=0, extra_bad=malicious)
    out = tmp_path / "manifest.json"
    result = run(source, tmp_path / "extracted", out, "--expected-scenarios", "0")
    assert result.returncode != 0
    assert not out.exists()


def test_partial_archive_does_not_silently_claim_23(tmp_path):
    source = tmp_path / "small.tar.gz"
    make_tar(source, attack_count=1, benign_count=1)
    out = tmp_path / "manifest.json"
    result = run(source, tmp_path / "extracted", out)
    assert result.returncode == 2, result.stderr
    doc = json.loads(out.read_text())
    assert doc["extracted_scenarios"] == 2
    assert "Scenario count differs" in doc["problems"][0]
