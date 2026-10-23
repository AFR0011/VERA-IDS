"""Read-only IoT-23 source support audit synthetic regression tests."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_audit_iot23_capture_support.py"
FIELDS = ("ts uid id.orig_h id.resp_h label detailed-label").split()


def make_capture(root: Path, name: str, rows: list[tuple[str, str]],
                 *, compressed: bool = False, broken: bool = False) -> None:
    folder = root / "opt" / "Malware-Project" / "BigDataset" / "IoTScenarios" / name / "bro"
    folder.mkdir(parents=True)
    (folder.parent / "README.md").write_text("# scenario\n", encoding="utf-8")
    # Match native Zeek quirk: earlier columns tab-separated; final label
    # columns can be whitespace-joined in the final tab field.
    content = "#separator \\x09\n#fields\t" + "\t".join(FIELDS[:-2]) + "\tlabel   detailed-label\n"
    for index, (main, detail) in enumerate(rows):
        line = (
            f"1540000000.{index:06d}\tuid{index}\t192.0.2.1\t192.0.2.2\t"
            f"{main}   {detail}\n"
        )
        content += line
    if broken:
        content += "1540000001.000000\tuid999\t192.0.2.1\n"
    filename = folder / ("conn.log.labeled.gz" if compressed else "conn.log.labeled")
    if compressed:
        with gzip.open(filename, "wt", encoding="utf-8") as out:
            out.write(content)
    else:
        filename.write_text(content, encoding="utf-8")


def invoke(root, out, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--input-root", str(root),
         "--out", str(out), *extra],
        capture_output=True, text=True,
    )


def test_native_mixed_delimiter_preflight_and_privacy(tmp_path):
    root = tmp_path / "source"
    for i in range(20):
        make_capture(root, f"CTU-IoT-Malware-Capture-{i}-1", [
            ("Malicious", "DDoS"), ("Malicious", "C&C"), ("Benign", "-")
        ])
    for i in range(3):
        make_capture(root, f"CTU-Honeypot-Capture-{i}-1", [("Benign", "-")])
    out = tmp_path / "preflight.json"
    proc = invoke(root, out, "--phase", "preflight", "--sample-rows-per-capture", "2")
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["capture_scenarios_found"] == 23
    assert result["source_kinds"] == {"benign_device": 3, "malware": 20}
    assert result["total_rows_inspected"] == 43
    assert not result["complete_source_row_census"]
    ddos = next(item for item in result["native_exact_detailed_label_support"]
                if item["native_exact_detailed_label"] == "DDoS")
    assert ddos["captures_with_observed_rows"] == 20
    assert len(result["schema_variants"]) == 1
    assert "192.0.2." not in out.read_text()
    assert "uid999" not in out.read_text()


def test_complete_census_exact_native_labels_and_gzip(tmp_path):
    root = tmp_path / "dataset"
    make_capture(root, "CTU-IoT-Malware-Capture-34-1",
                 [("Malicious", "C&C"), ("Benign", "-"), ("Malicious", "DDoS")],
                 compressed=True)
    make_capture(root, "CTU-Honeypot-Capture-4-1",
                 [("Benign", "-"), ("Benign", "-")])
    out = tmp_path / "census.json"
    proc = invoke(root, out, "--phase", "census", "--expected-captures", "0")
    assert proc.returncode == 0, proc.stderr
    info = json.loads(out.read_text())
    assert info["complete_source_row_census"]
    assert info["total_rows_inspected"] == 5
    assert {v["native_exact_detailed_label"]: v["captures_with_observed_rows"]
            for v in info["native_exact_detailed_label_support"]} == {
                "-": 2, "C&C": 1, "DDoS": 1
            }
    assert sum(len(x["utc_day_counts"]) for x in info["capture_records"]) == 2


def test_partial_when_malformed_row(tmp_path):
    root = tmp_path / "source"
    make_capture(root, "CTU-IoT-Malware-Capture-1-1",
                 [("Malicious", "DDoS")], broken=True)
    out = tmp_path / "partial.json"
    proc = invoke(root, out, "--expected-captures", "0")
    assert proc.returncode == 2, proc.stderr
    report = json.loads(out.read_text())
    assert report["problems"] == ["Malformed source rows or invalid timestamps observed"]
    assert report["capture_records"][0]["rows_malformed_width"] == 1
    assert len(report["capture_records"][0]["first_malformed_line_descriptors"]) == 1


def test_unrecognized_directory_refused(tmp_path):
    root = tmp_path / "source"
    make_capture(root, "CTU-IoT-Malware-Capture-1-1", [("Malicious", "DDoS")])
    old = root / "opt" / "Malware-Project" / "BigDataset" / "IoTScenarios" / "CTU-IoT-Malware-Capture-1-1"
    old.rename(old.with_name("not-a-certified-capture"))
    out = tmp_path / "should_not_exist.json"
    proc = invoke(root, out, "--expected-captures", "0")
    assert proc.returncode != 0
    assert not out.exists()


def test_fail_closed_without_native_detailed_label(tmp_path):
    root = tmp_path / "source"
    folder = root / "CTU-IoT-Malware-Capture-1-1" / "bro"
    folder.mkdir(parents=True)
    (folder.parent / "README.md").write_text("fixture", encoding="utf-8")
    (folder / "conn.log.labeled").write_text(
        "#fields\tts\tuid\tid.orig_h\tid.resp_h\tlabel\n"
        "1540000000.0\tuid0\t192.0.2.1\t192.0.2.2\tMalicious\n", encoding="utf-8")
    out = tmp_path / "should_not_exist.json"
    proc = invoke(root, out, "--expected-captures", "0")
    assert proc.returncode != 0
    assert not out.exists()


def test_directly_extracted_capture_folders(tmp_path):
    """Recognize data/raw/IoT23/CTU-* layout without re-extracting archive."""
    root = tmp_path / "IoT23"
    malware = "CTU-IoT-Malware-Capture-34-1"
    normal = "CTU-Honeypot-Capture-4-1"
    make_capture(root, malware, [("Malicious", "DDoS"), ("Benign", "-")])
    make_capture(root, normal, [("Benign", "-")])
    scenarios = root / "opt" / "Malware-Project" / "BigDataset" / "IoTScenarios"
    for name in (malware, normal):
        (scenarios / name).rename(root / name)
    output = tmp_path / "direct_root.json"
    proc = invoke(root, output, "--expected-captures", "0", "--phase", "preflight")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["capture_scenarios_found"] == 2
    assert report["source_kinds"] == {"benign_device": 1, "malware": 1}
    assert report["total_rows_inspected"] == 3
    assert {row["capture_id"] for row in report["capture_records"]} == {malware, normal}
    assert {row["log_relative"] for row in report["capture_records"]} == {
        f"{malware}/bro/conn.log.labeled", f"{normal}/bro/conn.log.labeled"
    }
    assert all(row["readme_present"] for row in report["capture_records"])


def test_nested_somfy_pcap_only_folders_reported_without_failing_labels(tmp_path):
    """Native Somfy-01 has Zeek labels; Somfy-02..06 have source PCAPs only."""
    root = tmp_path / "IoT23"
    capture = root / "CTU-Honeypot-Capture-7-1"
    labeled = capture / "Somfy-01" / "bro"
    labeled.mkdir(parents=True)
    (labeled / "conn.log.labeled").write_text(
        "#fields\tts\tuid\tid.orig_h\tid.resp_h\tlabel   detailed-label\n"
        "1540000000.0\tu0\t192.0.2.1\t192.0.2.2\tbenign -\n",
        encoding="utf-8",
    )
    (capture / "Somfy-01" / "capture.pcap").write_bytes(b"PCAP_ONLY_METADATA")
    for index in range(2, 7):
        child = capture / f"Somfy-{index:02d}"
        child.mkdir()
        (child / f"capture-{index}.pcap").write_bytes(b"PCAP_ONLY_METADATA")
    out = tmp_path / "somfy_report.json"
    proc = invoke(root, out, "--expected-captures", "0")
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["problems"] == []
    assert doc["capture_scenarios_found"] == 1
    item = doc["capture_records"][0]
    assert item["capture_id"] == "CTU-Honeypot-Capture-7-1"
    assert item["rows_inspected"] == 1
    assert item["native_binary_label_counts"] == {"benign": 1}
    assert item["readme_present"] is False
    assert item["root_pcap_file_count"] == 0
    assert item["pcap_file_count_including_nested"] == 6
    assert item["nested_pcap_only_directories"] == [
        "Somfy-02", "Somfy-03", "Somfy-04", "Somfy-05", "Somfy-06"
    ]
    assert len(doc["source_metadata_warnings"]) == 2
    assert "README.md absent" in doc["source_metadata_warnings"][0]
    assert "Somfy-02" in doc["source_metadata_warnings"][1]
    assert "PCAP_ONLY_METADATA" not in out.read_text(encoding="utf-8")


def test_one_selected_capture_root_allows_full_source_count(tmp_path):
    """Full census of one *selected capture*, not the 23-scenario corpus."""
    root = tmp_path / "IoT23"
    selected = "CTU-IoT-Malware-Capture-34-1"
    make_capture(root, selected, [
        ("Benign", "-"), ("Malicious", "C&C"), ("Malicious", "DDoS")
    ])
    source = (root / "opt" / "Malware-Project" / "BigDataset"
              / "IoTScenarios" / selected)
    out = tmp_path / "one_capture.json"
    result = invoke(source, out, "--expected-captures", "1", "--phase", "census")
    assert result.returncode == 0, result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["complete_source_row_census"] is True
    assert report["capture_scenarios_found"] == 1
    assert report["total_rows_inspected"] == 3
    assert report["capture_records"][0]["log_relative"] == "bro/conn.log.labeled"
    assert report["native_exact_detailed_label_support"][1]["native_exact_detailed_label"] == "C&C"
