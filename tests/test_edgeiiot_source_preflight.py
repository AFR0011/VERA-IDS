from pathlib import Path
import json
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "17_preflight_edgeiiot_source.py"


def test_inventory_and_literal_token_census(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "Attack traffic").mkdir()
    (root / "Normal traffic").mkdir()
    (root / "Attack traffic" / "Backdoor_attack.csv").write_text(
        "frame.time,ip.src_host,label\n1,a,x\n", encoding="utf8"
    )
    (root / "Normal traffic" / "Distance.pcap").write_bytes(b"abc")
    (root / "Readme.txt").write_text("source", encoding="utf8")
    (root / "ML-EdgeIIoT-dataset.csv").write_text(
        "Attack_label,Attack_type,dns.qry.name.len,mqtt.conack.flags,mqtt.protoname,mqtt.topic\n"
        "0,Normal,0,0,0,0\n"
        "1,MITM,0.0,0.0,0.0,0.0\n"
        "1,XSS,0,0.0,0.0,0.0\n", encoding="utf8"
    )
    out = tmp_path / "out.json"
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "--input-root", str(root),
         "--out", str(out), "--audit-curated"],
        capture_output=True, text=True,
    )
    assert p.returncode == 0, p.stderr
    result = json.loads(out.read_text(encoding="utf8"))
    assert result["csv_files"] == 2
    assert result["capture_files"] == 1
    counts = result["curated_audits"][0]["exact_text_zero_spellings_by_column_and_label"]["dns.qry.name.len"]
    assert counts["normal_label_0"] == {"0": 1}
    assert counts["attack_label_1"] == {"0": 1, "0.0": 1}
    assert all("a,x" not in json.dumps(x) for x in result["manifest"])


def test_missing_folder_fail(tmp_path):
    out = tmp_path / "out.json"
    p = subprocess.run([
        sys.executable, str(SCRIPT), "--input-root", str(tmp_path / "none"),
        "--out", str(out)
    ], capture_output=True, text=True)
    assert p.returncode != 0 and not out.exists()


def test_require_curated_labels(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "DNN-EdgeIIoT-dataset.csv").write_text("foo,bar\n1,2\n", encoding="utf8")
    out = tmp_path / "out.json"
    p = subprocess.run([
        sys.executable, str(SCRIPT), "--input-root", str(root),
        "--out", str(out), "--audit-curated"
    ], capture_output=True, text=True)
    assert p.returncode != 0 and not out.exists()
