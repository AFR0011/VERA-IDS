from __future__ import annotations

import csv
import hashlib
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_PATH = re.compile(r"(?i)(?<![A-Za-z])(?:[A-Z]:[\\/]|/home/[^/\s]+/|/Users/[^/\s]+/)")
PATTERN_DEFINITION_FILES = {"scripts/release_audit.py", "tests/test_release_contents.py"}


def release_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line and (ROOT / line).is_file()]


def test_curated_summaries_have_manifest_entries() -> None:
    manifest = ROOT / "outputs" / "summaries" / "SOURCE_MANIFEST.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    public_paths = {row["public_path"] for row in rows}
    summaries = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "outputs" / "summaries").iterdir()
        if path.is_file() and path.name not in {"SOURCE_MANIFEST.csv", "README.md"}
    }
    assert summaries == public_paths
    for row in rows:
        path = ROOT / row["public_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["sha256"]
        assert path.stat().st_size == int(row["bytes"])


def test_public_text_has_no_absolute_local_paths() -> None:
    allowed = {"AUDIT_REPORT.md", "DEV_LOG.md", "QA_REPORT.md", "PUBLICATION_CHECKLIST.md"}
    failures = []
    for path in release_files():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in PATTERN_DEFINITION_FILES:
            continue
        if path.suffix.lower() not in {".cff", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yml"}:
            continue
        if path.name in allowed:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if ABSOLUTE_PATH.search(content):
            failures.append(path.relative_to(ROOT).as_posix())
    assert failures == []


def test_no_raw_data_or_model_artifacts_are_selected() -> None:
    forbidden = {".gz", ".joblib", ".pkl", ".pickle", ".parquet"}
    selected = [
        path.relative_to(ROOT).as_posix()
        for path in release_files()
        if path.is_file() and path.suffix.lower() in forbidden
    ]
    assert selected == []


def test_citation_cff_has_required_fields() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["title"]
    assert citation["authors"][0]["family-names"] == "Farrokhnejad"


def test_relative_markdown_links_resolve() -> None:
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    failures = []
    for path in release_files():
        if path.suffix.lower() != ".md":
            continue
        for target in pattern.findall(path.read_text(encoding="utf-8", errors="replace")):
            clean = target.strip().strip("<>").split("#", 1)[0]
            if not clean or clean.startswith(("http://", "https://", "mailto:")):
                continue
            if not (path.parent / clean).resolve().exists():
                failures.append(f"{path.relative_to(ROOT).as_posix()} -> {target}")
    assert failures == []


def test_no_dynamic_legacy_runtime_or_removed_lanes() -> None:
    forbidden = (
        "run_legacy_main",
        "load_legacy_module",
        "spec_from_file_location",
        "paper_reproduction_bridge",
        "11_run_paper_reproduction_models",
        "12_run_reference_framework_eval",
    )
    failures: list[str] = []
    for path in release_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(("docs/source/", "tests/")) or path.suffix.lower() not in {".py", ".yml", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(value in text for value in forbidden):
            failures.append(relative)
    assert failures == []
