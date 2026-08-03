#!/usr/bin/env python3
"""Build a path-safe release inventory and scan a candidate public tree.

The scanner reports only relative paths and finding categories. It never writes
matched values to its output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TEXT_SUFFIXES = {
    ".cfg", ".cff", ".csv", ".ini", ".json", ".md", ".py", ".rst",
    ".lock", ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
BINARY_MODEL_SUFFIXES = {".joblib", ".pkl", ".pickle", ".onnx", ".pt", ".pth"}
ARCHIVE_SUFFIXES = {".7z", ".gz", ".rar", ".tar", ".tgz", ".zip"}

FINDING_PATTERNS = {
    "credential_assignment": re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd|"
        r"github[_-]?token|aws[_-]?(?:access|secret)|connection[_-]?string)\s*[:=]"
    ),
    "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "email_address": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "absolute_local_path": re.compile(
        r"(?i)(?<![A-Za-z])(?:[A-Z]:[\\/]|/home/[^/\s]+/|/Users/[^/\s]+/)"
    ),
    "private_network": re.compile(
        r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.(?:\d{1,3}\.)\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})\b"
    ),
}

PATTERN_DEFINITION_FILES = {"scripts/release_audit.py", "tests/test_release_contents.py"}

SENSITIVE_NAME = re.compile(
    r"(?i)(?:^|[._-])(?:\.env|credential|secret|password|passwd|private[_-]?key|"
    r"token|passport|id[_-]?card|signature|reviewer|supervisor|correspondence|"
    r"email|tracked[_-]?changes)(?:$|[._-])"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def category_for(relative: str, suffix: str) -> tuple[str, str, str, str]:
    lower = relative.lower().replace("\\", "/")
    name = Path(lower).name

    if "/__pycache__/" in f"/{lower}" or suffix == ".pyc":
        return "cache", "Python bytecode/cache", "exclude", "generated cache"
    if suffix == ".tmp" or name.startswith("~$") or name.endswith((".bak", ".old", "~")):
        return "temporary", "temporary or backup artifact", "exclude", "temporary material"
    if SENSITIVE_NAME.search(name) and "dataset_signature" not in name:
        return "private_material", "sensitive-name candidate", "exclude", "manual privacy review"
    if lower.startswith("outputs/02_prepared_data/"):
        return "input_data", "prepared benchmark row data or metadata", "exclude", "third-party data; row-level content"
    if suffix in BINARY_MODEL_SUFFIXES:
        return "model_artifact", "serialized model or preprocessor", "exclude", "binary provenance and unsafe deserialization"
    if "score" in name or "prediction" in name:
        return "generated_output", "row-level score or prediction output", "exclude", "row-level data"
    if lower.startswith("outputs/"):
        if any(token in lower for token in ("/summary/", "_summary", "paper_pack", "support_threshold_sensitivity", "/comparison/")):
            return "generated_output", "aggregate experiment summary", "summarize", "verify paths and scientific surface"
        if suffix == ".png":
            return "generated_output", "generated figure", "summarize", "verify provenance and publication rights"
        return "generated_output", "run-level or intermediate experiment artifact", "exclude", "bulk generated output"
    if lower.startswith("src/") or lower.startswith("scripts/") or suffix == ".py":
        return "source", "Python source or CLI", "include", "native dependency review"
    if lower.startswith("config/") or suffix in {".yml", ".yaml", ".lock", ".toml", ".ini", ".cfg"}:
        return "configuration", "experiment or project configuration", "include", "path and environment review"
    if suffix in {".md", ".rst", ".txt", ".pdf", ".docx"}:
        return "documentation", "documentation or manuscript-adjacent material", "include", "rights and staleness review"
    if suffix in ARCHIVE_SUFFIXES:
        return "archive", "compressed or archived material", "exclude", "content and licensing not inspectable by text scan"
    return "other", "unclassified project material", "review", "manual classification required"


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file() or path.is_symlink():
            yield path


def inventory(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source is not a directory: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    total_bytes = 0
    large = {"over_10_mb": 0, "over_50_mb": 0, "over_100_mb": 0}

    fields = [
        "relative_path", "file_type", "extension", "size_bytes", "sha256",
        "likely_purpose", "category", "disposition", "concerns", "is_symlink",
        "is_hidden", "is_readonly",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in iter_files(source):
            relative = path.relative_to(source).as_posix()
            stat = path.lstat()
            size = stat.st_size
            suffix = path.suffix.lower()
            category, purpose, disposition, concern = category_for(relative, suffix)
            writer.writerow(
                {
                    "relative_path": relative,
                    "file_type": suffix.lstrip(".") or "no_extension",
                    "extension": suffix,
                    "size_bytes": size,
                    "sha256": "SYMLINK" if path.is_symlink() else sha256_file(path),
                    "likely_purpose": purpose,
                    "category": category,
                    "disposition": disposition,
                    "concerns": concern,
                    "is_symlink": path.is_symlink(),
                    "is_hidden": path.name.startswith("."),
                    "is_readonly": not os.access(path, os.W_OK),
                }
            )
            counts[category] += 1
            dispositions[disposition] += 1
            total_bytes += size
            if size > 10 * 1024 * 1024:
                large["over_10_mb"] += 1
            if size > 50 * 1024 * 1024:
                large["over_50_mb"] += 1
            if size > 100 * 1024 * 1024:
                large["over_100_mb"] += 1

    return {
        "files": sum(counts.values()),
        "bytes": total_bytes,
        "categories": dict(sorted(counts.items())),
        "dispositions": dict(sorted(dispositions.items())),
        "large_files": large,
        "inventory": output.name,
    }


def scan(root: Path, output: Path | None = None) -> dict[str, object]:
    root = root.resolve()
    findings: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    skipped_binary = 0
    for path in iter_files(root):
        relative = path.relative_to(root).as_posix()
        if any(part in {".git", ".venv", ".release-audit", "__pycache__", ".pytest_cache"} for part in path.relative_to(root).parts):
            continue
        if SENSITIVE_NAME.search(path.name) and "dataset_signature" not in path.name.lower():
            findings["sensitive_filename"].append(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 25 * 1024 * 1024:
            skipped_binary += 1
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            findings["unreadable_text_file"].append(relative)
            continue
        scanned += 1
        for category, pattern in FINDING_PATTERNS.items():
            if category == "absolute_local_path" and relative in PATTERN_DEFINITION_FILES:
                continue
            if pattern.search(content):
                findings[category].append(relative)

    payload = {
        "root_name": root.name,
        "text_files_scanned": scanned,
        "binary_or_large_files_skipped": skipped_binary,
        "finding_counts": {key: len(set(values)) for key, values in sorted(findings.items())},
        "findings": {key: sorted(set(values)) for key, values in sorted(findings.items())},
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="write a SHA-256 source inventory CSV")
    inv.add_argument("source", type=Path)
    inv.add_argument("output", type=Path)
    check = sub.add_parser("scan", help="scan a candidate public tree")
    check.add_argument("root", type=Path)
    check.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inventory":
        result = inventory(args.source, args.output)
    else:
        result = scan(args.root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
