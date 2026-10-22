#!/usr/bin/env python3
"""Safely extract only native IoT-23 v1 capture READMEs and Zeek flow logs.

Streams the full source .tar.gz sequentially without extracting packet captures,
malware binaries, analysis artifacts or symlinks. Existing extracted files are
never overwritten. The archive remains unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile

SCENARIO = re.compile(r"^CTU-(IoT-Malware|Honeypot)-Capture-[A-Za-z0-9-]+$")
ALLOWED_FILE = {"README.md", "conn.log.labeled", "conn.log.labeled.gz"}


def eligible(name: str) -> tuple[str, list[str]] | None:
    if name.startswith("/") or "\\" in name:
        raise ValueError("Archive member has unsafe absolute or backslash path")
    parts = PurePosixPath(name).parts
    if ".." in parts:
        raise ValueError("Archive member includes parent traversal")
    scenario_pos = [i for i, token in enumerate(parts) if SCENARIO.fullmatch(token)]
    if len(scenario_pos) != 1:
        return None
    i = scenario_pos[0]
    if len(parts) == i + 2 and parts[-1] == "README.md":
        return parts[i], list(parts)
    if (len(parts) == i + 3 and parts[-2] == "bro"
            and parts[-1] in {"conn.log.labeled", "conn.log.labeled.gz"}):
        return parts[i], list(parts)
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-scenarios", type=int, default=23)
    args = parser.parse_args(argv)
    if args.expected_scenarios < 0:
        parser.error("Expected scenario count must be nonnegative")
    try:
        if not args.archive.is_file() or not tarfile.is_tarfile(args.archive):
            raise ValueError("Missing or invalid source .tar.gz archive")
        root = args.output_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        entries = []
        chosen = set()
        kinds = set()
        print("Streaming source tarball: only scenario README/conn.log.labeled will be extracted", flush=True)
        with tarfile.open(args.archive, "r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                match = eligible(member.name)
                if match is None:
                    continue
                scenario, parts = match
                relative = Path(*parts)
                target = root / relative
                if not target.resolve().is_relative_to(root):
                    raise ValueError("Selected extraction path escapes root")
                if target.exists():
                    raise ValueError("Selected destination already exists; refusing overwrite: " + scenario)
                key = (scenario, parts[-1])
                if key in chosen:
                    raise ValueError("Duplicate native source entry: " + scenario)
                chosen.add(key)
                target.parent.mkdir(parents=True, exist_ok=True)
                opened = archive.extractfile(member)
                if opened is None:
                    raise ValueError("Selected file missing data stream: " + scenario)
                with opened as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
                entries.append({
                    "capture_id": scenario,
                    "relative_path": relative.as_posix(),
                    "file_size_bytes": member.size,
                    "content_role": "native_capture_readme" if parts[-1] == "README.md"
                    else "native_zeek_labeled_flows",
                })
                kinds.add(scenario)
                print(f"  Extracted {scenario}: {parts[-1]} ({member.size:,} bytes)", flush=True)
        capture_log_counts = {}
        for row in entries:
            if row["content_role"] == "native_zeek_labeled_flows":
                capture_log_counts[row["capture_id"]] = (
                    capture_log_counts.get(row["capture_id"], 0) + 1
                )
        problems = []
        if args.expected_scenarios and len(kinds) != args.expected_scenarios:
            problems.append("Scenario count differs from expected original v1 capture count")
        if any(n != 1 for n in capture_log_counts.values()) or len(capture_log_counts) != len(kinds):
            problems.append("Missing or duplicate labeled-flow source within a capture")
        if sum(row["content_role"] == "native_capture_readme" for row in entries) != len(kinds):
            problems.append("At least one capture README absent")
        report = {
            "schema_version": "iot23-safe-flow-only-extraction-v1",
            "source_archive_filename": args.archive.name,
            "extracted_scenarios": len(kinds),
            "extracted_files": len(entries),
            "files": sorted(entries, key=lambda a: a["relative_path"]),
            "problems": problems,
            "warnings": [
                "No original PCAP or malware binary was extracted.",
                "This extracts only native inputs; capture-independent Protocol B support remains unverified.",
                "No model, train/test split, transformation, label mapping or feature blacklist was constructed.",
                "Exact publisher version and archive integrity should be recorded separately from official checksum.",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.out.with_name(args.out.name + ".partial")
        temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temp.replace(args.out)
        if problems:
            print("[PARTIAL] Source extraction finished with inventory warnings; check JSON", file=sys.stderr)
            return 2
        print(f"[PASS] Extracted {len(entries)} files across {len(kinds)} scenarios -> {args.out}")
        return 0
    except (OSError, ValueError, EOFError, tarfile.TarError) as error:
        print(f"[FAIL] IoT-23 selective extraction: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
