#!/usr/bin/env python3
"""Update or verify tracked aggregate-output and release-asset manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ids_eval_framework.src.protocol_a_correction import load_evidence  # noqa: E402


SUMMARY_DIR = ROOT / "outputs" / "summaries"
SUMMARY_MANIFEST = SUMMARY_DIR / "SOURCE_MANIFEST.csv"
CORRECTED = {
    "protocol_a_core_summary.csv",
    "external_protocol_a_summary.csv",
    "protocol_a_flat_vs_two_stage.csv",
    "reference_profile_metric_drop.csv",
    "seed_reliability_summary.csv",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_rows() -> list[dict[str, str]]:
    with SUMMARY_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        existing = {Path(row["public_path"]).name: row for row in csv.DictReader(handle)}
    files = sorted(
        path for path in SUMMARY_DIR.iterdir() if path.is_file() and path.name not in {"README.md", "SOURCE_MANIFEST.csv"}
    )
    rows: list[dict[str, str]] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        source = existing.get(path.name, {}).get("source_relative_path", "")
        if path.name in CORRECTED:
            source = "outputs/evidence/protocol_a_confusion_matrices.jsonl + retained historical summary fields"
        rows.append(
            {
                "public_path": relative,
                "source_relative_path": source,
                "sha256": sha256(path),
                "bytes": str(path.stat().st_size),
            }
        )
    return rows


def render_summary_manifest(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["public_path", "source_relative_path", "sha256", "bytes"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def verify_asset_manifest(asset_path: Path | None = None) -> None:
    path = ROOT / "release" / "ASSET_MANIFEST.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "release_tag", "package_version", "repository", "asset", "authoritative_source"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"Asset manifest missing fields: {sorted(missing)}")
    asset = data["asset"]
    source = data["authoritative_source"]
    asset_required = {"filename", "bytes", "page_count", "sha256", "license", "release_url"}
    source_required = {"filename", "sha256", "published"}
    if asset_required - set(asset):
        raise ValueError(f"Asset entry missing fields: {sorted(asset_required - set(asset))}")
    if source_required - set(source):
        raise ValueError(f"Source entry missing fields: {sorted(source_required - set(source))}")
    if asset["filename"] != "VERA-IDS-Thesis.pdf" or data["release_tag"] != "v2026.08":
        raise ValueError("Asset filename or release tag is incorrect")
    if source["published"] is not False:
        raise ValueError("The source DOCX must be marked unpublished")
    if asset_path is not None:
        from pypdf import PdfReader

        resolved = asset_path.resolve()
        if resolved.name != asset["filename"]:
            raise ValueError("Local release asset filename does not match the manifest")
        if resolved.stat().st_size != int(asset["bytes"]):
            raise ValueError("Local release asset size does not match the manifest")
        if sha256(resolved).upper() != str(asset["sha256"]).upper():
            raise ValueError("Local release asset SHA-256 does not match the manifest")
        if len(PdfReader(resolved).pages) != int(asset["page_count"]):
            raise ValueError("Local release asset page count does not match the manifest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--asset", type=Path, help="Optional local release PDF to verify")
    args = parser.parse_args()
    load_evidence(ROOT / "outputs" / "evidence" / "protocol_a_confusion_matrices.jsonl")
    expected = render_summary_manifest(summary_rows())
    if args.update:
        SUMMARY_MANIFEST.write_text(expected, encoding="utf-8", newline="\n")
    elif SUMMARY_MANIFEST.read_text(encoding="utf-8-sig") != expected:
        raise SystemExit("SOURCE_MANIFEST.csv is stale; run scripts/verify_manifests.py --update")
    verify_asset_manifest(args.asset)
    print("Tracked manifests OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
