#!/usr/bin/env python3
"""Read-only targeted Edge-IIoTset PCAP structure and shared-header investigation.

Reproduces the first invalid original Vulnerability Scanner record header,
then compares ordered packet timestamp/length metadata in the normal Modbus
and Temperature/Humidity captures on one UTC day. Never reads packet payloads.
Matching metadata is not proof of identical payloads or independent runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys

MAGIC = {
    bytes.fromhex("d4c3b2a1"): ("<", 1_000_000),
    bytes.fromhex("a1b2c3d4"): (">", 1_000_000),
    bytes.fromhex("4d3cb2a1"): ("<", 1_000_000_000),
    bytes.fromhex("a1b23c4d"): (">", 1_000_000_000),
}
SOURCE_NAMES = {
    "scanner": ("attack", "Vulnerability scanner attack.pcap"),
    "modbus": ("normal", "Modbus.pcap"),
    "temperature": ("normal", "Temperature_and_Humidity.pcap"),
}


def safe_source(root: Path, relative: str) -> Path:
    name = Path(relative)
    if not name.parts or name.is_absolute() or ".." in name.parts or name.suffix.casefold() != ".pcap":
        raise ValueError("Invalid relative PCAP location")
    target = root / name
    if not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("PCAP missing or source path escapes root")
    return target


def select_sources(root: Path, previous: dict) -> dict:
    if previous.get("schema_version") != "edgeiiot-full-pcap-header-timeline-census-v1":
        raise ValueError("Expected actual C9 original PCAP census")
    if previous.get("file_count") != 24 or len(previous.get("files", [])) != 24:
        raise ValueError("Expected full 24-source inventory")
    if previous.get("max_packets_per_file") != 0:
        raise ValueError("Expected uncapped C9 source report")
    selected = {}
    for nickname, (kind, basename) in SOURCE_NAMES.items():
        matches = [item for item in previous["files"]
                   if item.get("source_kind") == kind
                   and Path(item.get("source_file", "")).name == basename]
        if len(matches) != 1:
            raise ValueError(f"Expected one original file for {nickname}")
        selected[nickname] = (safe_source(root, matches[0]["source_file"]), matches[0])
    return selected


def parse_global(stream, basename: str):
    head = stream.read(24)
    if len(head) != 24 or head[:4] not in MAGIC:
        raise ValueError(f"Unsupported classic-PCAP header: {basename}")
    order, resolution = MAGIC[head[:4]]
    major, minor, zone, accuracy, snaplen, linktype = struct.unpack(order + "HHiIII", head[4:])
    if major != 2 or snaplen == 0:
        raise ValueError(f"Invalid classic-PCAP global header: {basename}")
    return order, resolution, snaplen, linktype


def scan_headers(path, on_record):
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        order, resolution, snaplen, linktype = parse_global(stream, path.name)
        record_struct = struct.Struct(order + "IIII")
        count = 0
        while True:
            offset = stream.tell()
            raw = stream.read(16)
            if not raw:
                return {"status": "full_file_scanned", "headers_valid": count,
                        "file_bytes": file_size}
            if len(raw) != 16:
                return {"status": "truncated_record_header", "headers_valid": count,
                        "record_header_offset": offset,
                        "remaining_bytes_from_offset": file_size - offset,
                        "header_bytes_available": len(raw), "file_bytes": file_size}
            sec, frac, captured, original = record_struct.unpack(raw)
            remaining = file_size - stream.tell()
            failures = []
            if captured > original:
                failures.append("captured_length_exceeds_original_length")
            if captured > snaplen:
                failures.append("captured_length_exceeds_snapshot_length")
            if captured > remaining:
                failures.append("captured_length_exceeds_remaining_file_bytes")
            if failures:
                stream.seek(offset)
                next_four = stream.read(4)
                return {
                    "status": "invalid_record_header", "headers_valid": count,
                    "record_header_offset": offset,
                    "remaining_bytes_from_offset": file_size - offset,
                    "remaining_payload_bytes": remaining,
                    "record_caplen": captured, "record_original_len": original,
                    "snapshot_length": snaplen,
                    "record_timestamp_fraction_in_range": frac < resolution,
                    "candidate_new_pcap_global_header": next_four in MAGIC,
                    "failed_constraints": failures,
                    "file_bytes": file_size,
                }
            on_record(sec, frac, captured, original)
            count += 1
            stream.seek(captured, 1)


def day_fingerprint(path: Path, day: str) -> dict:
    year, month, day_num = map(int, day.split("-"))
    lower = int(datetime(year, month, day_num, tzinfo=timezone.utc).timestamp())
    upper = lower + 86400
    h = hashlib.sha256()
    count = 0
    first = last = None

    def accumulate(sec, frac, included, original):
        nonlocal count, first, last
        if lower <= sec < upper:
            # A sequence fingerprint of timestamp/length fields ONLY.
            h.update(struct.pack("!IIII", sec, frac, included, original))
            first = (sec, frac) if first is None else first
            last = (sec, frac)
            count += 1

    status = scan_headers(path, accumulate)
    if status["status"] != "full_file_scanned":
        raise ValueError(f"Fingerprint source scan incomplete: {path.name}")
    return {
        "pcap_filename": path.name, "requested_utc_day": day,
        "packets_on_requested_day": count,
        "ordered_packet_header_fields_sha256": h.hexdigest(),
        "first_packet_epoch_seconds_and_fraction": first,
        "last_packet_epoch_seconds_and_fraction": last,
        "source_scan_status": status["status"],
        "packets_scanned_total": status["headers_valid"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--timeline-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(
        "outputs/17_revision_campaign/edgeiiot/pcap_integrity_followup.json"))
    parser.add_argument("--comparison-day", default="2021-12-20")
    args = parser.parse_args(argv)
    try:
        root = args.input_root
        prior = json.loads(args.timeline_report.read_text(encoding="utf8"))
        sources = select_sources(root, prior)
        scanner, prior_scanner = sources["scanner"]
        print("[1/3] Diagnosing scanner PCAP stop point", flush=True)
        failure = scan_headers(scanner, lambda *_: None)
        if prior_scanner["status"] == "invalid_record_size_or_truncation":
            failure["agrees_with_prior_record_count"] = (
                failure["headers_valid"] == prior_scanner["packets_scanned"])
        print("[2/3] Fingerprinting Modbus Dec 20 packet-header fields", flush=True)
        modbus = day_fingerprint(sources["modbus"][0], args.comparison_day)
        print("[3/3] Fingerprinting Temperature/Humidity Dec 20 packet-header fields", flush=True)
        temperature = day_fingerprint(sources["temperature"][0], args.comparison_day)
        same = (
            modbus["packets_on_requested_day"] > 0
            and modbus["packets_on_requested_day"] == temperature["packets_on_requested_day"]
            and modbus["ordered_packet_header_fields_sha256"] ==
                temperature["ordered_packet_header_fields_sha256"]
        )
        output = {
            "schema_version": "edgeiiot-c9-structural-followup-v1",
            "scope": "targeted_pcap_header_only_no_packet_payload_no_independent_session_identification",
            "comparison_utc_day": args.comparison_day,
            "vulnerability_scanner": {
                "pcap_filename": scanner.name,
                "previous_status": prior_scanner["status"],
                "previous_headers_scanned": prior_scanner["packets_scanned"],
                **failure,
            },
            "normal_source_day_comparison": {
                "modbus": modbus, "temperature_and_humidity": temperature,
                "ordered_packet_header_sequences_identical_for_day": same,
            },
            "warnings": [
                "This scan characterizes structural metadata but neither repairs nor decodes the source.",
                "Identical packet-header sequence hashes do NOT demonstrate identical packet payloads.",
                "Any shared metadata is source reuse evidence, not authenticated source lineage or independent operations.",
                "Packet dates/gaps are not attack-run IDs or approved train/test partition groups.",
                "No packet payloads, IP addresses, flows, source rows or raw records exported.",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(args.out.name + ".partial")
        temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
        temporary.replace(args.out)
        print(f"[PASS] Wrote targeted PCAP metadata -> {args.out}", flush=True)
        return 0
    except (OSError, ValueError, TypeError, KeyError, struct.error, OverflowError) as exc:
        print(f"[FAIL] Targeted PCAP integrity probe: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
