#!/usr/bin/env python3
"""Read-only full classic-PCAP timestamp metadata census for Edge-IIoTset.

Reads all packet *headers* and seeks past payload bytes. UTC timestamp gaps are
observational discontinuities, NEVER independent attack-run IDs. Source capture
clocks, labels, file origin and CSV record identity remain unverified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import heapq
import json
from pathlib import Path
import struct
import sys

MAGIC = {
    bytes.fromhex('d4c3b2a1'): ('<', 1_000_000, 'pcap_microsecond'),
    bytes.fromhex('a1b2c3d4'): ('>', 1_000_000, 'pcap_microsecond'),
    bytes.fromhex('4d3cb2a1'): ('<', 1_000_000_000, 'pcap_nanosecond'),
    bytes.fromhex('a1b23c4d'): ('>', 1_000_000_000, 'pcap_nanosecond'),
}
GAP_SECONDS = (60, 300, 1800, 3600, 21600, 86400)
MIN_EPOCH, MAX_EPOCH = 946684800, 2082758400  # 2000-01-01 to 2036-01-01


def iso_epoch(value: int | None, scale: int) -> str | None:
    if value is None or not (MIN_EPOCH <= value // scale < MAX_EPOCH):
        return None
    seconds, remainder = divmod(value, scale)
    try:
        stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
        # Preserve six significant fractional digits; nanosecond residuals
        # remain represented by the integer time-delta counts if relevant.
        return stamp.replace(microsecond=(remainder * 1_000_000) // scale).isoformat()
    except (ValueError, OverflowError, OSError):
        return None


def safe_pcap(root: Path, relative: str) -> Path:
    name = Path(relative)
    if (name.is_absolute() or '..' in name.parts or name.suffix.casefold() != '.pcap'
            or not name.parts):
        raise ValueError('Invalid relative PCAP path in prior manifest')
    result = root / name
    if not result.is_file() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('PCAP file missing or path escapes source root')
    return result


def full_scan(path: Path, *, max_packets: int = 0, progress_every: int = 2_000_000) -> dict:
    result = {
        'bytes': path.stat().st_size,
        'status': None,
        'packets_scanned': 0,
        'scan_complete': False,
        'bad_timestamp_fractions': 0,
        'implausible_epoch_packets': 0,
        'backward_adjacent_transitions': 0,
        'equal_adjacent_timestamps': 0,
        'forward_gap_counts_at_least_seconds': {str(v): 0 for v in GAP_SECONDS},
        'utc_day_packet_counts': {},
        'utc_hour_packet_counts': {},
    }
    total_size = result['bytes']
    with path.open('rb') as handle:
        head = handle.read(24)
        if len(head) < 4:
            result['status'] = 'truncated_magic'
            return result
        if head[:4] not in MAGIC:
            result['status'] = 'pcapng_not_scanned' if head[:4] == bytes.fromhex('0a0d0d0a') else 'unknown_format'
            return result
        if len(head) != 24:
            result['status'] = 'truncated_global_header'
            return result
        order, scale, fmt = MAGIC[head[:4]]
        major, minor, _zone, _accuracy, snaplen, linktype = struct.unpack(order+'HHiIII', head[4:24])
        result.update(format=fmt, byte_order='little' if order=='<' else 'big',
                      version_major=major, version_minor=minor, snaplen=snaplen, linktype=linktype)
        if major != 2 or snaplen == 0:
            result['status'] = 'invalid_global_header'
            return result
        row_struct = struct.Struct(order+'IIII')
        first = last = minimum = maximum = None
        days: Counter[str] = Counter()
        hours: Counter[str] = Counter()
        gaps = Counter()
        top_gaps: list[tuple[int, int, int]] = []
        last_second = None
        date_key = hour_key = None
        while True:
            if max_packets and result['packets_scanned'] >= max_packets:
                result['status'] = 'packet_limit_reached'
                break
            packet_header = handle.read(16)
            if not packet_header:
                result['status'] = 'full_file_scanned'
                result['scan_complete'] = True
                break
            if len(packet_header) < 16:
                result['status'] = 'truncated_packet_header'
                break
            sec, frac, included, orig = row_struct.unpack(packet_header)
            if included > orig or included > snaplen or handle.tell() + included > total_size:
                result['status'] = 'invalid_record_size_or_truncation'
                break
            result['packets_scanned'] += 1
            if frac >= scale:
                result['bad_timestamp_fractions'] += 1
            if not (MIN_EPOCH <= sec < MAX_EPOCH):
                result['implausible_epoch_packets'] += 1
            else:
                if sec != last_second:
                    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
                    date_key = dt.date().isoformat()
                    hour_key = dt.strftime('%Y-%m-%dT%H:00Z')
                    last_second = sec
                days[date_key] += 1
                hours[hour_key] += 1
            tick = sec * scale + frac
            if first is None:
                first = tick
            minimum = tick if minimum is None else min(minimum, tick)
            maximum = tick if maximum is None else max(maximum, tick)
            if last is not None:
                difference = tick - last
                if difference < 0:
                    result['backward_adjacent_transitions'] += 1
                elif difference == 0:
                    result['equal_adjacent_timestamps'] += 1
                else:
                    for boundary in GAP_SECONDS:
                        if difference >= boundary * scale:
                            gaps[str(boundary)] += 1
                    if len(top_gaps) < 10:
                        heapq.heappush(top_gaps, (difference, last, tick))
                    elif difference > top_gaps[0][0]:
                        heapq.heapreplace(top_gaps, (difference, last, tick))
            last = tick
            handle.seek(included, 1)
            if progress_every and result['packets_scanned'] % progress_every == 0:
                print(f'  {path.name}: {result["packets_scanned"]:,} packet headers', flush=True)
        result['first_packet_utc'] = iso_epoch(first, scale)
        result['last_packet_utc'] = iso_epoch(last, scale)
        result['earliest_timestamp_utc'] = iso_epoch(minimum, scale)
        result['latest_timestamp_utc'] = iso_epoch(maximum, scale)
        result['max_forward_adjacent_gap_seconds'] = (round(max((x[0] for x in top_gaps), default=0) / scale, 6))
        result['ten_largest_forward_adjacent_gaps'] = [
            {'gap_seconds': round(delta / scale, 6),
             'previous_packet_utc': iso_epoch(before, scale),
             'next_packet_utc': iso_epoch(after, scale)}
            for delta, before, after in sorted(top_gaps, reverse=True)
        ]
        result['forward_gap_counts_at_least_seconds'] = {str(n): gaps[str(n)] for n in GAP_SECONDS}
        result['utc_day_packet_counts'] = dict(sorted(days.items()))
        result['utc_hour_packet_counts'] = dict(sorted(hours.items()))
        result['occupied_utc_days'] = len(days)
        result['occupied_utc_hours'] = len(hours)
    return result


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--pcap-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/17_revision_campaign/edgeiiot/full_pcap_timeline_census.json'))
    parser.add_argument('--expected-files', type=int, default=24)
    parser.add_argument('--max-packets-per-file', type=int, default=0,
                        help='0 means scan entire file; positive values are bounded debugging only')
    parser.add_argument('--progress-every', type=int, default=2_000_000)
    a = parser.parse_args(argv)
    if a.expected_files < 0 or a.max_packets_per_file < 0 or a.progress_every < 0:
        parser.error('Nonnegative file, packet and progress settings required')
    try:
        root = a.input_root
        if not root.is_dir():
            raise ValueError('Input root does not exist')
        report = json.loads(a.pcap_report.read_text(encoding='utf-8'))
        if report.get('schema_version') != 'edgeiiot-original-pcap-timestamp-probe-v1':
            raise ValueError('Unexpected PCAP manifest schema version')
        pairs = report.get('files')
        if not isinstance(pairs, list) or not pairs or (a.expected_files and len(pairs) != a.expected_files):
            raise ValueError('Unexpected PCAP manifest file count')
        seen = set()
        paths = []
        for item in pairs:
            key = item['original_file']
            if key in seen:
                raise ValueError('Duplicate PCAP in prior manifest')
            seen.add(key)
            paths.append((item['source_kind'], key, safe_pcap(root, key)))
        if a.expected_files == 24 and Counter(kind for kind, _, _ in paths) != {'attack': 14, 'normal': 10}:
            raise ValueError('Unexpected attack/normal source-group counts')
        outputs = []
        for index, (kind, rel, path) in enumerate(paths, 1):
            print(f'[{index}/{len(paths)}] Whole-PCAP header census: {path.name}', flush=True)
            metrics = full_scan(path, max_packets=a.max_packets_per_file, progress_every=a.progress_every)
            outputs.append({'source_file': rel, 'source_kind': kind, **metrics})
        report = {
            'schema_version': 'edgeiiot-full-pcap-header-timeline-census-v1',
            'scope': 'all_packet_timestamp_headers_only_no_payload_read_no_attack_run_identification',
            'file_count': len(outputs),
            'max_packets_per_file': a.max_packets_per_file,
            'all_files_scanned_to_eof': all(row['scan_complete'] for row in outputs),
            'total_packet_headers_scanned': sum(row['packets_scanned'] for row in outputs),
            'source_group_counts': dict(sorted(Counter(row['source_kind'] for row in outputs).items())),
            'status_counts': dict(sorted(Counter(row['status'] for row in outputs).items())),
            'files': outputs,
            'warnings': [
                'UTC epochs are PCAP capture-header data; capture clock correctness and location/time zone not independently checked.',
                'Observed UTC day/hour occupancy and time gaps are descriptive ONLY, not independent attack session IDs.',
                'Same-filename source category is not verification of individual packet attack label or CSV row correspondence.',
                'No packet payloads, IP addresses, protocols, IDs, feature vectors, CSV records, or model outputs are exported.',
                'If max_packets_per_file is nonzero or a scan errors, report is NOT a whole-source census.',
                'No Protocol B partition, support admissibility, feature policy, or dataset promotion is asserted.',
            ],
        }
        a.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = a.out.with_name(a.out.name + '.partial')
        temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        temporary.replace(a.out)
        print(f'[PASS] Saved metadata for {len(outputs)} PCAPs, {report["total_packet_headers_scanned"]:,} headers -> {a.out}')
        if not report['all_files_scanned_to_eof']:
            print('[WARNING] Some files incomplete or deliberately capped; inspect status_counts.', file=sys.stderr)
        return 0
    except (OSError, ValueError, TypeError, KeyError, struct.error, OverflowError) as e:
        print(f'[FAIL] PCAP whole-timeline audit: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(run())
