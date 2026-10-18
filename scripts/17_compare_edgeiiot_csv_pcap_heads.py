#!/usr/bin/env python3
"""Read-only bounded ordinal comparison of Edge-IIoTset CSV and PCAP head times.

Uses an existing full 24-file PCAP header report ONLY for audited name pairs.
Compares first N original CSV rows to first N packet header timestamps by
ordinal position without asserting packet-row identity or time-zone policy.
No raw packet bytes, addresses, rows or literal timestamps are exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import struct
import sys

TIME_FRAGMENT = re.compile(
    r"^\s*(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\.(\d{1,9})\s*$"
)
MAGICS = {
    bytes.fromhex('d4c3b2a1'): ('<', 1_000_000),
    bytes.fromhex('a1b2c3d4'): ('>', 1_000_000),
    bytes.fromhex('4d3cb2a1'): ('<', 1_000_000_000),
    bytes.fromhex('a1b23c4d'): ('>', 1_000_000_000),
}
OFFSET_HOURS = tuple(range(-12, 15))


def source_file(root: Path, relative: str, extension: str) -> Path:
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or p.suffix.lower() != extension:
        raise ValueError('Unsafe or unexpected original source relative path')
    target = root / p
    if not target.is_file() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Original source pair missing or escaped root: {p.name}')
    return target


def csv_head(path: Path, n: int) -> list[tuple[int, int, int, int, int] | None]:
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'frame.time' not in reader.fieldnames:
            raise ValueError(f'CSV missing frame.time: {path.name}')
        for record in reader:
            if len(rows) >= n:
                break
            value = (record.get('frame.time') or '').strip()
            m = TIME_FRAGMENT.fullmatch(value)
            if not m:
                rows.append(None)
                continue
            year, hh, mm, ss = map(int, m.groups()[:4])
            nanos = int(m[5].ljust(9, '0'))
            if nanos % 1000 or not (2000 <= year <= 2100 and hh < 24 and mm < 60 and ss < 60):
                rows.append(None)
            else:
                rows.append((year, hh, mm, ss, nanos // 1000))
    return rows


def pcap_head(path: Path, n: int) -> list[datetime]:
    times = []
    size = path.stat().st_size
    with path.open('rb') as handle:
        head = handle.read(24)
        if len(head) != 24 or head[:4] not in MAGICS:
            raise ValueError(f'Unsupported or incomplete classic PCAP: {path.name}')
        order, resolution = MAGICS[head[:4]]
        major, minor, tz, sigfigs, snaplen, linktype = struct.unpack(order + 'HHiIII', head[4:24])
        if major != 2 or snaplen == 0:
            raise ValueError(f'Bad PCAP global header: {path.name}')
        for _ in range(n):
            packet = handle.read(16)
            if not packet:
                break
            if len(packet) != 16:
                raise ValueError(f'Truncated PCAP record header: {path.name}')
            sec, frac, included, original = struct.unpack(order + 'IIII', packet)
            if frac >= resolution or included > original or handle.tell() + included > size:
                raise ValueError(f'Invalid PCAP record header: {path.name}')
            if resolution == 1_000_000_000 and frac % 1000:
                raise ValueError(f'Sub-microsecond PCAP precision cannot be compared exactly: {path.name}')
            micros = frac if resolution == 1_000_000 else frac // 1000
            times.append(datetime.fromtimestamp(sec, tz=timezone.utc).replace(microsecond=micros))
            handle.seek(included, 1)
    return times


def compare(csv_times, pcap_times) -> dict:
    checked = min(len(csv_times), len(pcap_times))
    candidate = {str(h): 0 for h in OFFSET_HOURS}
    parsed = 0
    for index in range(checked):
        stamp = csv_times[index]
        if stamp is None:
            continue
        parsed += 1
        y, hh, mm, ss, us = stamp
        for h in OFFSET_HOURS:
            dt = pcap_times[index] + timedelta(hours=h)
            if (dt.year, dt.hour, dt.minute, dt.second, dt.microsecond) == (y, hh, mm, ss, us):
                candidate[str(h)] += 1
    observed = {k: v for k, v in candidate.items() if v}
    top = max(observed.values(), default=0)
    return {
        'csv_head_rows': len(csv_times),
        'pcap_head_packet_headers': len(pcap_times),
        'ordinal_positions_compared': checked,
        'csv_year_time_fragments_parseable_for_microsecond_comparison': parsed,
        'exact_ordinal_matches_by_candidate_hour_offset': observed,
        'max_matched_ordinals': top,
        'best_offset_candidates_hours': [int(k) for k, v in observed.items() if v == top] if top else [],
        'nonzero_match_presence_only': bool(top),
    }


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--pcap-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs/17_revision_campaign/edgeiiot/csv_pcap_head_correspondence.json'))
    parser.add_argument('--limit-per-pair', type=int, default=64)
    parser.add_argument('--expected-pairs', type=int, default=24)
    args = parser.parse_args(argv)
    if args.limit_per_pair < 1 or args.expected_pairs < 0:
        parser.error('Invalid sampling limit or expected pair count')
    try:
        root = args.input_root
        if not root.is_dir():
            raise ValueError('Source root does not exist')
        report = json.loads(args.pcap_report.read_text(encoding='utf-8'))
        if report.get('schema_version') != 'edgeiiot-original-pcap-timestamp-probe-v1':
            raise ValueError('Expected audited PCAP filename report version')
        pairs = report.get('files', [])
        if not isinstance(pairs, list) or not pairs:
            raise ValueError('Empty or invalid PCAP pairing report')
        if args.expected_pairs and len(pairs) != args.expected_pairs:
            raise ValueError('Unexpected PCAP pairing report file count')
        seen_csv, seen_pcap = set(), set()
        records = []
        for i, item in enumerate(pairs, 1):
            rel_csv = item['paired_csv_filename']
            rel_pcap = item['original_file']
            if rel_csv in seen_csv or rel_pcap in seen_pcap:
                raise ValueError('Duplicate source pairing key')
            seen_csv.add(rel_csv)
            seen_pcap.add(rel_pcap)
            csv_file = source_file(root, rel_csv, '.csv')
            pcap_file = source_file(root, rel_pcap, '.pcap')
            print(f'[{i}/{len(pairs)}] Comparing head positions: {csv_file.name}', flush=True)
            result = compare(csv_head(csv_file, args.limit_per_pair), pcap_head(pcap_file, args.limit_per_pair))
            records.append({
                'csv_file': rel_csv,
                'pcap_file': rel_pcap,
                'pairing_basis': item['pairing_basis'],
                **result,
            })
        summary = {
            'schema_version': 'edgeiiot-csv-pcap-bounded-ordinal-time-correspondence-v1',
            'scope': 'first_N_ordinal_time_only_no_full_join_no_timezone_or_independent_session_approval',
            'pair_count': len(records),
            'sample_limit_per_pair': args.limit_per_pair,
            'total_ordinal_positions_compared': sum(x['ordinal_positions_compared'] for x in records),
            'total_parsed_year_time_fragments': sum(x['csv_year_time_fragments_parseable_for_microsecond_comparison'] for x in records),
            'pairs_with_at_least_one_exact_match': sum(bool(x['max_matched_ordinals']) for x in records),
            'files': records,
            'warnings': [
                'Matches rely on CSV/PCAP ordinal position only and are diagnostic, not a verified packet-to-CSV record join.',
                'Hour offsets are candidate numerical relationships, NOT authoritative source time-zone or clock-calibration metadata.',
                'Original CSV year/time values omit day and month. This script NEVER fabricates dates or fills missing values.',
                'A mismatch can result from packet capture filtering, omitted CSV rows or record reordering and does not prove distinct source provenance.',
                'Only a bounded sample of the head is evaluated, not full capture chronology or independent sessions.',
                'No raw timestamp literals, source IPs, packet payloads, flow rows, model metrics or attacker identities are exported.',
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.out.with_name(args.out.name + '.partial')
        temp.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        temp.replace(args.out)
        print(f'[PASS] Evaluated {len(records)} source pairs -> {args.out}')
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error, struct.error, OverflowError) as exc:
        print(f'[FAIL] Bounded CSV/PCAP comparison: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(run())
