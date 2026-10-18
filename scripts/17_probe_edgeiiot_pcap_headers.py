#!/usr/bin/env python3
"""Read-only Edge-IIoTset original PCAP header and bounded packet-time probe.

Uses Python standard library; does not read or decode packet payloads.
This is a capture-provenance check, not a CSV-to-packet join or an
independent attack-session grouping method.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import sys

MAGICS = {
    bytes.fromhex('d4c3b2a1'): ('<', 1_000_000, 'pcap_microsecond'),
    bytes.fromhex('a1b2c3d4'): ('>', 1_000_000, 'pcap_microsecond'),
    bytes.fromhex('4d3cb2a1'): ('<', 1_000_000_000, 'pcap_nanosecond'),
    bytes.fromhex('a1b23c4d'): ('>', 1_000_000_000, 'pcap_nanosecond'),
}
PCAPNG_MAGIC = bytes.fromhex('0a0d0d0a')

# Exact author-distribution filenames observed in the local source manifest.
# A name alias is NOT evidence of packet-to-CSV row mapping or independent runs.
ATTACK_PCAP_TO_CSV_STEM = {
    'ddos http flood attacks': 'ddos_http_flood_attack',
    'ddos icmp flood attacks': 'ddos_icmp_flood_attack',
    'ddos tcp syn flood attacks': 'ddos_tcp_syn_flood_attack',
    'ddos udp flood attacks': 'ddos_udp_flood_attack',
    'mitm (arp spoofing + dns) attack': 'mitm_attack',
    'os fingerprinting attack': 'os_fingerprinting_attack',
    'password attacks': 'password_attack',
    'port scanning attack': 'port_scanning_attack',
    'ransomware attack': 'ransomware_attack',
    'sql injection attack': 'sql_injection_attack',
    'uploading attack': 'uploading_attack',
    'vulnerability scanner attack': 'vulnerability_scanner_attack',
    'xss attacks': 'xss_attack',
}


def originals(root: Path, extension: str) -> dict[tuple[str, str], Path]:
    if not root.is_dir():
        raise ValueError('Source root not found')
    found: dict[tuple[str, str], Path] = {}
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.casefold() != extension:
            continue
        if not p.resolve().is_relative_to(root.resolve()):
            raise ValueError('Source path escapes root')
        parts = [part.casefold() for part in p.relative_to(root).parts]
        if 'attack traffic' in parts:
            kind = 'attack'
        elif 'normal traffic' in parts:
            kind = 'normal'
        else:
            continue
        stem = p.stem.casefold()
        if extension == '.pcap' and kind == 'attack':
            stem = ATTACK_PCAP_TO_CSV_STEM.get(stem, stem)
        key = (kind, stem)
        if key in found:
            raise ValueError(f'Duplicate original file stem in {kind}: {p.stem}')
        found[key] = p
    return found


def utc_time(value: float) -> str | None:
    try:
        if 946684800 <= value < 2082758400:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat(timespec='microseconds')
    except (OverflowError, ValueError, OSError):
        pass
    return None


def probe_pcap(path: Path, limit: int) -> dict:
    size = path.stat().st_size
    result = {'bytes': size, 'format': None, 'status': None,
              'sampled_packet_headers': 0, 'sample_bad_fraction': 0,
              'sample_backward_time_transitions': 0}
    with path.open('rb') as handle:
        global_header = handle.read(24)
        if len(global_header) < 4:
            result.update(status='truncated_magic', format='unknown')
            return result
        magic = global_header[:4]
        if magic == PCAPNG_MAGIC:
            result.update(status='pcapng_not_parsed', format='pcapng')
            return result
        if magic not in MAGICS:
            result.update(status='unrecognized_magic', format='unknown')
            return result
        byte_order, resolution, pcap_kind = MAGICS[magic]
        result['format'] = pcap_kind
        result['byte_order'] = 'little' if byte_order == '<' else 'big'
        if len(global_header) != 24:
            result['status'] = 'truncated_global_header'
            return result
        major, minor, tz_offset, sigfigs, snaplen, linktype = struct.unpack(
            byte_order + 'HHiIII', global_header[4:24])
        result.update(version_major=major, version_minor=minor,
                      snaplen=snaplen, linktype=linktype)
        if major != 2 or snaplen == 0:
            result['status'] = 'invalid_global_header'
            return result
        first = last = minimum = maximum = None
        for _ in range(limit):
            packet_header = handle.read(16)
            if not packet_header:
                break
            if len(packet_header) < 16:
                result['status'] = 'truncated_packet_header'
                break
            seconds, fraction, incl_len, orig_len = struct.unpack(
                byte_order + 'IIII', packet_header)
            if incl_len > orig_len or handle.tell() + incl_len > size:
                result['status'] = 'invalid_packet_length'
                break
            result['sampled_packet_headers'] += 1
            if fraction >= resolution:
                result['sample_bad_fraction'] += 1
            value = seconds + fraction / resolution
            first = value if first is None else first
            minimum = value if minimum is None else min(minimum, value)
            maximum = value if maximum is None else max(maximum, value)
            if last is not None and value < last:
                result['sample_backward_time_transitions'] += 1
            last = value
            handle.seek(incl_len, 1)
        if result['status'] is None:
            result['status'] = 'packet_sample_read' if result['sampled_packet_headers'] else 'no_packets'
        result['sample_first_utc'] = utc_time(first) if first is not None else None
        result['sample_min_utc'] = utc_time(minimum) if minimum is not None else None
        result['sample_max_utc'] = utc_time(maximum) if maximum is not None else None
        result['first_packet_plausible_epoch'] = bool(first is not None and utc_time(first) is not None)
        result['sample_span_seconds'] = round(maximum - minimum, 6) if first is not None else None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--out', default=Path('outputs/17_revision_campaign/edgeiiot/original_pcap_timestamp_probe.json'), type=Path)
    parser.add_argument('--sample-packets-per-file', type=int, default=64)
    parser.add_argument('--expected-original-files', type=int, default=24)
    args = parser.parse_args(argv)
    if args.sample_packets_per_file < 1 or args.expected_original_files < 0:
        parser.error('Invalid sample size or expected source count')
    try:
        pcaps = originals(args.input_root, '.pcap')
        csvs = originals(args.input_root, '.csv')
        if args.expected_original_files and (len(pcaps) != args.expected_original_files or len(csvs) != args.expected_original_files):
            raise ValueError('Expected count of original PCAP and CSV files not met')
        if not pcaps:
            raise ValueError('No original PCAP files found')
        if set(pcaps) != set(csvs):
            missing_csv = sorted(set(pcaps) - set(csvs))
            missing_pcap = sorted(set(csvs) - set(pcaps))
            raise ValueError(
                f'Unmatched named source files: PCAP without CSV={missing_csv}; '
                f'CSV without PCAP={missing_pcap}'
            )
        per_file = []
        for i, key in enumerate(sorted(pcaps), 1):
            p = pcaps[key]
            print(f'[{i}/{len(pcaps)}] PCAP timestamp header: {p.name}', flush=True)
            observation = probe_pcap(p, args.sample_packets_per_file)
            csv_path = csvs[key]
            alias_used = (key[0] == 'attack' and p.stem.casefold() in ATTACK_PCAP_TO_CSV_STEM)
            per_file.append({'original_file': p.relative_to(args.input_root).as_posix(),
                             'paired_csv_filename': csv_path.relative_to(args.input_root).as_posix(),
                             'pairing_basis': 'documented_filename_alias' if alias_used else 'same_stem',
                             'source_kind': key[0], 'matching_csv_present': True, **observation})
        stats = Counter(row['status'] for row in per_file)
        payload = {
            'schema_version': 'edgeiiot-original-pcap-timestamp-probe-v1',
            'scope': 'pcap_global_header_and_first_packet_timestamps_only_no_capture_join_or_session_approval',
            'original_pcap_files': len(pcaps), 'corresponding_original_csv_files': len(csvs),
            'sample_packet_limit_per_file': args.sample_packets_per_file,
            'status_counts': dict(sorted(stats.items())),
            'first_packet_plausible_epoch_files': sum(row.get('first_packet_plausible_epoch', False) for row in per_file),
            'files': per_file,
            'warnings': [
                'Only initial packet headers are inspected. Full packet time ranges, ordering and capture repetitions are NOT established.',
                'Packet payloads, network identifiers and raw packet data are never read or exported.',
                'PCAP timestamp epoch UTC is interpreted for presentation; source time-zone / host clock accuracy is unverified.',
                'CSV/PCAP filenames are paired by exact stems or source-inventory documented attack-name aliases; neither proves record-level correspondence or identical labels.',
                'PCAPNG is recognized but timestamp blocks are intentionally not parsed by this bounded script.',
                'No events, independent sessions, label verification, leakage-free feature set, model or split are constructed.',
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_name(args.out.name + '.partial')
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        tmp.replace(args.out)
        print(f'[PASS] Examined {len(pcaps)} PCAP source headers -> {args.out}')
        return 0
    except (OSError, ValueError, struct.error) as e:
        print(f'[FAIL] PCAP timestamp probe: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
