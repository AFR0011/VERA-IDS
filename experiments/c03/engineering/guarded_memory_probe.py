"""Windows-aware supervisor for the original, unchanged C03 engineering worker.

Run from the repository root with --inspect or --probe. Fixed authorized scope:
30k/80k training samples, eight trees, one worker, no validation/test observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import psutil

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import c03_rf_bounded_memory_probe as original

GIB = 1024 ** 3
MIN_START = 3 * GIB
MIN_FREE = 1.5 * GIB
MAX_RSS = 3 * GIB
MAX_SECONDS = 900


def tree_rss(pid):
    """Include Windows venv redirector descendants; monitoring errors propagate."""
    root = psutil.Process(pid)
    processes = [root, *root.children(recursive=True)]
    # Summing working sets can double-count shared pages: conservatively bounded.
    return sum(p.memory_info().rss for p in processes)


def stop_tree(pid):
    """Kill and reap descendants as well as the Windows Python redirector."""
    try:
        root = psutil.Process(pid)
        children = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for process in [*reversed(children), root]:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([*children, root], timeout=5)
    if alive:
        raise RuntimeError("Probe descendants did not stop")


def supervise():
    original.precheck_environment()
    initial = original.os_memory()['physical_available']
    print(f"Available physical RAM before start: {initial/GIB:.2f} GiB", flush=True)
    if initial < MIN_START:
        raise RuntimeError("Start blocked: less than 3 GiB physical RAM available")
    command = [sys.executable, '-B', '-u', str(TOOLS/'c03_rf_bounded_memory_probe.py'),
               '--_worker', '--rows', '80000', '--trees', '8']
    peak, minimum, reason = 0, initial, None
    with tempfile.TemporaryFile(mode='w+t', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while process.poll() is None:
                try:
                    available = original.os_memory()['physical_available']
                    rss = tree_rss(process.pid)
                except psutil.NoSuchProcess:
                    if process.poll() is not None:
                        break
                    raise
                peak, minimum = max(peak, rss), min(minimum, available)
                if available < MIN_FREE:
                    reason = 'Available physical RAM below 1.5 GiB'
                elif rss > MAX_RSS:
                    reason = 'Combined worker-tree RSS above 3 GiB'
                elif time.monotonic() - started >= MAX_SECONDS:
                    reason = '15-minute engineering-probe time budget reached'
                if reason:
                    stop_tree(process.pid)
                    break
                time.sleep(.3)
        except BaseException:
            if process.poll() is None:
                stop_tree(process.pid)
            raise
        finally:
            process.wait()
            log.seek(0)
            print(log.read(), end='', flush=True)
            print(f"MONITOR tree_peak_RSS_GiB={peak/GIB:.3f} "
                  f"minimum_available_GiB={minimum/GIB:.3f} "
                  f"wall_s={time.monotonic()-started:.2f}", flush=True)
        if reason:
            print('SAFETY HALT: ' + reason)
            return 3
        if process.returncode:
            print(f'Worker incomplete: exit {process.returncode}; no feasibility conclusion')
            return 2
    print('Bounded engineering probe complete; full-size fit remains unvalidated.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect', action='store_true')
    mode.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        print(original.os_memory())
        return 0
    return supervise()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print(f'BLOCKED: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(2)
