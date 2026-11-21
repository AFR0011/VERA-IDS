"""Approved successor DOC recovery with a new clock and the original total epoch cap."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

from . import doc_pilot_io as rt, doc_pilot_test as evaluation, doc_readiness as readiness
from . import efc_fidelity as reference
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer.plan import output_location
from experiments.c03.engineering import preparation_probe as host, resource_policy as policy
from experiments.c03.engineering.cuda_readiness_probe import query_gpu
from experiments.c03.engineering.lifecycle_probe import Budget

LIMITS = {**policy.LIMITS, 'seconds': 8100}


def definition():
    from . import doc_pilot as original, doc_recovery_contract as recovery
    plan = original.definition()
    plan['recovery'] = recovery.binding()
    plan['schedule'] = {**plan['schedule'], 'fresh_initialization': False, 'total_epoch_cap': 100}
    for name in ('experiments/literature/doc_recovery.py', 'experiments/literature/doc_recovery_fit.py',
                 'experiments/literature/doc_recovery_contract.py', 'tests/test_doc_recovery_fit.py'):
        plan['sources'][name] = rt.file_hash(rt.ROOT / name)
    return plan


def verify_request(run):
    plan = rt.request(run)
    rt.require(all(plan[k] == v for k, v in definition().items()), 'Pilot definition drift')
    return plan


def verify_fit(run):
    # Verify all immutable checkpoint payloads without deserializing them here.
    from .doc_runtime import verify_journal
    fit = rt.read(run / 'fit-complete.json')
    verify_journal(run, fit['journal_commit'])
    rt.require(1 <= fit['epochs'] <= 100 and len(fit['history']) == fit['epochs']
               and fit['updates'] == 9923 * fit['epochs']
               and fit['best_epoch'] == fit['history'].index(min(fit['history'])) + 1
               and fit['best_validation_bce'] == min(fit['history']), 'Fit selection drift')
    rt.require(fit['stop_reason'] in ('patience', 'epoch_cap')
               and (fit['epochs'] == 100 if fit['stop_reason'] == 'epoch_cap'
                    else fit['epochs'] - fit['best_epoch'] == 5), 'Incomplete scientific schedule')
    inventory = rt.read(run / 'fit-inventory.json')
    rt.require(inventory['request_sha256'] == rt.file_hash(run / 'request.json'), 'Fit inventory binding drift')
    for name, sha in inventory['files'].items():
        rt.require(rt.file_hash(rt.safe_artifact(run, name)) == sha, 'Fit inventory file drift')
    return fit


def verify_completed(run):
    plan = verify_request(run); rt.preparation(plan); fit = verify_fit(run)
    complete = rt.read(run / 'complete.json'); result = evaluation.reconcile(run, plan)
    rt.require(complete['status'] == 'COMPLETE' and complete['request_sha256'] == rt.file_hash(run / 'request.json')
               and complete['evaluation_sha256'] == rt.file_hash(run / 'evaluation.json')
               and rt.read(run / 'evaluation.json') == {'request_sha256': rt.file_hash(run / 'request.json'), **result}
               and complete['completed_epoch'] < min(plan['deadline'], fit['completed_epoch'] + 900), 'Pilot completion drift')
    return {**result, 'fit': fit, 'resources': complete}


def run_stage(run, plan, stage, budget, hardware):
    attempt = 'a' + uuid.uuid4().hex; directory = run / 'attempts' / attempt; directory.mkdir(parents=True)
    rt.publish(directory / 'owner.json', rt.canonical({'pid': os.getpid(), 'created': psutil.Process().create_time()}))
    command = [str(rt.PYTHON) if stage != 'export' else sys.executable, '-B', '-u', '-m',
               'experiments.literature.doc_recovery_fit' if stage == 'fit' else 'experiments.literature.doc_pilot_test',
               '--run', str(run), '--attempt', attempt]
    if stage != 'fit': command += ['--stage', stage]
    env = {**os.environ, 'PYTHONPATH': str(rt.ROOT), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    process = tree = None; reason = None; peak = peak_gpu = size_peak = 0; minimum = None; last = -30
    try:
        with (directory / 'worker.log').open('x') as log:
            process = subprocess.Popen(command, cwd=rt.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            tree = host.ProcessTree(process.pid)
            while True:
                elapsed = budget.elapsed(); state = host.resources(run); rss = tree.rss(); size = host.artifact_size(run)
                peak = max(peak, rss); size_peak = max(size_peak, size)
                minimum = min(minimum or state['available'], state['available'])
                gpu = query_gpu(); peak_gpu = max(peak_gpu, gpu['used'])
                rt.require({k: gpu[k] for k in hardware} == hardware, 'GPU identity changed')
                reason = policy.stop_reason(state, gpu, rss, size, elapsed, limits=LIMITS)
                if time.time() >= rt.deadline(run, plan) or plan['started'] + elapsed >= rt.deadline(run, plan):
                    reason = 'Original phase deadline exhausted'
                if elapsed - last >= 30 or process.poll() is not None or reason:
                    progress = {'stage': stage}
                    try: progress = json.loads((directory / 'progress.jsonl').read_text().splitlines()[-1])
                    except (OSError, IndexError, ValueError): pass
                    print(f'[{elapsed / 60:.1f}/135 min] {progress} | worker {rss / 2**30:.2f} GiB | GPU {gpu["used"] / 2**30:.2f} GiB', flush=True)
                    last = elapsed
                if reason or process.poll() is not None: break
                time.sleep(.3)
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        if tree is not None: tree.stop()
        if process is not None:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: reason = reason or 'Worker termination timed out'
    result = {'stage': stage, 'attempt': attempt, 'reason': reason, 'worker_exit': None if process is None else process.returncode,
              'peak_tree_rss': peak, 'peak_gpu_used': peak_gpu, 'minimum_available': minimum,
              'peak_artifacts': size_peak, 'elapsed_seconds': budget.elapsed()}
    rt.publish(directory / 'result.json', rt.canonical(result))
    return result


def execute(run, recover=False):
    with file_lock(run / '.supervisor.guard'):
        plan = verify_request(run)
        if (run / 'complete.json').exists(): return verify_completed(run)
        rt.require(recover or not (run / 'attempts').exists(), 'Explicit recovery required')
        with file_lock(run / '.worker.guard'): pass
        budget = Budget({'started_epoch': plan['started']})
        records = []; reason = None
        try:
            rt.guard(run, plan); rt.preparation(plan)
            gpu = policy.query_gpu_start(); hardware = {k: gpu[k] for k in ('index', 'name', 'total', 'driver')}
            rt.require(policy.start_reason(host.resources(run), gpu, limits=LIMITS) is None, 'Disk/GPU start gate failed')
            for stage in ('fit', 'export', 'score'):
                rt.guard(run, plan)
                if stage == 'fit' and (run / 'seal.json').exists():
                    rt.verify_seal(run, plan); verify_fit(run); continue
                record = run_stage(run, plan, stage, budget, hardware); records.append(record)
                if record['reason'] or record['worker_exit'] != 0:
                    reason = record['reason'] or 'Worker failed; inspect worker.log'; break
            if reason is None:
                verify_request(run); verify_fit(run); rt.preparation(plan)
                result = evaluation.reconcile(run, plan); rt.bound_json(run, 'evaluation.json', result)
                rt.guard(run, plan)
                rt.bound_json(run, 'complete.json', {'status': 'COMPLETE', 'completed_epoch': time.time(),
                    'elapsed_seconds': budget.elapsed(), 'attempts': records, 'hardware': hardware,
                    'evaluation_sha256': rt.file_hash(run / 'evaluation.json')})
                return verify_completed(run)
        except BaseException as exc:
            reason = f'{type(exc).__name__}: {exc}'
        result = {'scope': rt.SCOPE, 'status': 'STOPPED', 'reason': reason, 'attempts': records,
                  'elapsed_seconds': budget.elapsed(), 'run_id': run.name}
        folder = run / 'stops'; folder.mkdir(exist_ok=True)
        rt.publish(folder / (uuid.uuid4().hex + '.json'), rt.canonical(result))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run', action='store_true'); modes.add_argument('--recover', type=Path); modes.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify: result = verify_completed(output_location(args.verify))
    elif args.recover: result = execute(output_location(args.recover), True)
    else:
        # Full read-only provenance verification precedes the new bounded clock.
        readiness.verify_completed(rt.PREPARED)
        plan = definition(); run = rt.OUTPUT / ('doc-recovery-' + datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz'))
        run.mkdir(); now = time.time()
        rt.publish(run / 'request.json', rt.canonical({**plan, 'started': now, 'fit_deadline': now + 7200, 'deadline': now + 8100}))
        print('DOC scientific run: ' + str(run), flush=True); result = execute(run)
    print(json.dumps(result, indent=2))
    if result.get('status') != 'COMPLETE': sys.exit(2)


if __name__ == '__main__': main()
