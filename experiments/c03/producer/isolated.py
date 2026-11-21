"""Resumable isolated component runner, restricted to tiny synthetic plans.

--resume RUN continues verified preparation/components; --fork RUN --run-id NEW
copies verified completed work into a fresh attempt, preserving interruptions.
No scientific fitting or implicit retries are permitted.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .plan import ROOT, OUTPUT_ROOT, canonical, digest, require
from .checkpoints import publish
from .recovery import open_store, fork_verified


def synthetic_gate(store):
    p = store.plan
    require(p.get('status') == 'SYNTHETIC_VALIDATION_PLAN' and p.get('evidence_scope') == 'SYNTHETIC_ONLY'
            and p.get('execution_authorized') is False, 'Only synthetic component execution is authorized')
    require(all(0 < p['sampling'][s+'_cap'] <= 10000 for s in ('train', 'val')), 'Synthetic sample cap exceeded')
    for node in p['graph']['components'].values():
        params = node['parameters']
        require(1 <= params['n_estimators'] <= 8 and params.get('n_jobs') == 1, 'Synthetic model/worker cap exceeded')
        if node['model_family'] == 'xgb':
            require(params.get('device') == 'cpu', 'Synthetic execution requires CPU')


def component_worker(run, name, attempt):
    from . import validation as v
    store = open_store(run)
    synthetic_gate(store)
    require(attempt.resolve().is_relative_to((store.path/'attempts').resolve()), 'Invalid attempt directory')
    with store.component_lock(name), (attempt/'stages.jsonl').open('x', encoding='utf-8') as log:
        def stage(label):
            log.write(json.dumps({'stage': name+'_'+label})+'\n')
            log.flush()
        stage('loading')
        data = v.load_component_prepared(store, name)
        stage('fitting_and_saving')
        v._fit_component(store, name, data)
        stage('verified')
        v.component_record(store, name)


def execute_component(store, name, started):
    from experiments.c03.engineering import preparation_probe as guard
    from .validation import component_record
    state = guard.resources(store.path)
    require(guard.start_reason(state) is None, guard.start_reason(state) or '')
    require(time.monotonic()-started < guard.LIMITS['seconds'], 'Pipeline time budget reached')
    attempt = store.path/'attempts'/('a'+uuid.uuid4().hex)
    attempt.mkdir(parents=True)
    publish(attempt/'request.json', canonical({'component': name, 'scope': 'SYNTHETIC_ONLY',
                                             'plan_sha256': digest(store.plan), 'limits': guard.LIMITS}))
    progress = guard.Progress(attempt)
    process = tree = None
    reason = None
    peak, minimum = 0, state['available']
    try:
        with (attempt/'worker.log').open('x', encoding='utf-8') as log, \
             (attempt/'monitor.jsonl').open('x', encoding='utf-8') as trace:
            process = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'experiments.c03.producer.isolated',
                '--_worker', str(store.path), '--component', name, '--attempt', str(attempt)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT)
            tree = guard.ProcessTree(process.pid)
            while True:
                state = guard.resources(store.path)
                rss, size = tree.rss(), guard.artifact_size(store.path)
                elapsed = time.monotonic()-started
                peak, minimum = max(peak, rss), min(minimum, state['available'])
                trace.write(json.dumps({'seconds': elapsed, 'rss': rss, 'artifact_bytes': size, **state})+'\n')
                trace.flush()
                progress.update(elapsed, rss, state, size)
                reason = guard.stop_reason(state, rss, size, elapsed)
                if reason:
                    break
                if process.poll() is not None:
                    require(not tree.live(), 'Component launcher exited with live descendants')
                    require(process.returncode == 0, f'Component worker failed: exit {process.returncode}; see {attempt.name}/worker.log')
                    component_record(store, name)
                    break
                time.sleep(guard.LIMITS['interval'])
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        if tree is not None:
            tree.stop()
        elif process is not None and process.poll() is None:
            guard.ProcessTree(process.pid).stop()
        if process is not None:
            process.wait(timeout=10)
    summary = {'component': name, 'scope': 'SYNTHETIC_ONLY', 'status': 'STOPPED' if reason else 'COMPLETE',
        'reason': reason, 'peak_tree_rss': peak, 'minimum_available': minimum, 'wall_seconds': time.monotonic()-started,
        'worker_exit': None if process is None else process.returncode}
    publish(attempt/'result.json', canonical(summary))
    progress.finish(summary)
    require(reason is None, reason or '')


def resume(store):
    from . import validation as v, sealing
    synthetic_gate(store)
    sealing.current_code(store.plan)
    with store.pipeline_lock():
        v.verify_prepared(store)
        started = time.monotonic()
        for name in store.plan['graph']['fit_order']:
            with store.component_lock(name):
                if store.component_path(name).exists():
                    require((store.component_path(name)/'complete.json').exists(),
                            'Interrupted component: use --fork to preserve completed work and restart this component')
                    v.component_record(store, name)
                    print(f'Checkpoint reused: {name}', file=sys.stderr, flush=True)
                    continue
            execute_component(store, name, started)
        if (store.path/'selection.json').exists():
            return sealing.finish_seal(store)
        return sealing.select_and_seal(store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--resume', type=Path)
    mode.add_argument('--fork', type=Path)
    mode.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--run-id')
    parser.add_argument('--component', help=argparse.SUPPRESS)
    parser.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        component_worker(args._worker, args.component, args.attempt)
        return
    store = open_store(args.fork or args.resume)
    synthetic_gate(store)
    if args.fork:
        require(args.run_id is not None, '--fork requires a new --run-id')
        store = fork_verified(store, OUTPUT_ROOT, args.run_id)
        print(f'Recovery fork created: {store.path}', file=sys.stderr, flush=True)
    result = resume(store)
    print(json.dumps({'status': 'SEALED_SYNTHETIC_ONLY', 'run_directory': str(store.path),
                      'scope': result['scope']}))


if __name__ == '__main__':
    main()
