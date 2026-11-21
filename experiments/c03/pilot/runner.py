"""Bounded fixed C03 producer: fresh preparation then five full fits and scores.

No selector or test reader is exposed. --recover preserves all previous attempts
and reuses verified fit/score checkpoints within the ORIGINAL ten-hour deadline.
An interrupted preparation needs a new run; models are not fitted before it commits.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from . import contract, preparation, models
from experiments.c03.engineering import lifecycle, lifecycle_probe, resource_policy
from experiments.c03.engineering import preparation_probe as host, xgb_full_row_sizing_probe as gpu
from experiments.c03.producer.plan import ROOT, OUTPUT_ROOT, canonical, digest, file_hash, require
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer.artifacts import safe_artifact


def checkpoint(store, name, phase, fit=None):
    root = safe_artifact(store.path, f'attempts/{name}-{phase}')
    complete = list(root.glob('a*/complete.json'))
    require(len(complete) <= 1, 'Multiple completed stage receipts')
    if not complete:
        return None
    directory = safe_artifact(root, complete[0].parent.name)
    if phase == 'prepare':
        receipt = store.read_json(directory/'complete.json')
        preparation.verify(store)
        require(receipt == {'scope': contract.SCOPE, 'plan_sha256': digest(store.plan),
                'prepared_receipt_sha256': file_hash(store.path/'prepared/complete.json')}, 'Preparation checkpoint drift')
    else:
        models.verify(store, directory, name, phase, fit)
    return directory


def watch_owner(pid, created, budget, done):
    while not done.wait(.3):
        if not lifecycle_probe.supervisor_alive(pid, created) or budget.elapsed() >= contract.LIMITS['seconds']:
            print('Worker stopping: supervisor exited or shared deadline expired', file=sys.stderr, flush=True)
            os._exit(15)


def worker(run, name, phase, attempt):
    store = contract.open_store(run)
    request = store.read_json(run/'request.json')
    require(request['limits'] == contract.LIMITS and request['plan_sha256'] == digest(store.plan), 'Worker request differs')
    directory = safe_artifact(run, f'attempts/{name}-{phase}/{attempt}')
    job = store.read_json(directory/'job.json')
    require(job['request_sha256'] == file_hash(run/'request.json') and job['name'] == name and job['phase'] == phase,
            'Worker job differs')
    require(lifecycle_probe.supervisor_alive(job['owner_pid'], job['owner_created']), 'No live supervisor')
    budget, done = lifecycle_probe.Budget(request), threading.Event()
    threading.Thread(target=watch_owner, args=(job['owner_pid'], job['owner_created'], budget, done), daemon=True).start()
    try:
        with file_lock(run/('.worker-'+name+'.guard')), (directory/'stages.jsonl').open('x', encoding='utf-8') as stream:
            def emit(stage):
                stream.write(json.dumps({'stage': name+':'+phase+':'+stage, 'seconds': budget.elapsed()})+'\n')
                stream.flush()
            if phase == 'prepare':
                require(name == 'preparation', 'Wrong preparation stage')
                from experiments.c03.engineering.combined_preparation_probe import inventory
                from experiments.c03.producer.sources import TrainingValidationFiles
                sources = TrainingValidationFiles(host.DATA_ROOT, inventory())
                result = preparation.prepare(store, sources, emit)
                # Fresh numeric preparation must agree with the already measured
                # exact same historical recipe; no engineering artifact is promoted.
                reference = lifecycle.prep.load_success()['identities']
                require(all(result['bindings'][s] == reference['bindings'][s] for s in ('stage1', 'stage2'))
                        and result['validation'] == reference['validation'], 'Fresh preparation differs from verified historical inputs')
                preparation.verify(store)
                payload = {'scope': contract.SCOPE, 'plan_sha256': digest(store.plan),
                           'prepared_receipt_sha256': file_hash(store.path/'prepared/complete.json')}
            else:
                require(name in contract.COMPONENTS and phase in ('fit', 'score'), 'Unknown component')
                record = models.imported(store, name, phase)
                emit('load_'+phase+'_array')
                X, y, weight = lifecycle.load_input(safe_artifact(run, record['array_path']), record, name, phase)
                if phase == 'fit':
                    result = models.fit(store, directory, name, record, X, y, weight, emit)
                    fit_sha = None
                else:
                    fit_dir = checkpoint(store, name, 'fit')
                    require(fit_dir is not None, 'No completed fit for scoring')
                    fit_result = store.read_json(fit_dir/'complete.json')['result']
                    result = models.score(store, directory, fit_dir, fit_result, name, X, emit)
                    fit_sha = file_hash(fit_dir/'complete.json')
            require(contract.open_store(run).plan == store.plan, 'Source changed during worker')
            require(budget.elapsed() < contract.LIMITS['seconds'], 'Deadline reached before checkpoint')
            if phase == 'prepare':
                publish(directory/'complete.json', canonical(payload))
            else:
                models.commit(store, directory, name, phase, record, result, fit_sha)
            emit('checkpoint_committed')
    finally:
        done.set()


def run_stage(store, name, phase, request, budget):
    import psutil
    run = store.path
    directory = safe_artifact(run, f'attempts/{name}-{phase}/a'+uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    publish(directory/'job.json', canonical({'request_sha256': file_hash(run/'request.json'),
        'name': name, 'phase': phase, 'owner_pid': os.getpid(), 'owner_created': psutil.Process().create_time()}))
    process = tree = None
    reason, telemetry, state = None, None, None
    peak, minimum, max_size, gpu_peak = 0, None, 0, 0
    previous_stage, last_report = None, float('-inf')
    try:
        telemetry, state = gpu.query_gpu(), host.resources(run)
        require(gpu.gpu_identity(telemetry) == request['gpu'], 'GPU identity changed')
        minimum = state['available']
        reason = resource_policy.start_reason(state, telemetry if '_xgb_' in name else None, limits=contract.LIMITS)
        reason = reason or resource_policy.stop_reason(state, telemetry, 0, host.artifact_size(run), budget.elapsed(), limits=contract.LIMITS)
        if not reason:
            command = [sys.executable, '-B', '-u', '-m', 'experiments.c03.pilot.runner',
                       '--_worker', str(run), '--component', name, '--phase', phase, '--attempt', directory.name]
            with (directory/'worker.log').open('x', encoding='utf-8') as log, (directory/'monitor.jsonl').open('x', encoding='utf-8') as trace:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                tree = host.ProcessTree(process.pid)
                last_gpu = time.monotonic()
                while True:
                    state, rss, size = host.resources(run), tree.rss(), host.artifact_size(run)
                    if time.monotonic()-last_gpu >= contract.LIMITS['gpu_interval']:
                        telemetry = gpu.query_gpu()
                        require(gpu.gpu_identity(telemetry) == request['gpu'], 'GPU identity changed')
                        last_gpu = time.monotonic()
                    elapsed = budget.elapsed()
                    peak, minimum, max_size = max(peak, rss), min(minimum, state['available']), max(max_size, size)
                    gpu_peak = max(gpu_peak, telemetry['used'])
                    trace.write(json.dumps({'seconds': elapsed, 'rss': rss, 'artifact_bytes': size, 'gpu': telemetry, **state})+'\n')
                    trace.flush()
                    stage = 'starting'
                    try:
                        text = (directory/'stages.jsonl').read_text()
                        lines = text[:text.rfind('\n')+1].splitlines()
                        if lines:
                            stage = json.loads(lines[-1])['stage']
                    except (OSError, ValueError, KeyError):
                        pass
                    if stage != previous_stage or elapsed-last_report >= 30:
                        print(f'[{elapsed/3600:.3f}/10 h] {stage} | worker {rss/2**30:.2f}/4 GiB | '
                              f'available {state["available"]/2**30:.2f} GiB (telemetry) | GPU {telemetry["used"]/2**30:.2f}/4 GiB | '
                              f'artifacts {size/2**30:.2f}/15 GiB', file=sys.stderr, flush=True)
                        previous_stage, last_report = stage, elapsed
                    reason = resource_policy.stop_reason(state, telemetry, rss, size, elapsed, limits=contract.LIMITS)
                    if reason:
                        break
                    if process.poll() is not None:
                        require(not tree.live(), 'Launcher exited with live descendants')
                        require(process.returncode == 0, f'Worker exit {process.returncode}; see {directory.relative_to(run)}/worker.log')
                        fit_dir = checkpoint(store, name, 'fit') if phase == 'score' else None
                        require(checkpoint(store, name, phase, fit_dir) == directory, 'Worker checkpoint missing')
                        break
                    time.sleep(contract.LIMITS['interval'])
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if tree is not None:
                tree.stop()
            elif process is not None and process.poll() is None:
                host.ProcessTree(process.pid).stop()
            if process is not None:
                process.wait(timeout=10)
        except Exception as exc:
            reason = f'Cleanup failed: {exc}; prior={reason}'
    if not reason:
        try:
            telemetry, state = gpu.query_gpu(), host.resources(run)
            require(gpu.gpu_identity(telemetry) == request['gpu'], 'GPU identity changed')
            reason = resource_policy.stop_reason(state, telemetry, 0, host.artifact_size(run), budget.elapsed(), limits=contract.LIMITS)
        except Exception as exc:
            reason = f'Final monitoring failed: {exc}'
    result = {'scope': contract.SCOPE, 'name': name, 'phase': phase, 'status': 'STOPPED' if reason else 'COMPLETE',
        'reason': reason, 'peak_tree_rss': peak, 'minimum_available': minimum, 'max_observed_artifact_bytes': max_size,
        'peak_gpu_used': gpu_peak, 'gpu_memory_scope': 'WHOLE_DEVICE', 'batch_elapsed_seconds': budget.elapsed(),
        'worker_exit': None if process is None else process.returncode, 'attempt': directory.relative_to(run).as_posix()}
    publish(directory/'supervisor.json', canonical(result))
    return result


def verify_completed(store):
    record = store.read_json(store.path/'producer-complete.json')
    require(record['scope'] == contract.SCOPE and record['plan_sha256'] == digest(store.plan)
            and record['selection_performed'] is False and record['test_accessed'] is False, 'Producer scope differs')
    expected = {}
    for name, phase in [('preparation', 'prepare'), *[(n, p) for n in contract.COMPONENTS for p in ('fit', 'score')]]:
        fit = checkpoint(store, name, 'fit') if phase == 'score' else None
        path = checkpoint(store, name, phase, fit)
        require(path is not None, 'Missing completed stage')
        expected[name+':'+phase] = file_hash(path/'complete.json')
    require(record['checkpoints'] == expected, 'Producer checkpoint drift')
    summary_path = safe_artifact(store.path, record['summary_path'])
    require(file_hash(summary_path) == record['summary_sha256'], 'Producer summary drift')
    summary = store.read_json(summary_path)
    require(summary['status'] == 'COMPLETE' and summary['elapsed_seconds'] < contract.LIMITS['seconds'], 'Producer deadline/status differs')
    return summary


def execute(store, *, recover=False):
    run = store.path
    with store.pipeline_lock():
        request = store.read_json(run/'request.json')
        require(request['plan_sha256'] == digest(store.plan) and request['limits'] == contract.LIMITS
                and request['deadline_epoch'] == request['started_epoch']+contract.LIMITS['seconds'], 'Request drift')
        if (run/'producer-complete.json').exists():
            return verify_completed(store)
        budget = lifecycle_probe.Budget(request)
        require(budget.elapsed() < contract.LIMITS['seconds'], 'Original ten-hour budget exhausted')
        with ExitStack() as locks:
            for name in ('preparation', *contract.COMPONENTS):
                locks.enter_context(file_lock(run/('.worker-'+name+'.guard')))
        require(recover or not (run/'attempts').exists(), 'Existing attempts require explicit recovery')
        outputs, reused, checkpoints, reason = {}, [], {}, None
        try:
            stages = [('preparation', 'prepare'), *[(n, p) for n in contract.COMPONENTS for p in ('fit', 'score')]]
            for index, (name, phase) in enumerate(stages):
                fit = checkpoint(store, name, 'fit') if phase == 'score' else None
                done = checkpoint(store, name, phase, fit)
                key = name+':'+phase
                if done:
                    reused.append(key)
                    print('Verified checkpoint reused: '+key, file=sys.stderr, flush=True)
                else:
                    if phase == 'prepare':
                        require(not (run/'prepared').exists(), 'Interrupted preparation requires a fresh run; old evidence preserved')
                    outputs[key] = run_stage(store, name, phase, request, budget)
                    require(outputs[key]['status'] == 'COMPLETE', outputs[key]['reason'] or 'Incomplete stage')
                    done = checkpoint(store, name, phase, fit)
                require(budget.elapsed() < contract.LIMITS['seconds'], 'Shared deadline reached')
                checkpoints[key] = file_hash(done/'complete.json')
                print(f'Stages complete {index+1}/{len(stages)}', file=sys.stderr, flush=True)
            require(contract.open_store(run).plan == store.plan, 'Final producer source drift')
            telemetry, state = gpu.query_gpu(), host.resources(run)
            require(gpu.gpu_identity(telemetry) == request['gpu'], 'GPU identity changed')
            reason = resource_policy.stop_reason(state, telemetry, 0, host.artifact_size(run), budget.elapsed(), limits=contract.LIMITS)
        except BaseException as exc:
            reason = f'{type(exc).__name__}: {exc}'
        summary = {'scope': contract.SCOPE, 'status': 'COMPLETE' if reason is None else 'STOPPED',
            'reason': reason, 'elapsed_seconds': budget.elapsed(), 'stages': outputs, 'reused_stages': reused,
            'completed_stages': list(checkpoints), 'run_directory': str(run)}
        directory = run/'summaries'
        directory.mkdir(exist_ok=True)
        path = directory/('a'+uuid.uuid4().hex+'.json')
        publish(path, canonical(summary))
        if summary['status'] == 'COMPLETE':
            publish(run/'producer-complete.json', canonical({'scope': contract.SCOPE,
                'plan_sha256': digest(store.plan), 'checkpoints': checkpoints,
                'summary_path': path.relative_to(run).as_posix(), 'summary_sha256': file_hash(path),
                'selection_performed': False, 'test_accessed': False}))
            verify_completed(store)
        return summary


def start(recover=None):
    plan = contract.frozen_plan()
    state, telemetry = host.resources(OUTPUT_ROOT), resource_policy.query_gpu_start()
    require(gpu.gpu_identity(telemetry) == gpu.load_readiness()['selected_gpu'], 'Unqualified GPU identity')
    reason = resource_policy.start_reason(state, telemetry, limits=contract.LIMITS)
    if reason:
        return {'scope': contract.SCOPE, 'status': 'START_REFUSED', 'reason': reason}
    if recover:
        store = contract.open_store(recover)
    else:
        name = 'pilot-producer-'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz')
        store = contract.Store.create(OUTPUT_ROOT, name, plan)
        started = time.time()
        publish(store.path/'request.json', canonical({'plan_sha256': digest(plan), 'limits': contract.LIMITS,
            'started_epoch': started, 'deadline_epoch': started+contract.LIMITS['seconds'], 'gpu': gpu.gpu_identity(telemetry)}))
    print('Scientific producer run: '+str(store.path), file=sys.stderr, flush=True)
    return execute(store, recover=recover is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--plan', action='store_true')
    modes.add_argument('--run', action='store_true')
    modes.add_argument('--resume', type=Path)
    modes.add_argument('--recover', type=Path)
    modes.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--component', choices=('preparation', *contract.COMPONENTS), help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=('prepare', 'fit', 'score'), help=argparse.SUPPRESS)
    parser.add_argument('--attempt', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker, args.component, args.phase, args.attempt)
        return 0
    result = (contract.frozen_plan() if args.plan else verify_completed(contract.open_store(args.resume))
              if args.resume else start(args.recover))
    print(json.dumps(result, indent=2))
    return 0 if args.plan or result['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
