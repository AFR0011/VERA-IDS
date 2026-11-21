"""One approved, bounded five-component fit/save/reload/validation rehearsal.

--probe creates a fresh engineering run; --resume only verifies completion.
--recover explicitly retries unfinished phases in new immutable attempt folders,
reusing completed fit/score receipts. The original wall-clock deadline, including
downtime, and artifact budget never reset. No mid-estimator/score-array resume.
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

from . import lifecycle as core
from . import preparation_probe as host
from . import xgb_full_row_sizing_probe as gpu_guard
from . import resource_policy
from experiments.c03.producer.artifacts import safe_artifact
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer.plan import ROOT, OUTPUT_ROOT, build_plan, canonical, digest, file_hash, require, output_location

LIMITS = dict(resource_policy.LIMITS)
SCOPE, COMPONENTS = core.SCOPE, core.COMPONENTS


def verify_request(run):
    output_location(run)
    request = core.read(safe_artifact(run, 'request.json'))
    require({k: v for k, v in request.items() if k not in ('started_epoch', 'deadline_epoch', 'selected_gpu')} == core.request_base(),
            'Lifecycle source/definition drift')
    require(request['deadline_epoch'] == request['started_epoch']+LIMITS['seconds'], 'Deadline differs')
    require(request['selected_gpu'] == gpu_guard.load_readiness()['selected_gpu'], 'Qualified GPU differs')
    return request


class Budget:
    def __init__(self, request):
        self.start = request['started_epoch']
        self.base = time.time()-self.start
        require(self.base >= 0, 'Clock moved before run start')
        self.monotonic = time.monotonic()

    def elapsed(self):
        return max(self.base+time.monotonic()-self.monotonic, time.time()-self.start)


def completed_phase(run, name, phase, fit=None):
    root = safe_artifact(run, f'components/{name}/{phase}')
    matches = sorted(root.glob('a*/complete.json'))
    require(len(matches) <= 1, 'Multiple completed phase receipts')
    if not matches:
        return None
    directory = safe_artifact(root, matches[0].parent.name)
    expected_fit = file_hash(fit/'complete.json') if fit is not None else None
    receipt = core.verify_phase(directory, name, phase, file_hash(run/'request.json'), fit_receipt_sha=expected_fit)
    if '_xgb_' in name:
        recorded = receipt['result']['fitted'] if phase == 'fit' else receipt['result']['inference']
        require(recorded['device'] == 'cuda:0', 'Unapproved CPU fallback in lifecycle receipt')
    if phase == 'score':
        require(fit is not None and receipt['result']['model_sha256'] == core.read(fit/'complete.json')['result']['model_sha256'],
                'Scored model differs from committed fit')
    return directory


def validate_current_import(run, name, phase, directory):
    success = core.prep.load_success()
    source = safe_artifact(OUTPUT_ROOT, success['run_id'])
    imported = core.verify_import(source, success, build_plan(), name, phase)
    require(core.read(directory/'complete.json')['import'] == imported, 'Reused phase import drift')


def supervisor_alive(pid, created):
    import psutil
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.create_time() == created
    except psutil.Error:
        return False


def watch_owner(pid, created, budget, done):
    while not done.wait(.3):
        if not supervisor_alive(pid, created) or budget.elapsed() >= LIMITS['seconds']:
            print('Worker stopped: supervisor exited or original deadline expired', file=sys.stderr, flush=True)
            os._exit(15)


def worker(run, name, phase, attempt):
    request = verify_request(run)
    budget = Budget(request)
    require(name in COMPONENTS and phase in ('fit', 'score'), 'Invalid worker phase')
    directory = safe_artifact(run, f'components/{name}/{phase}/{attempt}')
    job = core.read(directory/'job.json')
    require(job['request_sha256'] == file_hash(run/'request.json') and job['component'] == name and job['phase'] == phase,
            'Worker job drift')
    require(supervisor_alive(job['owner_pid'], job['owner_created']), 'Missing live supervisor')
    done = threading.Event()
    watcher = threading.Thread(target=watch_owner, args=(job['owner_pid'], job['owner_created'], budget, done), daemon=True)
    watcher.start()
    try:
        with file_lock(run/('.worker-'+name+'.guard')), (directory/'stages.jsonl').open('x', encoding='utf-8') as log:
            def emit(stage):
                log.write(json.dumps({'stage': name+':'+phase+':'+stage, 'seconds': budget.elapsed()})+'\n')
                log.flush()
            success, plan = core.prep.load_success(), build_plan()
            source = safe_artifact(OUTPUT_ROOT, success['run_id'])
            emit('verify_import')
            imported = core.verify_import(source, success, plan, name, phase)
            emit('load_'+phase+'_input')
            X, y, weight = core.load_input(safe_artifact(source, imported['array_path']), imported, name, phase)
            fit_sha = None
            if phase == 'fit':
                result = core.fit_save(directory, plan, name, imported, X, y, weight, emit)
            else:
                fit_dir = completed_phase(run, name, 'fit')
                require(fit_dir is not None, 'Scoring requires a completed fit')
                fit = core.read(fit_dir/'complete.json')
                fit_sha = file_hash(fit_dir/'complete.json')
                result = core.reload_score(directory, fit_dir, fit['result'], name, X, emit)
            emit('verify_before_commit')
            require(core.verify_import(source, success, plan, name, phase) == imported, 'Import changed during phase')
            require(verify_request(run) == request, 'Worker source changed')
            require(budget.elapsed() < LIMITS['seconds'], 'Shared deadline reached before commit')
            core.commit_phase(directory, name, phase, file_hash(run/'request.json'), imported, result, fit_receipt_sha=fit_sha)
            emit('checkpoint_committed')
    finally:
        done.set()


def command(run, name, phase, attempt):
    return [sys.executable, '-B', '-u', '-m', 'experiments.c03.engineering.lifecycle_probe',
            '--_worker', str(run), '--component', name, '--phase', phase, '--attempt', attempt]


def run_phase(run, name, phase, request, budget):
    import psutil
    directory = safe_artifact(run, f'components/{name}/{phase}/a'+uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    publish(directory/'job.json', canonical({'request_sha256': file_hash(run/'request.json'),
        'component': name, 'phase': phase, 'owner_pid': os.getpid(), 'owner_created': psutil.Process().create_time()}))
    process = tree = None
    reason, gpu, state = None, None, None
    peak, minimum, max_bytes, gpu_peak, gpu_min = 0, None, 0, None, None
    progress = gpu_guard.Progress(directory)
    phase_started = time.monotonic()
    try:
        gpu, state = gpu_guard.query_gpu(), host.resources(run)
        require(gpu_guard.gpu_identity(gpu) == request['selected_gpu'], 'GPU identity changed')
        minimum, gpu_peak, gpu_min = state['available'], gpu['used'], gpu['free']
        reason = resource_policy.stop_reason(state, gpu, 0, host.artifact_size(run), budget.elapsed()) or resource_policy.start_reason(state)
        if '_xgb_' in name:
            reason = reason or resource_policy.start_reason(state, gpu)
        if not reason:
            with (directory/'worker.log').open('x', encoding='utf-8') as log, (directory/'monitor.jsonl').open('x', encoding='utf-8') as trace:
                process = subprocess.Popen(command(run, name, phase, directory.name), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                tree = host.ProcessTree(process.pid)
                last_gpu = time.monotonic()
                while True:
                    state, rss, size = host.resources(run), tree.rss(), host.artifact_size(run)
                    if time.monotonic()-last_gpu >= LIMITS['gpu_interval']:
                        gpu = gpu_guard.query_gpu()
                        require(gpu_guard.gpu_identity(gpu) == request['selected_gpu'], 'GPU identity changed')
                        last_gpu = time.monotonic()
                    elapsed = budget.elapsed()
                    peak, minimum, max_bytes = max(peak, rss), min(minimum, state['available']), max(max_bytes, size)
                    gpu_peak, gpu_min = max(gpu_peak, gpu['used']), min(gpu_min, gpu['free'])
                    trace.write(json.dumps({'seconds': elapsed, 'rss': rss, 'artifact_bytes': size, 'gpu': gpu, **state})+'\n')
                    trace.flush()
                    progress.update(elapsed, rss, state, gpu, size)
                    reason = resource_policy.stop_reason(state, gpu, rss, size, elapsed)
                    if reason:
                        break
                    if process.poll() is not None:
                        require(not tree.live(), 'Launcher exited with live descendants')
                        require(process.returncode == 0, f'Worker exit {process.returncode}; see {directory.name}/worker.log')
                        fit = completed_phase(run, name, 'fit')
                        require(completed_phase(run, name, phase, fit if phase == 'score' else None) == directory,
                                'Worker lacks its completed checkpoint')
                        break
                    time.sleep(LIMITS['interval'])
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
            reason = f'Worker cleanup failed: {exc}; earlier={reason}'
    if not reason:
        try:
            gpu, state = gpu_guard.query_gpu(), host.resources(run)
            require(gpu_guard.gpu_identity(gpu) == request['selected_gpu'], 'GPU identity changed')
            gpu_peak, gpu_min = max(gpu_peak, gpu['used']), min(gpu_min, gpu['free'])
            minimum, max_bytes = min(minimum, state['available']), max(max_bytes, host.artifact_size(run))
            reason = resource_policy.stop_reason(state, gpu, 0, max_bytes, budget.elapsed())
        except Exception as exc:
            reason = f'Final resource verification failed: {exc}'
    summary = {'scope': SCOPE, 'component': name, 'phase': phase, 'status': 'STOPPED' if reason else 'COMPLETE',
        'reason': reason, 'peak_tree_rss': peak, 'minimum_available': minimum,
        'max_observed_artifact_bytes': max_bytes, 'peak_gpu_used': gpu_peak, 'minimum_gpu_free': gpu_min,
        'gpu_memory_scope': 'WHOLE_DEVICE', 'last_gpu_sample': gpu, 'last_host_sample': state,
        'wall_seconds': time.monotonic()-phase_started, 'batch_elapsed_seconds': budget.elapsed(),
        'worker_exit': None if process is None else process.returncode, 'attempt': directory.relative_to(run).as_posix()}
    publish(directory/'supervisor.json', canonical(summary))
    print(f'[{host.Progress.clock(budget.elapsed())}] {name}:{phase} {summary["status"]}'+
          (f' | {reason}' if reason else ''), file=sys.stderr, flush=True)
    return summary


def verify_completed(run):
    verify_request(run)
    receipt = core.read(run/'complete.json')
    require(receipt['scope'] == SCOPE and receipt['scientific_checkpoint'] is False
            and receipt['request_sha256'] == file_hash(run/'request.json'), 'Completion scope/source drift')
    require(set(receipt['checkpoints']) == set(COMPONENTS), 'Incomplete component set')
    for name in COMPONENTS:
        fit = completed_phase(run, name, 'fit')
        score = completed_phase(run, name, 'score', fit)
        require(fit is not None and score is not None and receipt['checkpoints'][name] ==
                {'fit': file_hash(fit/'complete.json'), 'score': file_hash(score/'complete.json')}, 'Checkpoint receipt drift')
    summary_path = safe_artifact(run, receipt['summary_path'])
    require(file_hash(summary_path) == receipt['summary_sha256'], 'Summary drift')
    summary = core.read(summary_path)
    require(summary['status'] == 'COMPLETE' and summary['wall_seconds'] < LIMITS['seconds'], 'Incomplete supervision')
    return {**summary, 'run_directory': str(run), 'reused_completed_result': True}


def execute(run, *, recover=False):
    with file_lock(run/'batch.guard'):
        request = verify_request(run)
        if (run/'complete.json').exists():
            return verify_completed(run)
        budget = Budget(request)
        require(budget.elapsed() < LIMITS['seconds'], 'Original 900-second deadline exhausted; no budget reset')
        # Check every possible orphan worker before reusing or starting any work.
        with ExitStack() as locks:
            for name in COMPONENTS:
                locks.enter_context(file_lock(run/('.worker-'+name+'.guard')))
        if not recover:
            require(not (run/'components').exists(), 'Existing attempts require explicit --recover')
        phases, reused, checkpoints = {}, [], {}
        reason = None
        try:
            for index, name in enumerate(COMPONENTS):
                fit = None
                for phase in ('fit', 'score'):
                    directory = completed_phase(run, name, phase, fit if phase == 'score' else None)
                    key = name+':'+phase
                    if directory is not None:
                        validate_current_import(run, name, phase, directory)
                        reused.append(key)
                        print('Verified checkpoint reused: '+key, file=sys.stderr, flush=True)
                    else:
                        phases[key] = run_phase(run, name, phase, request, budget)
                        require(phases[key]['status'] == 'COMPLETE', phases[key]['reason'] or 'Incomplete phase')
                        directory = completed_phase(run, name, phase, fit if phase == 'score' else None)
                    require(budget.elapsed() < LIMITS['seconds'], 'Original shared deadline exhausted')
                    if phase == 'fit':
                        fit = directory
                    else:
                        checkpoints[name] = {'fit': file_hash(fit/'complete.json'), 'score': file_hash(directory/'complete.json')}
                print(f'Components complete: {index+1}/{len(COMPONENTS)}', file=sys.stderr, flush=True)
            require(verify_request(run) == request, 'Source changed before batch completion')
            gpu, state = gpu_guard.query_gpu(), host.resources(run)
            require(gpu_guard.gpu_identity(gpu) == request['selected_gpu'], 'GPU identity changed')
            reason = resource_policy.stop_reason(state, gpu, 0, host.artifact_size(run), budget.elapsed())
        except BaseException as exc:
            reason = f'{type(exc).__name__}: {exc}'
        summary = {'scope': SCOPE, 'status': 'COMPLETE' if not reason and len(checkpoints) == 5 else 'STOPPED',
            'reason': reason, 'wall_seconds': budget.elapsed(), 'limits': LIMITS,
            'phases': phases, 'reused_phases': reused, 'completed_components': list(checkpoints)}
        summaries = run/'summaries'
        summaries.mkdir(exist_ok=True)
        path = summaries/('a'+uuid.uuid4().hex+'.json')
        publish(path, canonical(summary))
        if summary['status'] == 'COMPLETE':
            publish(run/'complete.json', canonical({'scope': SCOPE, 'scientific_checkpoint': False,
                'request_sha256': file_hash(run/'request.json'), 'checkpoints': checkpoints,
                'summary_path': path.relative_to(run).as_posix(), 'summary_sha256': file_hash(path)}))
            verify_completed(run)
        return {**summary, 'run_directory': str(run)}


def probe(recover=None):
    base = core.request_base()
    readiness = gpu_guard.load_readiness()
    core.prep.load_success()  # Compact metadata only before the start gate.
    state, gpu = host.resources(OUTPUT_ROOT), None
    try:
        gpu = resource_policy.query_gpu_start()
        require(gpu_guard.gpu_identity(gpu) == readiness['selected_gpu'], 'Unqualified GPU identity')
        state = host.resources(OUTPUT_ROOT)
        reason = resource_policy.start_reason(state, gpu)
    except Exception as exc:
        reason = f'GPU monitoring unavailable: {type(exc).__name__}: {exc}'
    if reason:
        return {'scope': SCOPE, 'status': 'START_REFUSED', 'reason': reason, **state, 'gpu': gpu}
    if recover is None:
        run = output_location(OUTPUT_ROOT/('lifecycle-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
        run.mkdir(parents=True, exist_ok=False)
        started = time.time()
        publish(run/'request.json', canonical({**base, 'started_epoch': started,
            'deadline_epoch': started+LIMITS['seconds'], 'selected_gpu': gpu_guard.gpu_identity(gpu)}))
    else:
        run = output_location(recover)
    print('Rehearsal run: '+str(run), file=sys.stderr, flush=True)
    return execute(run, recover=recover is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe', action='store_true')
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--resume', type=Path, help='Verify completed result; no fitting, scoring or GPU query')
    group.add_argument('--recover', type=Path, help='Explicit recovery within the ORIGINAL deadline; no budget reset')
    group.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--component', choices=COMPONENTS, help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=('fit', 'score'), help=argparse.SUPPRESS)
    parser.add_argument('--attempt', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker, args.component, args.phase, args.attempt)
        return 0
    result = ({**host.resources(OUTPUT_ROOT), 'gpu': resource_policy.query_gpu_start()} if args.inspect else
              verify_completed(args.resume.resolve()) if args.resume else probe(args.recover))
    print(json.dumps(result, indent=2))
    return 0 if args.inspect or result['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
