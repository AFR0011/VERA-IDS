"""Approved synthetic XGBoost CUDA readiness; no benchmark readers or model files.

Binary and five-class workers share one five-minute budget. --resume verifies a
completed batch. --fork starts a separately authorized recovery attempt, retaining
completed diagnostics. No CPU fallback, automatic retry or mid-fit continuation.
"""
import argparse
from contextlib import ExitStack
import copy
import csv
from datetime import datetime, timezone
import gc
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import warnings

from . import preparation_probe as host
from experiments.c03.producer.artifacts import safe_artifact
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer.plan import ROOT, OUTPUT_ROOT, build_plan, canonical, digest, file_hash, require

GIB = 1024**3
SCOPE = 'SYNTHETIC_XGBOOST_CUDA_READINESS_ONLY'
COMPONENTS = ('stage1_xgb_inv_family_clipped', 'stage2_xgb_shared')
DEFINITION = {'components': list(COMPONENTS), 'rows': 4096, 'features': 46,
              'generation_seed': 20261108, 'n_estimators': 8, 'n_jobs': 2,
              'device': 'cuda:0', 'model_seeds': [123, 133]}
LIMITS = {'seconds': 300, 'interval': .3, 'gpu_interval': 1., 'query_timeout': 2.,
          'start_available': 3*GIB, 'max_rss': 2*GIB, 'min_available': 2*GIB,
          'start_gpu_free': 6*GIB, 'max_gpu_used': 4*GIB, 'min_gpu_free': 2*GIB,
          'start_disk': 2*GIB, 'min_disk': GIB, 'max_artifacts': 64*1024**2}
FILES = ('request.json', 'worker-result.json', 'supervisor.json', 'stages.jsonl',
         'monitor.jsonl', 'worker.log', 'complete.json')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_record():
    plan = build_plan()
    return {'plan_sha256': digest(plan), 'sources': {
        p.relative_to(ROOT).as_posix(): file_hash(p) for p in
        [Path(__file__), ROOT/'tests/test_c03_cuda_readiness_probe.py']}}


def request_record():
    return {'scope': SCOPE, 'definition': DEFINITION, 'limits': LIMITS, 'source_record': source_record()}


def parse_gpu(text):
    rows = list(csv.reader(io.StringIO(text.strip()), skipinitialspace=True))
    require(len(rows) == 1 and len(rows[0]) == 6, 'Invalid GPU query shape')
    index, name, total, used, free, driver = [v.strip() for v in rows[0]]
    require(index == '0' and name and driver, 'Unexpected GPU identity')
    require(all(v.isdigit() for v in (total, used, free)), 'Invalid GPU memory reading')
    total, used, free = [int(v)*1024**2 for v in (total, used, free)]
    require(total > 0 and 0 <= used <= total and 0 <= free <= total and used+free <= total,
            'Inconsistent GPU memory reading')
    return {'index': 0, 'name': name, 'total': total, 'used': used, 'free': free, 'driver': driver}


def query_gpu():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0'), 'Unsupported CUDA device remapping')
    # This approved laptop has one NVIDIA GPU. Reject ambiguous multi-GPU
    # enumeration rather than monitor a different physical device from cuda:0.
    result = subprocess.run(['nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        timeout=LIMITS['query_timeout'], check=True)
    return parse_gpu(result.stdout)


def gpu_identity(gpu):
    return {key: gpu[key] for key in ('index', 'name', 'total', 'driver')}


def start_reason(state, gpu):
    if state['available'] < LIMITS['start_available']:
        return 'Less than 3 GiB available RAM'
    if state['disk_free'] < LIMITS['start_disk']:
        return 'Less than 2 GiB free disk'
    if gpu['free'] < LIMITS['start_gpu_free']:
        return 'Less than 6 GiB free GPU memory'


def stop_reason(state, gpu, rss, size, elapsed):
    if elapsed >= LIMITS['seconds']:
        return '300-second shared budget reached'
    if rss > LIMITS['max_rss']:
        return 'Worker-tree RSS above 2 GiB'
    if state['available'] < LIMITS['min_available']:
        return 'Available RAM below 2 GiB'
    if state['disk_free'] < LIMITS['min_disk']:
        return 'Free disk below 1 GiB'
    if size > LIMITS['max_artifacts']:
        return 'New artifacts above 64 MiB'
    if gpu['used'] > LIMITS['max_gpu_used']:
        return 'Total device-used VRAM above 4 GiB'
    if gpu['free'] < LIMITS['min_gpu_free']:
        return 'Free VRAM below 2 GiB'


class Progress:
    def __init__(self, profile):
        self.profile, self.last_stage, self.last_time = profile, None, float('-inf')

    def update(self, elapsed, rss, state, gpu, size):
        stage = 'starting_worker'
        try:
            raw = (self.profile/'stages.jsonl').read_text()
            lines = raw[:raw.rfind('\n')+1].splitlines()
            if lines:
                stage = json.loads(lines[-1])['stage']
        except (OSError, ValueError, KeyError):
            pass
        if stage != self.last_stage or elapsed-self.last_time >= 30:
            print(f'[{host.Progress.clock(elapsed)} / 05:00 budget] {stage} | '
                  f'worker {rss/GIB:.2f}/2.00 GiB | available {state["available"]/GIB:.2f} GiB | '
                  f'GPU used {gpu["used"]/GIB:.2f}/4.00 GiB, free {gpu["free"]/GIB:.2f} GiB | '
                  f'artifacts {size/1024**2:.2f}/64 MiB', file=sys.stderr, flush=True)
            self.last_stage, self.last_time = stage, elapsed


def generated_inputs(name):
    import numpy as np
    from scipy.sparse import csr_matrix
    from experiments.c03.producer.validation import historical
    require(name in COMPONENTS, 'Unknown component')
    rng = np.random.RandomState(DEFINITION['generation_seed'])
    X = csr_matrix(rng.standard_normal((DEFINITION['rows'], DEFINITION['features'])))
    _, grid, _ = historical()
    if name == COMPONENTS[0]:
        families = np.arange(len(X.indptr)-1, dtype=np.int64) % 6
        y = (families != 0).astype(np.int64)
        w = grid.make_stage1_sample_weights(y, families.astype(str), 'inv_family_clipped',
                                            float(grid.CFG['inv_family_clip_max']))
    else:
        y = np.arange(X.shape[0], dtype=np.int64) % 5
        w = grid.stage2_sample_weights(y, 5, 'balanced')
    return X, y, w


def check_booster(config, name, *, device='cuda:0'):
    learner = config['learner']
    require(learner['generic_param']['device'] == device, 'CPU fallback or unexpected fitted device')
    require(int(learner['generic_param']['nthread']) == 2, 'Fitted CPU worker count differs')
    require(int(learner['learner_model_param']['num_feature']) == 46, 'Fitted feature count differs')
    expected = 'binary:logistic' if name == COMPONENTS[0] else 'multi:softprob'
    require(learner['objective']['name'] == expected, 'Fitted objective differs')
    require(int(learner['learner_model_param']['num_class']) == (0 if name == COMPONENTS[0] else 5),
            'Fitted class count differs')
    require(learner['gradient_booster']['gbtree_train_param']['tree_method'] == 'hist', 'Fitted tree method differs')


def fit_generated(plan, name, emit, *, device='cuda:0'):
    # The private device argument permits CPU-only generated-data unit tests.
    # CLI workers always use the fixed CUDA definition and acceptance check.
    import xgboost as xgb
    from experiments.c03.producer.validation import historical
    from experiments.c03.producer import models
    from experiments.c03.producer.artifacts import tensor_hash, array_hash
    require(name in COMPONENTS and device in ('cuda:0', 'cpu'), 'Unsupported diagnostic definition')
    node = copy.deepcopy(plan['graph']['components'][name])
    _, grid, _ = historical()
    original = models.json_parameters(models.construct(grid, node).get_params())
    node['parameters'].update(n_estimators=8, n_jobs=2, device=device)
    model = models.construct(grid, node)
    parameters = models.json_parameters(model.get_params())
    require(parameters == {**original, 'n_estimators': 8, 'n_jobs': 2, 'device': device}, 'Unexpected parameter change')
    emit('generate_inputs')
    X, y, weight = generated_inputs(name)
    identities = lambda: [tensor_hash(X), array_hash(y), array_hash(weight)]
    before = identities()

    class Reporter(xgb.callback.TrainingCallback):
        def after_iteration(self, booster, epoch, evals_log):
            check_booster(json.loads(booster.save_config()), name, device=device)
            emit(f'round_{epoch+1}_of_8')
            return False

    model.set_params(callbacks=[Reporter()])
    emit('fit_cuda' if device == 'cuda:0' else 'fit_cpu_fixture')
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            model.fit(X, y, sample_weight=weight)
        finally:
            for item in caught:
                print('XGBoost warning: '+str(item.message), file=sys.stderr, flush=True)
    elapsed = time.perf_counter()-started
    messages = [str(item.message) for item in caught]
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    check_booster(config, name, device=device)
    require(booster.num_boosted_rounds() == 8, 'Incomplete boosting rounds')
    require(before == identities(), 'Fit mutated synthetic inputs')
    result = {'component': name, 'rows': X.shape[0], 'features': X.shape[1], 'input_hashes': before,
        'effective_parameters': parameters, 'booster_config': config, 'boosted_rounds': 8,
        'fit_seconds': elapsed, 'warnings': messages, 'quality_metrics': None, 'model_retained': False,
        'input_kind': 'GENERATED_ONLY', 'callbacks': 'read-only round/device reporter; no early stopping'}
    del model, booster, X, y, weight
    gc.collect()
    return result


def validate_result(result, name):
    require(result['scope'] == SCOPE and result['status'] == 'CUDA_COMPONENT_COMPLETE'
            and result['component'] == name and result['scientific_checkpoint'] is False
            and result['rows'] == 4096 and result['features'] == 46 and result['boosted_rounds'] == 8
            and result['model_retained'] is False and result['quality_metrics'] is None
            and result['input_kind'] == 'GENERATED_ONLY', 'Invalid CUDA diagnostic result')
    check_booster(result['booster_config'], name)


def worker(run, name):
    require(name in COMPONENTS, 'Unknown component')
    profile = safe_artifact(run, 'profiles/'+name)
    with file_lock(profile/'worker.guard'):
        request = read(profile/'request.json')
        require(request == {**request_record(), 'component': name}, 'Worker source/request drift')
        import xgboost
        require(xgboost.build_info()['USE_CUDA'] is True, 'Installed XGBoost lacks CUDA support')
        with (profile/'stages.jsonl').open('x', encoding='utf-8') as stream:
            def emit(stage):
                stream.write(json.dumps({'stage': name+'_'+stage})+'\n'); stream.flush()
            result = {'scope': SCOPE, 'status': 'CUDA_COMPONENT_COMPLETE', 'scientific_checkpoint': False,
                      **fit_generated(build_plan(), name, emit)}
            validate_result(result, name)
            require(request == {**request_record(), 'component': name}, 'Worker source changed during fit')
            publish(profile/'worker-result.json', canonical(result))
            emit('complete')


def verify_profile(profile, name):
    require(read(profile/'request.json') == {**request_record(), 'component': name}, 'Profile source/definition differs')
    receipt = read(profile/'complete.json')
    require(receipt['scope'] == SCOPE and receipt['component'] == name and receipt['scientific_checkpoint'] is False,
            'Profile completion scope differs')
    require(set(receipt['files']) == set(FILES)-{'complete.json'}, 'Incomplete profile receipt')
    for path, sha in receipt['files'].items():
        require(file_hash(safe_artifact(profile, path)) == sha, 'Profile receipt byte drift')
    result = read(profile/'worker-result.json')
    validate_result(result, name)
    require(read(profile/'supervisor.json')['status'] == 'COMPLETE', 'Incomplete profile supervision')
    return result


def command(run, name):
    return [sys.executable, '-B', '-u', '-m', 'experiments.c03.engineering.cuda_readiness_probe',
            '--_worker', str(run), '--component', name]


def run_profile(run, name, batch_started, selected_gpu):
    profile = run/'profiles'/name
    profile.mkdir(parents=True, exist_ok=False)
    publish(profile/'request.json', canonical({**request_record(), 'component': name}))
    process = tree = None
    reason, gpu, state = None, None, None
    started = time.monotonic()
    peak, minimum, max_bytes, gpu_peak, gpu_min = 0, None, 0, None, None
    progress = Progress(profile)
    try:
        gpu = query_gpu()
        state = host.resources(run)
        require(gpu_identity(gpu) == selected_gpu, 'Selected GPU identity changed')
        minimum, gpu_peak, gpu_min = state['available'], gpu['used'], gpu['free']
        reason = stop_reason(state, gpu, 0, host.artifact_size(run), time.monotonic()-batch_started) or start_reason(state, gpu)
        if not reason:
            with (profile/'worker.log').open('x', encoding='utf-8') as log, (profile/'monitor.jsonl').open('x', encoding='utf-8') as trace:
                process = subprocess.Popen(command(run, name), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                tree = host.ProcessTree(process.pid)
                last_gpu = time.monotonic()
                while True:
                    state, rss, size = host.resources(run), tree.rss(), host.artifact_size(run)
                    now = time.monotonic()
                    reason = stop_reason(state, gpu, rss, size, now-batch_started)
                    if not reason and now-last_gpu >= LIMITS['gpu_interval']:
                        gpu = query_gpu()
                        require(gpu_identity(gpu) == selected_gpu, 'Selected GPU identity changed')
                        last_gpu = time.monotonic()
                        state = host.resources(run)
                    elapsed = time.monotonic()-batch_started
                    peak, minimum, max_bytes = max(peak, rss), min(minimum, state['available']), max(max_bytes, size)
                    gpu_peak, gpu_min = max(gpu_peak, gpu['used']), min(gpu_min, gpu['free'])
                    trace.write(json.dumps({'seconds': elapsed, 'rss': rss, 'artifact_bytes': size,
                        'gpu': gpu, 'gpu_sample_age_seconds': time.monotonic()-last_gpu, **state})+'\n'); trace.flush()
                    progress.update(elapsed, rss, state, gpu, size)
                    reason = reason or stop_reason(state, gpu, rss, size, elapsed)
                    if reason:
                        break
                    if process.poll() is not None:
                        require(not tree.live(), 'Launcher exited with live descendants')
                        require(process.returncode == 0, f'Worker exit {process.returncode}')
                        validate_result(read(profile/'worker-result.json'), name)
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
            reason = f'Cleanup failed: {type(exc).__name__}: {exc}; earlier={reason}'
            if process is not None and process.poll() is None:
                process.kill()
    if not reason:
        try:
            gpu, state = query_gpu(), host.resources(run)
            require(gpu_identity(gpu) == selected_gpu, 'Selected GPU identity changed')
            gpu_peak, gpu_min = max(gpu_peak, gpu['used']), min(gpu_min, gpu['free'])
            minimum = min(minimum, state['available'])
            reason = stop_reason(state, gpu, 0, host.artifact_size(run), time.monotonic()-batch_started)
        except Exception as exc:
            reason = f'Final monitoring failed: {type(exc).__name__}: {exc}'
    summary = {'scope': SCOPE, 'component': name, 'status': 'STOPPED' if reason else 'COMPLETE', 'reason': reason,
        'peak_tree_rss': peak, 'minimum_available': minimum, 'max_observed_artifact_bytes': max_bytes,
        'peak_gpu_used': gpu_peak, 'minimum_gpu_free': gpu_min, 'gpu_memory_scope': 'WHOLE_DEVICE',
        'last_gpu_sample': gpu, 'last_host_sample': state, 'wall_seconds': time.monotonic()-started,
        'batch_elapsed_seconds': time.monotonic()-batch_started,
        'worker_exit': None if process is None else process.returncode, 'limits': LIMITS}
    publish(profile/'supervisor.json', canonical(summary))
    if not reason:
        publish(profile/'complete.json', canonical({'scope': SCOPE, 'component': name, 'scientific_checkpoint': False,
            'files': {path: file_hash(profile/path) for path in FILES if path != 'complete.json'}}))
        verify_profile(profile, name)
    print(f'[{host.Progress.clock(summary["batch_elapsed_seconds"])}] {name}: {summary["status"]}'
          + (f' | {reason}' if reason else ''), file=sys.stderr, flush=True)
    return summary


def verify_request(run):
    request = read(run/'request.json')
    require({k: v for k, v in request.items() if k not in ('recovery', 'selected_gpu')} == request_record(),
            'Batch source/definition differs')
    require(set(request['selected_gpu']) == {'index', 'name', 'total', 'driver'} and request['selected_gpu']['index'] == 0,
            'Missing selected GPU identity')
    if 'recovery' in request:
        reuse = read(run/'recovery.json')
        require(digest(reuse) == request['recovery'], 'Incomplete recovery commit')
        for name, files in reuse['profiles'].items():
            require(name in COMPONENTS and set(files) == set(FILES), 'Invalid recovered profile')
            for path, sha in files.items():
                require(file_hash(safe_artifact(run/'profiles'/name, path)) == sha, 'Recovered profile drift')
            verify_profile(run/'profiles'/name, name)
    return request


def execute(run):
    with file_lock(run/'batch.guard'):
        started = time.monotonic()
        request = verify_request(run)
        profiles, reused = {}, []
        for name in COMPONENTS:
            profile = run/'profiles'/name
            if (profile/'complete.json').exists():
                with file_lock(profile/'worker.guard'):
                    verify_profile(profile, name)
                reused.append(name)
                print('Reusing completed diagnostic: '+name, file=sys.stderr, flush=True)
                continue
            require(not profile.exists(), 'Incomplete component requires an explicit recovery fork')
            profiles[name] = run_profile(run, name, started, request['selected_gpu'])
            if profiles[name]['status'] != 'COMPLETE':
                break
        completed = [name for name in COMPONENTS if (run/'profiles'/name/'complete.json').exists()]
        reason = next((v['reason'] for v in profiles.values() if v['status'] != 'COMPLETE'), None)
        if time.monotonic()-started >= LIMITS['seconds']:
            reason = reason or '300-second shared budget reached'
        summary = {'scope': SCOPE, 'status': 'COMPLETE' if len(completed) == 2 and not reason else 'STOPPED',
            'reason': reason, 'wall_seconds': time.monotonic()-started, 'limits': LIMITS,
            'profiles': profiles, 'completed_components': completed, 'reused_components': reused}
        publish(run/'supervisor.json', canonical(summary))
        if summary['status'] == 'COMPLETE':
            publish(run/'complete.json', canonical({'scope': SCOPE, 'scientific_checkpoint': False,
                'request_sha256': file_hash(run/'request.json'), 'supervisor_sha256': file_hash(run/'supervisor.json'),
                'profiles': {name: file_hash(run/'profiles'/name/'complete.json') for name in COMPONENTS}}))
        return {**summary, 'run_directory': str(run)}


def verify_completed(run):
    verify_request(run)
    receipt, summary = read(run/'complete.json'), read(run/'supervisor.json')
    require(receipt['scope'] == SCOPE and receipt['scientific_checkpoint'] is False
            and summary['status'] == 'COMPLETE', 'Incomplete batch')
    require(receipt['request_sha256'] == file_hash(run/'request.json')
            and receipt['supervisor_sha256'] == file_hash(run/'supervisor.json'), 'Batch receipt drift')
    require(set(receipt['profiles']) == set(COMPONENTS), 'Missing completed components')
    for name, sha in receipt['profiles'].items():
        profile = safe_artifact(run, 'profiles/'+name)
        require(file_hash(profile/'complete.json') == sha, 'Profile completion receipt drift')
        verify_profile(profile, name)
    return {**summary, 'run_directory': str(run), 'reused_completed_result': True}


def new_run():
    run = OUTPUT_ROOT/('cuda-readiness-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    require(not run.resolve().is_relative_to(ROOT), 'Output redirected into repository')
    run.mkdir(parents=True, exist_ok=False)
    return run


def fork_completed(source, selected_gpu):
    with ExitStack() as stack:
        stack.enter_context(file_lock(source/'batch.guard'))
        request = verify_request(source)
        require(request['selected_gpu'] == selected_gpu, 'Recovery GPU identity differs')
        for name in COMPONENTS:
            profile = safe_artifact(source, 'profiles/'+name)
            if profile.exists():
                stack.enter_context(file_lock(profile/'worker.guard'))
        completed = {}
        for name in COMPONENTS:
            profile = safe_artifact(source, 'profiles/'+name)
            if (profile/'complete.json').exists():
                verify_profile(profile, name)
                completed[name] = {path: file_hash(profile/path) for path in FILES}
        require(completed, 'No completed component to recover; a fresh probe is required')
        target = new_run()
        for name, files in completed.items():
            destination = target/'profiles'/name
            destination.mkdir(parents=True)
            for path, sha in files.items():
                shutil.copyfile(safe_artifact(source/'profiles'/name, path), destination/path)
                require(file_hash(destination/path) == sha, 'Recovery copy drift')
        reuse = {'parent_run_id': source.name, 'parent_request_sha256': file_hash(source/'request.json'), 'profiles': completed}
        publish(target/'recovery.json', canonical(reuse))
        publish(target/'request.json', canonical({**request_record(), 'selected_gpu': selected_gpu, 'recovery': digest(reuse)}))
        verify_request(target)
        return target


def probe(recover=None):
    request = request_record()
    state = host.resources(OUTPUT_ROOT)
    try:
        gpu = query_gpu()
        state = host.resources(OUTPUT_ROOT)
        reason = start_reason(state, gpu)
    except Exception as exc:
        gpu, reason = None, f'GPU monitoring unavailable: {type(exc).__name__}: {exc}'
    if reason:
        return {'scope': SCOPE, 'status': 'START_REFUSED', 'reason': reason, **state, 'gpu': gpu}
    selected_gpu = gpu_identity(gpu)
    if recover is None:
        run = new_run()
        publish(run/'request.json', canonical({**request, 'selected_gpu': selected_gpu}))
    else:
        run = fork_completed(recover, selected_gpu)
    print('Probe run: '+str(run), file=sys.stderr, flush=True)
    return execute(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe', action='store_true')
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--resume', type=Path, help='Verify a completed batch without fitting or GPU work')
    group.add_argument('--fork', type=Path, help='Explicit authorized recovery attempt; no automatic retry')
    group.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--component', choices=COMPONENTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker, args.component)
        return 0
    result = {**host.resources(OUTPUT_ROOT), 'gpu': query_gpu()} if args.inspect else (
        verify_completed(args.resume.resolve()) if args.resume else probe(args.fork.resolve() if args.fork else None))
    print(json.dumps(result, indent=2))
    return 0 if args.inspect or result['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
