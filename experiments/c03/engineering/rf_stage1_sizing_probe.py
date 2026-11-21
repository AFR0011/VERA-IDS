"""Two approved Stage-1 RF diagnostics, sequential workers, one 900-second budget.

--resume verifies a completed batch. --fork explicitly starts a new attempt,
reusing verified completed profile diagnostics; a new attempt needs authorization.
No scoring, retained models, automatic retry or mid-estimator continuation.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import preparation_probe as guard
from . import rf_stage2_sizing_probe as base
from experiments.c03.producer.artifacts import safe_artifact
from experiments.c03.producer.checkpoints import ENGINEERING_SCOPE, publish
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer.plan import ROOT, OUTPUT_ROOT, build_plan, canonical, digest, file_hash, require

SCOPE = 'ENGINEERING_RF_STAGE1_SIZING_ONLY'
COMPONENTS = ('stage1_rf_class_weight_balanced', 'stage1_rf_inv_family_clipped')
DEFINITION = {'components': list(COMPONENTS), 'n_estimators': 8, 'n_jobs': 2, 'seed': 123}
ARRAY = 'diagnostic/prepared/stage1.npz'
PROFILE_FILES = ('request.json', 'import.json', 'worker-result.json', 'supervisor.json',
                 'stages.jsonl', 'monitor.jsonl', 'worker.log', 'complete.json')


def source_record():
    return {'base': base.source_record(), 'sources': {
        p.relative_to(ROOT).as_posix(): file_hash(p) for p in
        [Path(__file__), ROOT/'tests/test_c03_rf_stage1_sizing_probe.py']}}


def request_record():
    return {'scope': SCOPE, 'definition': DEFINITION, 'limits': guard.LIMITS,
            'source_record': source_record()}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_import(source, success, current):
    require(success['scope'] == ENGINEERING_SCOPE
            and success['status'] == 'VERIFIED_COMBINED_PREPARATION_COMPLETE', 'Invalid engineering source')
    for name, sha in success['evidence_hashes'].items():
        require(file_hash(safe_artifact(source, name)) == sha, 'Original evidence drift')
    request, result, completion = [read(source/name) for name in
                                  ('request.json', 'worker-result.json', 'complete.json')]
    require(completion['scope'] == ENGINEERING_SCOPE and completion['scientific_checkpoint'] is False
            and read(source/'supervisor.json')['status'] == 'COMPLETE' and result['reload_verified'] is True,
            'Incomplete source preparation')
    for name, key in [('request.json','request_sha256'), ('worker-result.json','worker_result_sha256'),
                      ('supervisor.json','supervisor_sha256')]:
        require(file_hash(source/name) == completion[key], 'Source completion receipt drift')
    old = request['plan']
    require(digest(old) == success['source_record']['plan_sha256'] == result['plan_sha256']
            and request['source_record'] == success['source_record'], 'Original plan drift')
    for key in ('environment', 'policy_sha256', 'original_snapshot_sha256', 'sampling', 'expected_support'):
        require(old[key] == current[key], 'Import environment/policy/sample definition differs')
    for name in ('src/ids_eval_framework/_native/protocol_b_grid.py', 'src/ids_eval_framework/src/two_stage_engine.py'):
        require(old['sources'][name] == current['sources'][name], 'Historical helper changed')
    for name in COMPONENTS:
        require(old['graph']['components'][name] == current['graph']['components'][name], 'Stage-1 RF definition differs')
    prep_path = safe_artifact(source, 'diagnostic/prepared/complete.json')
    require(file_hash(prep_path) == result['files']['diagnostic/prepared/complete.json'], 'Preparation receipt drift')
    prep, expected = read(prep_path), success['identities']['bindings']['stage1']
    require(prep['scope'] == ENGINEERING_SCOPE and prep['plan_sha256'] == digest(old)
            and prep['bindings']['stage1'] == expected == result['identities']['bindings']['stage1'],
            'Stage-1 binding drift')
    sha = result['files'][ARRAY]
    require(sha == prep['files']['prepared/stage1.npz'] and file_hash(safe_artifact(source, ARRAY)) == sha,
            'Stage-1 array byte drift')
    return {'scope': SCOPE, 'scientific_checkpoint': False, 'source_run_id': success['run_id'],
        'source_plan_sha256': digest(old), 'current_plan_sha256': digest(current),
        'approved_success_sha256': base.SUCCESS_SHA, 'array_path': ARRAY, 'array_sha256': sha,
        'expected_binding': expected, 'source_receipt_sha256': file_hash(prep_path), 'definition': DEFINITION}


def load_stage1(path, imported, name, columns=46):
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from experiments.c03.producer.artifacts import read_arrays, tensor_hash, array_hash, identity_hash
    require(name in COMPONENTS, 'Unknown profile')
    require(file_hash(path) == imported['array_sha256'], 'Array changed before load')
    values = read_arrays(path)
    X = csr_matrix((values['data'], values['indices'], values['indptr']), shape=tuple(values['shape']))
    y, weight = values['target'], values.get('weight_'+name)
    ids = pd.DataFrame({key: values[key] for key in ('source_part', 'source_sha256', 'row_offset')})
    expected = imported['expected_binding']
    require(X.shape == (expected['rows'], columns) and X.dtype == np.dtype('float64'), 'Stage-1 shape/dtype drift')
    require(tensor_hash(X) == expected['tensor'] and array_hash(y) == expected['labels']
            and identity_hash(ids) == expected['row_ids'], 'Stage-1 input identity drift')
    require(set(np.unique(y)) == {0, 1}, 'Binary support differs')
    if name == COMPONENTS[0]:
        require(weight is None and expected['weights'][name] == digest(None), 'Native class weighting requires no sample weights')
    else:
        require(weight is not None and weight.shape == y.shape and np.isfinite(weight).all()
                and (weight > 0).all() and array_hash(weight) == expected['weights'][name], 'Family weight identity drift')
    require(file_hash(path) == imported['array_sha256'], 'Array changed during load')
    return X, y, weight


def fit_diagnostic(X, y, weight, plan, name, emit):
    import copy
    from experiments.c03.producer.validation import historical
    from experiments.c03.producer import models
    from experiments.c03.producer.artifacts import tensor_hash, array_hash
    require(name in COMPONENTS, 'Unknown profile')
    _, grid, _ = historical()
    node = copy.deepcopy(plan['graph']['components'][name])
    require(node['stage'] == 'stage1' and node['model_family'] == 'rf' and node['seed'] == 123,
            'Stage-1 definition differs')
    historical_params = models.json_parameters(models.construct(grid, node).get_params())
    node['parameters'].update(n_estimators=8, n_jobs=2)
    model = models.construct(grid, node)
    params = models.json_parameters(model.get_params())
    require(params == {**historical_params, 'n_estimators': 8, 'n_jobs': 2}, 'Unapproved model parameter change')
    require((name == COMPONENTS[0] and weight is None and params['class_weight'] == 'balanced_subsample')
            or (name == COMPONENTS[1] and weight is not None and params['class_weight'] is None), 'Wrong weighting path')
    def identities():
        return tensor_hash(X), array_hash(y), digest(None) if weight is None else array_hash(weight)
    before = identities()
    emit('fit_eight_trees_two_workers')
    started = time.monotonic()
    model.fit(X, y, **({} if weight is None else {'sample_weight': weight}))
    elapsed = time.monotonic()-started
    emit('tree_storage_diagnostics')
    require(len(model.estimators_) == 8, 'Unexpected tree count')
    nodes, storage = [], []
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        nodes.append(int(estimator.tree_.node_count))
        storage.append(int(state['nodes'].nbytes+state['values'].nbytes))
    require(before == identities(), 'Fit mutated inputs')
    del state, estimator, model
    gc.collect()
    return {'component': name, 'rows': X.shape[0], 'features': X.shape[1], 'fit_seconds': elapsed,
        'effective_parameters': params, 'input_identities': list(before), 'tree_node_counts': nodes,
        'tree_storage_bytes': storage, 'total_tree_storage_bytes': sum(storage),
        'quality_metrics': None, 'model_retained': False}


def worker(run, name):
    require(name in COMPONENTS, 'Unknown profile')
    profile = safe_artifact(run, 'profiles/'+name)
    with file_lock(profile/'worker.guard'):
        require(read(profile/'request.json') == {**request_record(), 'component': name}, 'Profile request/source drift')
        success, plan = base.load_success(), build_plan()
        source = safe_artifact(OUTPUT_ROOT, success['run_id'])
        with (profile/'stages.jsonl').open('x', encoding='utf-8') as stream:
            def emit(stage):
                stream.write(json.dumps({'stage': name+'_'+stage})+'\n'); stream.flush()
            emit('verify_engineering_import')
            imported = verify_import(source, success, plan)
            require(imported['expected_binding']['rows'] == 1270025, 'Approved Stage-1 row count differs')
            publish(profile/'import.json', canonical(imported))
            emit('load_stage1_training_only')
            X, y, weight = load_stage1(safe_artifact(source, ARRAY), imported, name)
            diagnostics = fit_diagnostic(X, y, weight, plan, name, emit)
            emit('final_verification')
            require(verify_import(source, success, plan) == imported
                    and read(profile/'request.json') == {**request_record(), 'component': name}, 'Input/code changed during fit')
            publish(profile/'worker-result.json', canonical({'scope': SCOPE, 'status': 'PROFILE_COMPLETE',
                'scientific_checkpoint': False, **diagnostics, 'files': {'import.json': file_hash(profile/'import.json')}}))
            emit('complete')


def verify_profile(profile, name):
    require(read(profile/'request.json') == {**request_record(), 'component': name}, 'Profile source/definition differs')
    receipt = read(profile/'complete.json')
    require(receipt['scope'] == SCOPE and receipt['component'] == name
            and receipt['scientific_checkpoint'] is False, 'Profile completion scope differs')
    require(set(receipt['files']) == set(PROFILE_FILES)-{'complete.json'}, 'Incomplete profile receipt')
    for path, sha in receipt['files'].items():
        require(file_hash(safe_artifact(profile, path)) == sha, 'Profile receipt byte drift')
    result, summary = read(profile/'worker-result.json'), read(profile/'supervisor.json')
    require(result['scope'] == SCOPE and result['status'] == 'PROFILE_COMPLETE' and result['component'] == name
            and result['scientific_checkpoint'] is False and result['model_retained'] is False
            and result['quality_metrics'] is None and summary['status'] == 'COMPLETE', 'Invalid profile completion')
    require(result['files'] == {'import.json': file_hash(profile/'import.json')}, 'Profile import drift')
    return result


def command(run, name):
    return [sys.executable, '-B', '-u', '-m', 'experiments.c03.engineering.rf_stage1_sizing_probe',
            '--_worker', str(run), '--component', name]


def run_profile(run, name, batch_started):
    profile = run/'profiles'/name
    profile.mkdir(parents=True, exist_ok=False)
    publish(profile/'request.json', canonical({**request_record(), 'component': name}))
    initial = guard.resources(run)
    reason = guard.stop_reason(initial, 0, guard.artifact_size(run), time.monotonic()-batch_started) or guard.start_reason(initial)
    started = time.monotonic()
    progress = guard.Progress(profile)
    process = tree = None
    peak, minimum, maximum = 0, initial['available'], guard.artifact_size(run)
    try:
        if not reason:
            with (profile/'worker.log').open('x', encoding='utf-8') as log, (profile/'monitor.jsonl').open('x', encoding='utf-8') as trace:
                process = subprocess.Popen(command(run, name), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                tree = guard.ProcessTree(process.pid)
                while True:
                    state = guard.resources(run)
                    rss, size, elapsed = tree.rss(), guard.artifact_size(run), time.monotonic()-batch_started
                    peak, minimum, maximum = max(peak, rss), min(minimum, state['available']), max(maximum, size)
                    trace.write(json.dumps({'seconds': elapsed, 'rss': rss, 'artifact_bytes': size, **state})+'\n'); trace.flush()
                    progress.update(elapsed, rss, state, size)
                    reason = guard.stop_reason(state, rss, size, elapsed)
                    if reason:
                        break
                    if process.poll() is not None:
                        require(not tree.live(), 'Launcher exited with live descendants')
                        require(process.returncode == 0, f'Worker exit {process.returncode}')
                        result = read(profile/'worker-result.json')
                        require(result['scope'] == SCOPE and result['status'] == 'PROFILE_COMPLETE'
                                and result['component'] == name and result['scientific_checkpoint'] is False
                                and result['model_retained'] is False and result['quality_metrics'] is None,
                                'Invalid worker result')
                        require(result['files'] == {'import.json': file_hash(profile/'import.json')}, 'Import receipt drift')
                        break
                    time.sleep(guard.LIMITS['interval'])
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if tree is not None:
                tree.stop()
            elif process is not None and process.poll() is None:
                guard.ProcessTree(process.pid).stop()
            if process is not None:
                process.wait(timeout=10)
        except Exception as exc:
            reason = f'Cleanup failed: {type(exc).__name__}: {exc}; earlier={reason}'
            if process is not None and process.poll() is None:
                process.kill()
    reason = reason or guard.stop_reason(guard.resources(run), 0, guard.artifact_size(run), time.monotonic()-batch_started)
    summary = {'scope': SCOPE, 'component': name, 'status': 'STOPPED' if reason else 'COMPLETE', 'reason': reason,
        'peak_tree_rss': peak, 'minimum_available': minimum, 'max_observed_artifact_bytes': maximum,
        'wall_seconds': time.monotonic()-started, 'batch_elapsed_seconds': time.monotonic()-batch_started,
        'worker_exit': None if process is None else process.returncode, 'limits': guard.LIMITS}
    publish(profile/'supervisor.json', canonical(summary))
    if not reason:
        publish(profile/'complete.json', canonical({'scope': SCOPE, 'component': name, 'scientific_checkpoint': False,
            'files': {path: file_hash(profile/path) for path in PROFILE_FILES if path != 'complete.json'}}))
        verify_profile(profile, name)
    progress.finish({**summary, 'wall_seconds': summary['batch_elapsed_seconds']})
    return summary


def verify_request(run):
    request = read(run/'request.json')
    require({k: v for k, v in request.items() if k != 'recovery'} == request_record(), 'Batch source/definition differs')
    if 'recovery' in request:
        reuse = read(run/'recovery.json')
        require(digest(reuse) == request['recovery'], 'Incomplete recovery commit')
        for name, files in reuse['profiles'].items():
            require(name in COMPONENTS, 'Unknown recovered profile')
            for path, sha in files.items():
                require(file_hash(safe_artifact(run/'profiles'/name, path)) == sha, 'Recovered profile drift')
            verify_profile(run/'profiles'/name, name)


def execute(run):
    with file_lock(run/'batch.guard'):
        started = time.monotonic()
        verify_request(run)
        profiles, reused = {}, []
        for name in COMPONENTS:
            profile = run/'profiles'/name
            if (profile/'complete.json').exists():
                with file_lock(profile/'worker.guard'):
                    verify_profile(profile, name)
                reused.append(name)
                print('Reusing completed diagnostic: '+name, file=sys.stderr, flush=True)
                continue
            require(not profile.exists(), 'Incomplete profile requires an explicit fresh recovery fork')
            profiles[name] = run_profile(run, name, started)
            if profiles[name]['status'] != 'COMPLETE':
                break
        completed = [name for name in COMPONENTS if (run/'profiles'/name/'complete.json').exists()]
        reason = next((v['reason'] for v in profiles.values() if v['status'] != 'COMPLETE'), None)
        reason = reason or guard.stop_reason(guard.resources(run), 0, guard.artifact_size(run), time.monotonic()-started)
        summary = {'scope': SCOPE, 'status': 'COMPLETE' if len(completed) == 2 and not reason else 'STOPPED',
            'reason': reason, 'wall_seconds': time.monotonic()-started, 'limits': guard.LIMITS,
            'profiles': profiles, 'completed_profiles': completed, 'reused_profiles': reused}
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
    require(set(receipt['profiles']) == set(COMPONENTS), 'Missing completed profiles')
    for name, sha in receipt['profiles'].items():
        profile = safe_artifact(run, 'profiles/'+name)
        require(file_hash(profile/'complete.json') == sha, 'Profile completion receipt drift')
        verify_profile(profile, name)
    return {**summary, 'run_directory': str(run), 'reused_completed_result': True}


def new_run():
    run = OUTPUT_ROOT/('rf-stage1-sizing-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    require(not run.resolve().is_relative_to(ROOT), 'Output redirected into repository')
    run.mkdir(parents=True, exist_ok=False)
    return run


def fork_completed(source):
    # Both parent and possible orphaned workers must release their locks first.
    with ExitStack() as stack:
        stack.enter_context(file_lock(source/'batch.guard'))
        verify_request(source)
        for name in COMPONENTS:
            profile = safe_artifact(source, 'profiles/'+name)
            if profile.exists():
                stack.enter_context(file_lock(profile/'worker.guard'))
        completed = {}
        for name in COMPONENTS:
            profile = safe_artifact(source, 'profiles/'+name)
            if (profile/'complete.json').exists():
                verify_profile(profile, name)
                completed[name] = {path: file_hash(profile/path) for path in PROFILE_FILES}
        require(completed, 'No completed profile to recover; a fresh probe is required')
        target = new_run()
        for name, files in completed.items():
            destination = target/'profiles'/name
            destination.mkdir(parents=True)
            for path, sha in files.items():
                shutil.copyfile(safe_artifact(source/'profiles'/name, path), destination/path)
                require(file_hash(destination/path) == sha, 'Recovery copy drift')
        reuse = {'parent_run_id': source.name, 'parent_request_sha256': file_hash(source/'request.json'), 'profiles': completed}
        publish(target/'recovery.json', canonical(reuse))
        publish(target/'request.json', canonical({**request_record(), 'recovery': digest(reuse)}))
        verify_request(target)
        return target


def probe(recover=None):
    request = request_record()
    base.load_success()  # Metadata only before the start gate.
    state = guard.resources(OUTPUT_ROOT)
    reason = guard.start_reason(state)
    if reason:
        return {'scope': SCOPE, 'status': 'START_REFUSED', 'reason': reason, **state}
    if recover is None:
        run = new_run()
        publish(run/'request.json', canonical(request))
    else:
        run = fork_completed(recover)
    print('Probe run: '+str(run), file=sys.stderr, flush=True)
    return execute(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe', action='store_true')
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--resume', type=Path, help='Verify a completed batch without fitting')
    group.add_argument('--fork', type=Path, help='Explicit fresh attempt reusing completed profiles; requires authorization')
    group.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--component', choices=COMPONENTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker, args.component)
        return 0
    result = guard.resources(OUTPUT_ROOT) if args.inspect else (verify_completed(args.resume.resolve())
        if args.resume else probe(args.fork.resolve() if args.fork else None))
    print(json.dumps(result, indent=2))
    return 0 if args.inspect or result['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
