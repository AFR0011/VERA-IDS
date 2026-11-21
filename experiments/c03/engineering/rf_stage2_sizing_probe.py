"""Approved eight-tree, one-worker RF sizing on the verified full Stage-2 array.

No validation/test scoring, persisted model, scientific checkpoint promotion or
implicit retry. --resume verifies an already completed result without refitting.
"""
import argparse
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

from . import preparation_probe as guard
from experiments.c03.producer.artifacts import safe_artifact
from experiments.c03.producer.checkpoints import ENGINEERING_SCOPE, publish
from experiments.c03.producer.plan import ROOT, OUTPUT_ROOT, build_plan, canonical, digest, file_hash, require

SUCCESS = ROOT/'experiments/c03/engineering/COMBINED_PREPARATION_SUCCESS.json'
SUCCESS_SHA = 'd92114e6bc8d1c4564300d240364342c1d0f351678f47ef691e0169dbca992d4'
SCOPE = guard.RF_SCOPE
ARRAY = 'diagnostic/prepared/stage2.npz'
COMPONENT = 'stage2_rf_shared'


def load_success():
    require(file_hash(SUCCESS) == SUCCESS_SHA, 'Approved engineering source record drift')
    return json.loads(SUCCESS.read_text())


def node_for_probe(plan):
    node = copy.deepcopy(plan['graph']['components'][COMPONENT])
    require(node['model_family'] == 'rf' and node['stage'] == 'stage2' and node['seed'] == 133
            and node['weight_mode'] == 'balanced', 'RF sizing definition drift')
    node['parameters'].update(n_estimators=8, n_jobs=1)
    return node


def source_record():
    plan = build_plan()
    return {'plan_sha256': digest(plan), 'approved_success_sha256': SUCCESS_SHA,
            'sources': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
                [Path(__file__), ROOT/'tests/test_c03_rf_stage2_sizing_probe.py']}}


def verify_source(source, success, current):
    """Receipt/byte verification; no array decoding or current-source substitution."""
    require(success['scope'] == ENGINEERING_SCOPE and success['status'] == 'VERIFIED_COMBINED_PREPARATION_COMPLETE',
            'Only the completed engineering source is eligible')
    metadata = {}
    for name, sha in success['evidence_hashes'].items():
        path = safe_artifact(source, name)
        require(file_hash(path) == sha, 'Original engineering evidence drift')
        if name in ('request.json', 'worker-result.json', 'supervisor.json', 'complete.json'):
            metadata[name] = json.loads(path.read_text())
    request, result = metadata['request.json'], metadata['worker-result.json']
    completion = metadata['complete.json']
    require(completion['scope'] == ENGINEERING_SCOPE and completion['scientific_checkpoint'] is False
            and metadata['supervisor.json']['status'] == 'COMPLETE' and result['reload_verified'] is True,
            'Source preparation was not complete')
    for name, key in [('request.json','request_sha256'), ('worker-result.json','worker_result_sha256'),
                      ('supervisor.json','supervisor_sha256')]:
        require(completion[key] == file_hash(source/name), 'Source completion receipt drift')
    old = request['plan']
    require(digest(old) == success['source_record']['plan_sha256'] == result['plan_sha256'] and
            request['source_record'] == success['source_record'], 'Original plan identity drift')
    for key in ('environment', 'policy_sha256', 'original_snapshot_sha256', 'sampling', 'expected_support'):
        require(old[key] == current[key], 'Import environment/policy/sample definition differs')
    for name in ('src/ids_eval_framework/_native/protocol_b_grid.py', 'src/ids_eval_framework/src/two_stage_engine.py'):
        require(old['sources'][name] == current['sources'][name], 'Historical helper changed')
    require(old['graph']['components'][COMPONENT] == current['graph']['components'][COMPONENT], 'Historical RF definition differs')
    prep_path = safe_artifact(source, 'diagnostic/prepared/complete.json')
    require(file_hash(prep_path) == result['files']['diagnostic/prepared/complete.json'], 'Preparation receipt drift')
    prep = json.loads(prep_path.read_text())
    expected = success['identities']['bindings']['stage2']
    require(prep['scope'] == ENGINEERING_SCOPE and prep['plan_sha256'] == digest(old)
            and prep['bindings']['stage2'] == expected == result['identities']['bindings']['stage2'],
            'Imported Stage-2 binding drift')
    sha = result['files'][ARRAY]
    require(sha == prep['files']['prepared/stage2.npz'] and file_hash(safe_artifact(source, ARRAY)) == sha,
            'Imported Stage-2 array byte drift')
    return {'scope': SCOPE, 'scientific_checkpoint': False, 'source_run_id': success['run_id'],
        'source_plan_sha256': digest(old), 'current_plan_sha256': digest(current),
        'approved_success_sha256': SUCCESS_SHA, 'array_path': ARRAY, 'array_sha256': sha,
        'expected_binding': expected, 'source_receipt_sha256': file_hash(prep_path),
        'overrides': {'n_estimators': 8, 'n_jobs': 1}, 'seed': 133}


def load_stage2(path, record, columns):
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from experiments.c03.producer.artifacts import read_arrays, tensor_hash, array_hash, identity_hash
    require(file_hash(path) == record['array_sha256'], 'Array changed before load')
    values = read_arrays(path)
    X = csr_matrix((values['data'], values['indices'], values['indptr']), shape=tuple(values['shape']))
    y, weights = values['target'], values['weight_'+COMPONENT]
    ids = pd.DataFrame({k: values[k] for k in ('source_part', 'source_sha256', 'row_offset')})
    expected = record['expected_binding']
    require(X.shape == (expected['rows'], columns) and X.dtype == np.dtype('float64'), 'Stage-2 shape/dtype drift')
    require(tensor_hash(X) == expected['tensor'] and array_hash(y) == expected['labels']
            and identity_hash(ids) == expected['row_ids'] and array_hash(weights) == expected['weights'][COMPONENT],
            'Imported tensor/label/row/weight identity drift')
    require(set(y.tolist()) == set(range(5)) and np.isfinite(weights).all() and (weights > 0).all(),
            'Invalid represented-family support or weights')
    require(file_hash(path) == record['array_sha256'], 'Array changed during load')
    return X, y, weights


def fit_diagnostic(X, y, weights, plan, emit):
    from experiments.c03.producer.validation import historical
    from experiments.c03.producer import models
    from experiments.c03.producer.artifacts import array_hash, tensor_hash
    _, grid, _ = historical()
    require(array_hash(grid.stage2_sample_weights(y, 5, 'balanced')) == array_hash(weights), 'Historical weighting differs')
    before = (tensor_hash(X), array_hash(y), array_hash(weights))
    node = node_for_probe(plan)
    model = models.construct(grid, node)
    params = models.json_parameters(model.get_params())
    require(params['n_estimators'] == 8 and params['n_jobs'] == 1 and params['random_state'] == 133, 'Effective RF settings differ')
    emit('fit_eight_trees')
    started = time.monotonic()
    model.fit(X, y, sample_weight=weights)
    elapsed = time.monotonic()-started
    emit('tree_storage_diagnostics')
    require(len(model.estimators_) == 8, 'Unexpected fitted tree count')
    nodes, storage = [], []
    for estimator in model.estimators_:
        state = estimator.tree_.__getstate__()
        nodes.append(int(estimator.tree_.node_count))
        storage.append(int(state['nodes'].nbytes+state['values'].nbytes))
    require(before == (tensor_hash(X), array_hash(y), array_hash(weights)), 'Fit mutated imported inputs')
    del model
    gc.collect()
    return {'fit_seconds': elapsed, 'tree_node_counts': nodes, 'tree_storage_bytes': storage,
            'total_tree_storage_bytes': sum(storage), 'rows': X.shape[0], 'features': X.shape[1],
            'effective_parameters': params, 'quality_metrics': None, 'model_retained': False}


def worker(run):
    request = json.loads((run/'request.json').read_text())
    require(request['scope'] == SCOPE and request['limits'] == guard.LIMITS and
            request['source_record'] == source_record(), 'RF sizing request/source drift')
    success, plan = load_success(), build_plan()
    source = safe_artifact(OUTPUT_ROOT, success['run_id'])
    with (run/'stages.jsonl').open('x', encoding='utf-8') as stream:
        started = time.monotonic()
        def emit(stage):
            stream.write(json.dumps({'stage': stage, 'seconds': time.monotonic()-started})+'\n'); stream.flush()
        emit('verify_engineering_import')
        imported = verify_source(source, success, plan)
        require(imported['expected_binding']['rows'] == 1201422, 'Approved full-row count differs')
        publish(run/'import.json', canonical(imported))
        emit('load_stage2_training_only')
        X, y, weights = load_stage2(safe_artifact(source, ARRAY), imported, 46)
        diagnostics = fit_diagnostic(X, y, weights, plan, emit)
        emit('final_verification')
        require(verify_source(source, success, plan) == imported and request['source_record'] == source_record(), 'Import/code changed during fit')
        result = {'scope': SCOPE, 'status': 'RF_SIZING_COMPLETE', 'scientific_checkpoint': False,
                  **diagnostics, 'files': {'import.json': file_hash(run/'import.json')}}
        publish(run/'worker-result.json', canonical(result))
        emit('rf_sizing_complete')


def verify_completed(run):
    request = json.loads((run/'request.json').read_text())
    require(request['scope'] == SCOPE and request['source_record'] == source_record(), 'RF result source revision differs')
    receipt = json.loads((run/'complete.json').read_text())
    require(receipt['scope'] == SCOPE and receipt['scientific_checkpoint'] is False, 'Wrong completion scope')
    for name, key in [('request.json','request_sha256'), ('worker-result.json','worker_result_sha256'), ('supervisor.json','supervisor_sha256')]:
        require(file_hash(run/name) == receipt[key], 'RF completion receipt drift')
    result = json.loads((run/'worker-result.json').read_text())
    require(result['status'] == 'RF_SIZING_COMPLETE' and result['scope'] == SCOPE, 'Incomplete RF result')
    for name, sha in result['files'].items():
        require(file_hash(safe_artifact(run, name)) == sha, 'RF diagnostic artifact drift')
    summary = json.loads((run/'supervisor.json').read_text())
    require(summary['status'] == 'COMPLETE', 'RF attempt did not complete')
    return {**summary, 'run_directory': str(run), 'reused_completed_result': True}


def probe():
    record = source_record()
    load_success()  # Metadata only; no imported arrays opened before the gate.
    state = guard.resources(OUTPUT_ROOT)
    reason = guard.start_reason(state)
    if reason:
        return {'scope': SCOPE, 'status': 'START_REFUSED', 'reason': reason, **state}
    run = OUTPUT_ROOT/('rf-stage2-sizing-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    require(not run.resolve().is_relative_to(ROOT), 'Probe output redirected into repository')
    run.mkdir(parents=True, exist_ok=False)
    publish(run/'request.json', canonical({'scope': SCOPE, 'limits': guard.LIMITS, 'source_record': record}))
    result = guard.supervise(run, [sys.executable, '-B', '-u', '-m',
        'experiments.c03.engineering.rf_stage2_sizing_probe', '--_worker', str(run)], scope=SCOPE)
    return {**result, 'run_directory': str(run)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe', action='store_true')
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--resume', type=Path, help='Verify a completed result without refitting; incomplete fits require a fresh attempt')
    group.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker); return 0
    result = guard.resources(OUTPUT_ROOT) if args.inspect else (verify_completed(args.resume.resolve()) if args.resume else probe())
    print(json.dumps(result, indent=2))
    return 0 if args.inspect or result['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
