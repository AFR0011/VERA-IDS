"""Approved eight-tree RF comparison with two workers and fixed resource limits.

Reuses the verified Stage-2 engineering array; no scoring or retained model.
Completed results can be verified without refitting; incomplete fits stay intact.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

from . import rf_stage2_sizing_probe as base
from . import preparation_probe as guard
from experiments.c03.producer.artifacts import safe_artifact
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.plan import ROOT, OUTPUT_ROOT, build_plan, canonical, file_hash, require

SCOPE = base.SCOPE
BASELINE = ROOT/'experiments/c03/engineering/RF_STAGE2_SIZING_SUCCESS.json'
BASELINE_SHA = '601e7d8f38e63f71c40aa13979d70da5b9045b968b752f7c8569eb1fadff01e6'
DEFINITION = {'n_estimators': 8, 'n_jobs': 2, 'seed': 133}


def load_baseline():
    require(file_hash(BASELINE) == BASELINE_SHA, 'Approved one-worker baseline drift')
    record = json.loads(BASELINE.read_text())
    require(record['status'] == 'VERIFIED_RF_STAGE2_SIZING_COMPLETE'
            and record['source_record'] == base.source_record(), 'Baseline source revision differs')
    return record


def source_record():
    return {'base': base.source_record(), 'baseline_sha256': BASELINE_SHA,
            'sources': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in
                [Path(__file__), ROOT/'tests/test_c03_rf_two_worker_sizing_probe.py']}}


def verify_import(source, success, plan, baseline):
    original = base.verify_source(source, success, plan)
    require(original == baseline['import_record'], 'Comparison input differs from one-worker baseline')
    return {**original, 'overrides': {'n_estimators': 8, 'n_jobs': 2},
            'baseline_sha256': BASELINE_SHA}


def fit_diagnostic(X, y, weights, plan, baseline, emit):
    from experiments.c03.producer.validation import historical
    from experiments.c03.producer import models
    from experiments.c03.producer.artifacts import array_hash, tensor_hash
    _, grid, _ = historical()
    require(array_hash(grid.stage2_sample_weights(y, 5, 'balanced')) == array_hash(weights), 'Historical weighting differs')
    before = (tensor_hash(X), array_hash(y), array_hash(weights))
    node = base.node_for_probe(plan)
    node['parameters']['n_jobs'] = 2
    model = models.construct(grid, node)
    params = models.json_parameters(model.get_params())
    expected = {**baseline['diagnostics']['effective_parameters'], 'n_jobs': 2}
    require(params == expected and params['n_estimators'] == 8 and params['random_state'] == 133,
            'Comparison changed parameters beyond worker count')
    emit('fit_eight_trees_two_workers')
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
    del state, estimator, model
    gc.collect()
    return {'fit_seconds': elapsed, 'tree_node_counts': nodes, 'tree_storage_bytes': storage,
            'total_tree_storage_bytes': sum(storage), 'rows': X.shape[0], 'features': X.shape[1],
            'effective_parameters': params, 'quality_metrics': None, 'model_retained': False,
            'comparison': {'baseline_sha256': BASELINE_SHA,
                'fit_speedup_ratio': baseline['diagnostics']['fit_seconds']/elapsed,
                'same_tree_node_counts': nodes == baseline['diagnostics']['tree_node_counts'],
                'same_tree_storage_bytes': storage == baseline['diagnostics']['tree_storage_bytes'],
                'limitation': 'Single-run timing comparison; tree sizes do not establish model identity'}}


def worker(run):
    request = json.loads((run/'request.json').read_text())
    require(request['scope'] == SCOPE and request['limits'] == guard.LIMITS
            and request['definition'] == DEFINITION and request['source_record'] == source_record(),
            'Two-worker sizing request/source drift')
    baseline, success, plan = load_baseline(), base.load_success(), build_plan()
    source = safe_artifact(OUTPUT_ROOT, success['run_id'])
    with (run/'stages.jsonl').open('x', encoding='utf-8') as stream:
        started = time.monotonic()
        def emit(stage):
            stream.write(json.dumps({'stage': stage, 'seconds': time.monotonic()-started})+'\n')
            stream.flush()
        emit('verify_engineering_import')
        imported = verify_import(source, success, plan, baseline)
        require(imported['expected_binding']['rows'] == 1201422, 'Approved full-row count differs')
        publish(run/'import.json', canonical(imported))
        emit('load_stage2_training_only')
        X, y, weights = base.load_stage2(safe_artifact(source, base.ARRAY), imported, 46)
        diagnostics = fit_diagnostic(X, y, weights, plan, baseline, emit)
        emit('final_verification')
        require(verify_import(source, success, plan, load_baseline()) == imported
                and request['source_record'] == source_record(), 'Import/code changed during fit')
        publish(run/'worker-result.json', canonical({'scope': SCOPE, 'status': 'RF_SIZING_COMPLETE',
            'scientific_checkpoint': False, 'definition': DEFINITION, **diagnostics,
            'files': {'import.json': file_hash(run/'import.json')}}))
        emit('rf_sizing_complete')


def verify_completed(run):
    request = json.loads((run/'request.json').read_text())
    require(request['scope'] == SCOPE and request['limits'] == guard.LIMITS
            and request['definition'] == DEFINITION and request['source_record'] == source_record(),
            'Two-worker result source/definition differs')
    receipt = json.loads((run/'complete.json').read_text())
    require(receipt['scope'] == SCOPE and receipt['scientific_checkpoint'] is False, 'Wrong completion scope')
    for name, key in [('request.json','request_sha256'), ('worker-result.json','worker_result_sha256'),
                      ('supervisor.json','supervisor_sha256')]:
        require(file_hash(run/name) == receipt[key], 'RF completion receipt drift')
    result = json.loads((run/'worker-result.json').read_text())
    require(result['status'] == 'RF_SIZING_COMPLETE' and result['scope'] == SCOPE
            and result['definition'] == DEFINITION and result['scientific_checkpoint'] is False,
            'Incomplete two-worker RF result')
    for name, sha in result['files'].items():
        require(file_hash(safe_artifact(run, name)) == sha, 'RF diagnostic artifact drift')
    summary = json.loads((run/'supervisor.json').read_text())
    require(summary['status'] == 'COMPLETE', 'RF attempt did not complete')
    return {**summary, 'run_directory': str(run), 'reused_completed_result': True}


def probe():
    record = source_record()
    load_baseline()
    base.load_success()  # Metadata only before the resource gate.
    state = guard.resources(OUTPUT_ROOT)
    reason = guard.start_reason(state)
    if reason:
        return {'scope': SCOPE, 'definition': DEFINITION, 'status': 'START_REFUSED', 'reason': reason, **state}
    run = OUTPUT_ROOT/('rf-two-worker-sizing-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    require(not run.resolve().is_relative_to(ROOT), 'Probe output redirected into repository')
    run.mkdir(parents=True, exist_ok=False)
    publish(run/'request.json', canonical({'scope': SCOPE, 'limits': guard.LIMITS,
        'definition': DEFINITION, 'source_record': record}))
    result = guard.supervise(run, [sys.executable, '-B', '-u', '-m',
        'experiments.c03.engineering.rf_two_worker_sizing_probe', '--_worker', str(run)], scope=SCOPE)
    return {**result, 'run_directory': str(run)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe', action='store_true')
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--resume', type=Path, help='Verify a completed result without refitting')
    group.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker)
        return 0
    result = guard.resources(OUTPUT_ROOT) if args.inspect else (verify_completed(args.resume.resolve()) if args.resume else probe())
    print(json.dumps(result, indent=2))
    return 0 if args.inspect or result['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
