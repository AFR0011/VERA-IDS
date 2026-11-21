"""Approved TRAIN/validation preparation, serialization and reload probe.

python -B -m experiments.c03.engineering.combined_preparation_probe --probe
No classifier fit, selection, test-data reader, GPU work or budget overrides.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time

from . import preparation_probe as guard
from experiments.c03.producer.checkpoints import Checkpoints, ENGINEERING_SCOPE, publish
from experiments.c03.producer.plan import ROOT, OUTPUT_ROOT, build_plan, canonical, digest, file_hash, require, safe_relative


def engineering_plan():
    plan = build_plan()
    plan.update(status='ENGINEERING_PREPARATION_PLAN', evidence_scope=ENGINEERING_SCOPE)
    return plan


def source_record():
    paths = [Path(__file__), Path(guard.__file__), ROOT/'tests/test_c03_combined_preparation_probe.py',
             ROOT/'tests/test_c03_preparation_probe.py']
    return {'plan_sha256': digest(engineering_plan()),
            'probe_sources': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}}


def inventory():
    manifest = json.loads((ROOT/'.release-audit/c03/C03_input_content_manifest.json').read_text())
    result = {}
    for split, count in (('train', 239), ('val', 65)):
        entries = []
        for entry in manifest['data_files']:
            if entry['role'] == split:
                path = ROOT/safe_relative(entry['relative_path'])
                entries.append({'path': path.relative_to(guard.DATA_ROOT).as_posix(), 'sha256': entry['sha256']})
        require(len(entries) == count, 'TRAIN/validation inventory count differs')
        result[split] = sorted(entries, key=lambda x: x['path'])
    return result


def prepare_and_reload(store, sources, emit, *, resume=False):
    from experiments.c03.producer import validation as v
    require(store.plan.get('evidence_scope') == ENGINEERING_SCOPE, 'Engineering scope required')
    if resume:
        receipt = v.verify_prepared(store)
        require(receipt['source_inventory'] == sources.inventory(), 'Resume source inventory drift')
        sources.verify()
        before = {'bindings': receipt['bindings'], 'validation': receipt['validation']}
        emit('prepared_checkpoint_reused')
    else:
        data = v.prepare_engineering(store, sources, emit)
        before = {'bindings': v.input_bindings(data, store.plan), 'validation': v.validation_identity(data)}
        del data
        gc.collect()
    emit('reload')
    loaded = v.load_prepared(store)
    tensors = {stage: {'shape': list(value.shape), 'dtype': str(value.dtype),
                       'csr_bytes': value.data.nbytes+value.indices.nbytes+value.indptr.nbytes}
               for stage, value in {**loaded.tensors, 'validation': loaded.validation}.items()}
    after = {'bindings': v.input_bindings(loaded, store.plan), 'validation': v.validation_identity(loaded)}
    require(before == after, 'Preparation reload identity differs')
    del loaded
    gc.collect()
    emit('reload_verified')
    return {'identities': before, 'tensors': tensors, 'reload_verified': True}


def worker(run):
    # The CLI worker accepts only the exact current approved plan/inventory.
    request = json.loads((run/'request.json').read_text())
    require(request['scope'] == ENGINEERING_SCOPE and request['limits'] == guard.LIMITS,
            'Unapproved combined preparation request')
    require(request['source_record'] == source_record() and request['plan'] == engineering_plan(),
            'Combined preparation source/plan drift')
    require(request['inventory'] == inventory(), 'Combined source inventory drift')
    from experiments.c03.producer.sources import TrainingValidationFiles
    sources = TrainingValidationFiles(guard.DATA_ROOT, request['inventory'])
    if request.get('resume_from'):
        from experiments.c03.producer.recovery import open_store, fork_verified
        source = open_store(Path(request['resume_from'])/'diagnostic')
        require(source.plan == request['plan'], 'Resume plan differs')
        store = fork_verified(source, run, 'diagnostic')
    else:
        store = Checkpoints.create(run, 'diagnostic', request['plan'])
    with (run/'stages.jsonl').open('x', encoding='utf-8') as stream:
        started = time.monotonic()
        def emit(stage):
            stream.write(json.dumps({'stage': stage, 'seconds': time.monotonic()-started})+'\n')
            stream.flush()
        with store.pipeline_lock():
            result = prepare_and_reload(store, sources, emit, resume=bool(request.get('resume_from')))
        emit('final_verification')
        sources.verify()
        require(request['source_record'] == source_record(), 'Source changed during combined preparation')
        files = {p.relative_to(run).as_posix(): file_hash(p) for p in store.path.rglob('*') if p.is_file()}
        result.update(scope=ENGINEERING_SCOPE, status='PREPARATION_COMPLETE', scientific_checkpoint=False,
                      plan_sha256=digest(store.plan), files=files)
        publish(run/'worker-result.json', canonical(result))
        emit('preparation_complete')


def probe(resume_from=None):
    record, plan, entries = source_record(), engineering_plan(), inventory()
    if resume_from is not None:
        from experiments.c03.producer.recovery import open_store
        from experiments.c03.producer.validation import verify_prepared
        source = open_store(Path(resume_from)/'diagnostic')
        require(source.plan == plan, 'Resume code/configuration changed; original run is preserved')
        verify_prepared(source)
    parent = OUTPUT_ROOT
    while not parent.exists():
        parent = parent.parent
    state = guard.resources(parent)
    reason = guard.start_reason(state)
    if reason:
        return {'scope': ENGINEERING_SCOPE, 'status': 'START_REFUSED', 'reason': reason, **state}
    run = OUTPUT_ROOT/('combined-prep-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    require(not run.resolve().is_relative_to(ROOT), 'Probe output redirected into repository')
    run.mkdir(parents=True, exist_ok=False)
    publish(run/'request.json', canonical({'scope': ENGINEERING_SCOPE, 'limits': guard.LIMITS,
        'source_record': record, 'plan': plan, 'inventory': entries,
        'resume_from': None if resume_from is None else str(Path(resume_from).resolve())}))
    result = guard.supervise(run, [sys.executable, '-B', '-u', '-m',
        'experiments.c03.engineering.combined_preparation_probe', '--_worker', str(run)], scope=ENGINEERING_SCOPE)
    return {**result, 'run_directory': str(run)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe', action='store_true')
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--resume', type=Path, help='Fork and verify a completed preparation checkpoint in a fresh attempt')
    group.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker)
        return 0
    result = guard.resources(ROOT) if args.inspect else probe(args.resume)
    print(json.dumps(result, indent=2))
    return 0 if args.inspect or result['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
