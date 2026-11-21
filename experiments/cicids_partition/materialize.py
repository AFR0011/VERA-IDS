"""Materialize only the approved seed-123 contiguous-unit assignment; no models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pandas as pd
import psutil

from . import seeded_assignments as assignments
from experiments.c03.producer.plan import ROOT, canonical, digest, file_hash, require, output_location
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.artifacts import safe_artifact, generated_file, write_arrays
from experiments.c03.producer.locking import file_lock
from experiments.c03.engineering import preparation_probe as host, resource_policy as policy
from experiments.c03.engineering.lifecycle_probe import Budget
from experiments.literature.efc_input_probe import watch

SCOPE = 'CICIDS_CONTIGUOUS_SEED123_MATERIALIZATION_ONLY'
LIMITS = dict(assignments.LIMITS)
RAW = ROOT / 'Datasets/CICIDS 2017'
REFERENCE = assignments.METADATA
OUTPUT = assignments.replay.OUTPUT
RECEIPTS = ('CONTIGUOUS_ASSIGNMENT_SUCCESS.json', 'PREDICTOR_RECONCILIATION_SUCCESS.json')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def definition():
    from ids_eval_framework._native import prepare_datasets as native
    prior, reconciled = [read(Path(__file__).parent / name) for name in RECEIPTS]
    require(prior['status'] == reconciled['status'] == 'COMPLETE' and reconciled['all_values_exact'], 'Missing prior evidence')
    metadata = assignments.load_metadata()
    require(metadata == prior['definition']['metadata'], 'Assignment metadata drift')
    chosen = next(v for v in prior['results'] if v['seed'] == 123)
    require(chosen == assignments.evaluate_seed(prior['definition'], 123), 'Seed-123 assignment drift')
    require(chosen['row_totals'] == {'train': 1223347, 'val': 432067, 'test': 444465}, 'Unexpected split populations')
    require(file_hash(Path(native.__file__)) == reconciled['native_source_sha256'], 'Native mapper drift')
    require(native.CFG['datasets']['CICIDS2017']['drop_families'] == ['Other'], 'Native family policy drift')
    by_id = {u['unit_id']: u for u in reconciled['units']}
    require(len(by_id) == 20 and set(by_id) == {u['id'] for u in metadata['units']}, 'Incomplete reconciled units')
    items = []
    for split, indices in chosen['assignment'].items():
        for part, index in enumerate(indices):
            unit = metadata['units'][index]; proof = by_id[unit['id']]
            require(proof['source_unit'] == unit and proof['numeric_and_text_values_exact'], 'Unit reconciliation drift')
            items.append({'unit': unit, 'unit_index': index, 'split': split,
                'output': f'{split}/part_{part:05d}.csv.gz', 'identity': f'identities/unit-{index:02d}.npz',
                'reference': proof['historical_part'], 'reference_sha256': proof['prepared_part_sha256'],
                'raw_position_sha256': proof['ordered_raw_position_sha256'],
                'family_support': metadata['family_counts'][index]})
    sources = set(prior['definition']['sources'])
    sources.update(('experiments/cicids_partition/materialize.py', 'tests/test_cicids_materialize.py',
                    'experiments/literature/efc_input_probe.py', 'experiments/c03/engineering/lifecycle_probe.py'))
    sources.add(Path(native.__file__).relative_to(ROOT).as_posix())
    return {'scope': SCOPE, 'partition_seed': 123, 'assignment': chosen, 'items': items,
        'raw_hashes': reconciled['raw_source_hashes'], 'metadata_hashes': reconciled['metadata_hashes'],
        'receipts': {n: file_hash(Path(__file__).parent / n) for n in RECEIPTS},
        'sources': {n: file_hash(ROOT / n) for n in sorted(sources)}, 'limits': LIMITS,
        'numpy': np.__version__, 'pandas': pd.__version__, 'model_fit': False, 'scoring': False,
        'interpretation': 'Dependent contiguous-source partitions; no independent attack-execution claim'}


def request(run):
    req = read(safe_artifact(run, 'request.json'))
    require(req['definition'] == definition() and req['deadline_epoch'] == req['started_epoch'] + 900, 'Definition/deadline drift')
    return req


def guard(req):
    require(req['started_epoch'] <= time.time() < req['deadline_epoch'], 'Original materialization deadline exhausted')


def copy_verified(source, destination, sha, guard=lambda: None, hook=lambda event: None):
    """Independent copy with atomic publication; interrupted temporary bytes are discarded."""
    require(file_hash(source) == sha, 'Reference source drift')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        require(not destination.is_symlink() and file_hash(destination) == sha, 'Copied part drift')
    else:
        with generated_file(destination) as temporary:
            with source.open('rb') as incoming, temporary.open('wb') as outgoing:
                for block in iter(lambda: incoming.read(1024 * 1024), b''):
                    guard(); outgoing.write(block); hook('copy_block')
            require(file_hash(temporary) == sha, 'Copy checksum differs')
        hook('copy_published')
    require(not os.path.samefile(source, destination) and file_hash(source) == sha, 'Copy aliases/changed reference')


def positions_for(unit, raw_root=RAW):
    from ids_eval_framework._native import prepare_datasets as native
    frame = native.read_csv_slice(str(raw_root / unit['source_file']), native.CFG['datasets']['CICIDS2017'],
        start_row=unit['row_start'], nrows=unit['row_end']-unit['row_start'],
        usecols=lambda name: str(name).strip().lower() == 'label')
    frame = native.canonicalize_columns(frame)
    require(list(frame.columns) == ['label'] and len(frame) == unit['row_end']-unit['row_start'], 'Raw label slice differs')
    family = frame['label'].astype(str).str.strip().map(native.mapper_cicids2017).astype(str)
    keep = family != 'Other'
    positions = np.arange(unit['row_start'], unit['row_end'], dtype='<i8')[keep.to_numpy()]
    support = [int((family[keep] == name).sum()) for name in assignments.FAMILIES]
    require(sum(support) == len(positions) == unit['kept_rows'], 'Raw family population differs')
    return positions, support


def unit_checkpoint(run, item, positions, support, *, reference=REFERENCE, guard=lambda: None, hook=lambda event: None):
    guard(); unit = item['unit']
    require(positions.dtype == np.dtype('<i8') and positions.shape == (unit['kept_rows'],)
            and np.all(np.diff(positions) > 0) and positions[0] >= unit['row_start'] and positions[-1] < unit['row_end'],
            'Invalid source row positions')
    require(hashlib.sha256(positions.tobytes()).hexdigest() == item['raw_position_sha256']
            and support == item['family_support'], 'Reconciled raw identities/support differ')
    source = safe_artifact(reference, item['reference']); output = safe_artifact(run, item['output'])
    copy_verified(source, output, item['reference_sha256'], guard, hook)
    saved = pd.read_csv(output, usecols=['y_stage2_family'], low_memory=True)
    require(len(saved) == len(positions) and [int((saved['y_stage2_family'] == n).sum()) for n in assignments.FAMILIES] == support,
            'Copied part support differs')
    identity = safe_artifact(run, item['identity']); identity.parent.mkdir(exist_ok=True)
    if identity.exists():
        with np.load(identity, allow_pickle=False) as arrays:
            require(np.array_equal(arrays['row_offset'], positions), 'Raw identity checkpoint drift')
    else:
        write_arrays(identity, row_offset=positions)
    hook('identity_published'); guard()
    record = {'request_sha256': file_hash(run/'request.json'), 'item': item, 'rows': len(positions),
              'output_sha256': file_hash(output), 'identity_sha256': file_hash(identity)}
    path = run / f'units/unit-{item["unit_index"]:02d}.json'; path.parent.mkdir(exist_ok=True)
    if path.exists(): require(read(path) == record, 'Unit checkpoint drift')
    else: publish(path, canonical(record))
    hook('unit_published')
    return record


def verify_materialized(run, req):
    plan = req['definition']; rows = {s: 0 for s in ('train', 'val', 'test')}
    support = {s: np.zeros(7, dtype=np.int64) for s in rows}; records = []
    for item in plan['items']:
        record = read(safe_artifact(run, f'units/unit-{item["unit_index"]:02d}.json'))
        require(record['request_sha256'] == file_hash(run/'request.json') and record['item'] == item
                and record['rows'] == item['unit']['kept_rows'], 'Completed unit binding drift')
        for name, key in ((item['output'], 'output_sha256'), (item['identity'], 'identity_sha256')):
            require(file_hash(safe_artifact(run, name)) == record[key], 'Materialized file drift')
        require(record['output_sha256'] == item['reference_sha256'], 'Copied values differ from reconciliation')
        with np.load(safe_artifact(run,item['identity']), allow_pickle=False) as arrays:
            positions = arrays['row_offset']
        require(hashlib.sha256(positions.tobytes()).hexdigest() == item['raw_position_sha256'], 'Completed source identity drift')
        rows[item['split']] += record['rows']; support[item['split']] += item['family_support']; records.append(record)
    require(rows == plan['assignment']['row_totals'] and {s: v.tolist() for s,v in support.items()} == plan['assignment']['support'],
            'Split support/population differs')
    return {'scope': SCOPE, 'status': 'COMPLETE', 'request_sha256': file_hash(run/'request.json'),
            'partition_seed': 123, 'assignment_sha256': plan['assignment']['assignment_sha256'],
            'rows': rows, 'support': {s: v.tolist() for s,v in support.items()}, 'units': records,
            'model_fit': False, 'scoring': False, 'interpretation': plan['interpretation']}


def materialize(run, req):
    plan = req['definition']
    for name, sha in plan['metadata_hashes'].items(): require(file_hash(REFERENCE/name) == sha, 'Reference metadata drift')
    for name, sha in plan['raw_hashes'].items(): guard(req); require(file_hash(RAW/name) == sha, 'Raw source drift')
    for i,item in enumerate(plan['items']):
        guard(req); print(f'{i+1}/20 {item["split"]}: {item["unit"]["id"]}', flush=True)
        positions,support = positions_for(item['unit'])
        unit_checkpoint(run,item,positions,support,guard=lambda:guard(req))
    for name, sha in plan['raw_hashes'].items(): guard(req); require(file_hash(RAW/name) == sha, 'Raw source changed')
    request(run); result = verify_materialized(run,req); guard(req)
    path=run/'result.json'
    if path.exists(): require(read(path) == result, 'Interrupted result differs')
    else: publish(path,canonical(result))


def verify_completed(run):
    req=request(run); summary=read(run/'complete.json'); result=verify_materialized(run,req)
    require(summary['status']=='COMPLETE' and summary['elapsed_seconds']<900
            and summary['result_sha256']==file_hash(run/'result.json') and read(run/'result.json')==result, 'Completion drift')
    for item in req['definition']['items']:
        require(file_hash(safe_artifact(REFERENCE,item['reference']))==item['reference_sha256'], 'Protected source drift')
    return {**result,'resources':summary}


def execute(run,recover=False):
    with file_lock(run/'.supervisor.guard'):
        req=request(run)
        if (run/'complete.json').exists(): return verify_completed(run)
        require(recover or not (run/'attempts').exists(),'Explicit recovery required')
        budget=Budget(req); attempt=run/'attempts'/('a'+uuid.uuid4().hex); attempt.mkdir(parents=True)
        publish(attempt/'owner.json',canonical({'pid':os.getpid(),'created':psutil.Process().create_time()}))
        process=tree=None; reason=None; peak=size_peak=0; minimum=None
        try:
            guard(req); require(policy.start_reason(host.resources(run),limits=LIMITS) is None,'Disk start gate')
            with file_lock(run/'.worker.guard'): pass
            with (attempt/'worker.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'-B','-u','-m','experiments.cicids_partition.materialize',
                    '--_worker',str(run),'--attempt',attempt.name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                tree=host.ProcessTree(process.pid); last=-15
                while True:
                    elapsed=budget.elapsed(); state=host.resources(run); rss=tree.rss(); size=host.artifact_size(run)
                    peak=max(peak,rss); size_peak=max(size_peak,size); minimum=min(minimum or state['available'],state['available'])
                    reason=policy.stop_reason(state,None,rss,size,elapsed,limits=LIMITS)
                    if elapsed-last>=15 or reason or process.poll() is not None:
                        print(f'[{elapsed:.0f}/900s] {len(list((run/"units").glob("*.json")))}/20 units | worker {rss/2**30:.2f} GiB',flush=True);last=elapsed
                    if reason or process.poll() is not None: break
                    time.sleep(.3)
        except BaseException as exc: reason=f'{type(exc).__name__}: {exc}'
        finally:
            if tree is not None: tree.stop()
            if process is not None: process.wait(timeout=10)
        if process is None or process.returncode!=0: reason=reason or 'Worker failed; inspect worker.log'
        if reason is None:
            try:
                request(run); require(read(run/'result.json')==verify_materialized(run,req),'Result differs'); guard(req)
            except BaseException as exc: reason=f'{type(exc).__name__}: {exc}'
        summary={'scope':SCOPE,'status':'STOPPED' if reason else 'COMPLETE','reason':reason,'elapsed_seconds':budget.elapsed(),
                 'peak_tree_rss':peak,'minimum_available':minimum,'peak_artifacts':size_peak,
                 'worker_exit':None if process is None else process.returncode,'run_id':run.name,
                 'request_sha256':file_hash(run/'request.json')}
        if reason is None: summary['result_sha256']=file_hash(run/'result.json')
        publish(attempt/'result.json',canonical(summary))
        if reason is None:
            publish(run/'complete.json',canonical(summary)); return verify_completed(run)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__); modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--materialize',action='store_true'); modes.add_argument('--recover',type=Path)
    modes.add_argument('--verify',type=Path); modes.add_argument('--_worker',type=Path); parser.add_argument('--attempt')
    args=parser.parse_args()
    if args._worker:
        run=args._worker; req=request(run); owner=read(run/'attempts'/args.attempt/'owner.json'); done=threading.Event()
        threading.Thread(target=watch,args=(owner['pid'],owner['created'],req['deadline_epoch'],done),daemon=True).start()
        try:
            with file_lock(run/'.worker.guard'): materialize(run,req)
        finally: done.set()
        return
    if args.verify: result=verify_completed(output_location(args.verify))
    elif args.recover: result=execute(output_location(args.recover),True)
    else:
        plan=definition(); run=OUTPUT/('contiguous-seed123-'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz'))
        run.mkdir(parents=True); now=time.time()
        publish(run/'request.json',canonical({'definition':plan,'started_epoch':now,'deadline_epoch':now+900}))
        print('Materialization run: '+str(run),flush=True); result=execute(run)
    print(json.dumps(result,indent=2))
    if result['status']!='COMPLETE': sys.exit(2)


if __name__=='__main__': main()
