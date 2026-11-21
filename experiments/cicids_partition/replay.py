"""Replay only the historical seed-123 group assignment; never materialize splits.

--probe creates immutable audit evidence outside the repository. --recover reuses
a verified completed assignment within the original 15-minute deadline. --resume
only verifies a completed audit. No model fit or new scientific partition policy.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from experiments.c03.engineering import preparation_probe as host, resource_policy as policy, lifecycle_probe as lifecycle
from experiments.c03.producer.plan import ROOT, canonical, digest, file_hash, require, output_location
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.artifacts import array_hash, safe_artifact, write_arrays
from experiments.c03.producer.locking import file_lock

SCOPE = 'ENGINEERING_CICIDS_HISTORICAL_ASSIGNMENT_REPLAY_ONLY'
GENERATOR = ROOT/'.release-audit/cicids-partition/materializer-bff6691.py'
GENERATOR_SHA = '88c3091f14997f7cfc059d415284b00fcaf49b131ae859a7a1155b7136646e3d'
GENERATOR_REVISION = 'bff66918cbf28d6a1d70e17bcfaeb51a7f11057b'
SOURCE = ROOT/'processed_V5/A_stratified/CICIDS2017'
REFERENCE = ROOT/'outputs/12_jisa_finalization/04_group_safe_surface'
OUTPUT = ROOT.parent/'VERA-IDS-local-runs/cicids-partition'
LIMITS = {k: policy.LIMITS[k] for k in ('seconds','interval','start_available','min_available','max_rss','start_disk','min_disk','max_artifacts')}
REFERENCES = ('group_safe_surface_summary.json','materialized_family_support.csv',
              'materialized_support_repair_moves.csv','feature_columns.txt')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_generator():
    require(file_hash(GENERATOR) == GENERATOR_SHA, 'Historical generator bytes differ')
    spec = importlib.util.spec_from_file_location('cicids_historical_replay',GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args,**kwargs):
        raise RuntimeError('Replay cannot materialize, replace or delete historical splits')
    module.main = module.materialize = forbidden
    return module


def definition():
    require(file_hash(GENERATOR)==GENERATOR_SHA,'Historical generator bytes differ')
    sources=[Path(__file__),Path(__file__).with_name('__init__.py'),ROOT/'tests/test_cicids_partition_replay.py',
             Path(host.__file__),Path(policy.__file__),Path(lifecycle.__file__),
             ROOT/'experiments/c03/producer/plan.py',ROOT/'experiments/c03/producer/artifacts.py',
             ROOT/'experiments/c03/producer/checkpoints.py',ROOT/'experiments/c03/producer/locking.py']
    return {'scope':SCOPE,'generator_revision':GENERATOR_REVISION,'generator_sha256':GENERATOR_SHA,
        'seed':123,'minimum_support':200,'chunksize':200000,'limits':LIMITS,
        'sources':{p.relative_to(ROOT).as_posix():file_hash(p) for p in sources},
        'reference_hashes':{name:file_hash(REFERENCE/name) for name in REFERENCES},
        'environment':{'python':sys.version.split()[0],**{n:version(n) for n in ('numpy','pandas','pyarrow','psutil')}},
        'allowed_output':'Audit arrays and compact comparisons only; no split materialization or training',
        'historical_identity_limit':'Historical aggregate fingerprint is not a row-assignment checksum; 64-bit hashes are not collision-audited here'}


def inventory(source,emit):
    import pyarrow.parquet as pq
    result=[]
    for split in ('train','val','test'):
        paths=sorted((source/split).glob('*.parquet'))
        require(paths and not list((source/split).glob('*.csv*')),'Mixed/missing source formats')
        for path in paths:
            require(not path.is_symlink(),'Redirected source part')
            emit('hashing_source:'+path.relative_to(source).as_posix())
            result.append({'path':path.relative_to(source).as_posix(),'sha256':file_hash(path),
                           'rows':pq.ParquetFile(path).metadata.num_rows,'bytes':path.stat().st_size})
    return result


def request(run):
    value=read(safe_artifact(run,'request.json'))
    require(value['definition']==definition() and value['deadline_epoch']==value['started_epoch']+LIMITS['seconds'],
            'Replay source, limits or deadline drift')
    return value


def assignment_digest(groups,splits):
    import numpy as np
    order=np.argsort(groups)
    return digest({'sorted_group_hashes':array_hash(groups[order]),'split_codes':array_hash(splits[order]),
                   'codes':{'0':'train','1':'val','2':'test'}})


def compare(meta,support,moves,features,source_files,reference):
    import pandas as pd
    observed={'source_part_files':int(source_files),'seed':123,'minimum_family_support':200,
        'feature_columns':len(features),'rows':int(meta['n'].sum()),'unique_exact_feature_groups':len(meta),
        'family_conflict_groups':int((meta['nfam']>1).sum()),'repair_moves':len(moves),
        'repair_rows':sum(int(m['rows']) for m in moves),
        'assignment_row_totals':{s:int(meta.loc[meta['split']==s,'n'].sum()) for s in ('train','val','test')}}
    old=read(reference/'group_safe_surface_summary.json')
    checks={k:observed[k]==old[k] for k in observed}
    expected_support=pd.read_csv(reference/'materialized_family_support.csv').set_index('family')
    actual_support=support.copy();actual_support['min_support']=support.min(axis=1)
    checks['support_table']=actual_support.sort_index().to_dict()==expected_support.sort_index().to_dict()
    checks['repair_log']=moves==pd.read_csv(reference/'materialized_support_repair_moves.csv').to_dict('records')
    checks['feature_order']=features==(reference/'feature_columns.txt').read_text().splitlines()
    return {'observed':observed,'checks':checks,'all_match':all(checks.values())}


def verify_assignment(run):
    receipt=read(safe_artifact(run,'assignment-complete.json'))
    require(receipt['scope']==SCOPE and receipt['request_sha256']==file_hash(run/'request.json')
            and receipt['inventory_sha256']==file_hash(run/'inventory.json'),'Assignment binding drift')
    for name,sha in receipt['files'].items():
        require(file_hash(safe_artifact(run,name))==sha,'Assignment artifact drift')
    return receipt


def calculate(run,directory,emit):
    import numpy as np
    module=load_generator()
    current=inventory(SOURCE,emit)
    require(len(current)==24 and sum(x['rows'] for x in current)==module.EXPECTED_ROWS,'Historical source dimensions differ')
    if (run/'inventory.json').exists():
        require(read(run/'inventory.json')==current,'Replay source bytes changed')
    else: publish(run/'inventory.json',canonical(current))
    if (run/'assignment-complete.json').exists():
        emit('reusing_verified_assignment')
        receipt=verify_assignment(run)
        return read(safe_artifact(run,receipt['comparison_path']))
    emit('rebuilding_historical_seed123_assignment')
    assignment,meta,support,moves,features,source_files=module.build_assignment(SOURCE,seed=123,min_support=200,chunksize=200000)
    emit('comparing_saved_aggregate_support_and_repairs')
    result=compare(meta,support,moves,features,source_files,REFERENCE)
    codes=meta['split'].map({'train':0,'val':1,'test':2}).to_numpy(dtype=np.uint8)
    groups=meta['h'].to_numpy(dtype=np.uint64)
    require(len(np.unique(groups))==len(groups) and len(assignment)==len(groups),'Duplicate assignment entries')
    result['assignment_sha256']=assignment_digest(groups,codes)
    result['identity_scope']='New current-environment group-to-split checksum; historical evidence supplied aggregates, not this checksum'
    write_arrays(directory/'assignment.npz',group_hash=groups,split_code=codes,group_rows=meta['n'].to_numpy(dtype=np.int64))
    publish(directory/'comparison.json',canonical(result))
    require(inventory(SOURCE,emit)==current,'Source changed during replay')
    request(run)
    files={p.relative_to(run).as_posix():file_hash(p) for p in (directory/'assignment.npz',directory/'comparison.json')}
    publish(run/'assignment-complete.json',canonical({'scope':SCOPE,'request_sha256':file_hash(run/'request.json'),
        'inventory_sha256':file_hash(run/'inventory.json'),'files':files,'comparison_path':(directory/'comparison.json').relative_to(run).as_posix()}))
    emit('assignment_checkpoint_committed')
    return result


def watch(pid,created,budget,done):
    while not done.wait(.3):
        if not lifecycle.supervisor_alive(pid,created) or budget.elapsed()>=LIMITS['seconds']:os._exit(15)


def worker(run,attempt):
    req=request(run);directory=safe_artifact(run,'attempts/'+attempt)
    job=read(directory/'job.json')
    require(job['request_sha256']==file_hash(run/'request.json') and lifecycle.supervisor_alive(job['pid'],job['created']),
            'Replay worker has no valid owner')
    budget,done=lifecycle.Budget(req),threading.Event()
    threading.Thread(target=watch,args=(job['pid'],job['created'],budget,done),daemon=True).start()
    try:
        with file_lock(run/'.worker.guard'),(directory/'progress.jsonl').open('x') as stream:
            def emit(stage):
                stream.write(json.dumps({'stage':stage,'seconds':budget.elapsed()})+'\n');stream.flush()
            require(budget.elapsed()<LIMITS['seconds'],'Deadline reached')
            result=calculate(run,directory,emit)
            require(budget.elapsed()<LIMITS['seconds'],'Deadline reached')
            publish(directory/'complete.json',canonical({'result':result,'request_sha256':file_hash(run/'request.json')}))
    finally:done.set()


def command(run,attempt):
    return [sys.executable,'-B','-u','-m','experiments.cicids_partition.replay','--_worker',str(run),'--attempt',attempt]


def execute(run,recover=False):
    import psutil
    with file_lock(run/'.pipeline.guard'):
        req=request(run)
        if (run/'complete.json').exists():return verify_completed(run)
        budget=lifecycle.Budget(req)
        require(budget.elapsed()<LIMITS['seconds'],'Original replay deadline exhausted')
        with file_lock(run/'.worker.guard'):pass
        require(recover or not (run/'attempts').exists(),'Explicit recovery required')
        directory=run/'attempts'/('a'+uuid.uuid4().hex);directory.mkdir(parents=True)
        publish(directory/'job.json',canonical({'request_sha256':file_hash(run/'request.json'),
            'pid':os.getpid(),'created':psutil.Process().create_time()}))
        process=tree=None;reason=None;peak=size_peak=0;minimum=None
        try:
            state=host.resources(run)
            reason=policy.start_reason(state,limits=LIMITS)
            reason=reason or policy.stop_reason(state,None,0,host.artifact_size(run),budget.elapsed(),limits=LIMITS)
            if not reason:
                with (directory/'worker.log').open('x') as log,(directory/'monitor.jsonl').open('x') as trace:
                    process=subprocess.Popen(command(run,directory.name),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    tree=host.ProcessTree(process.pid);previous=None;last_report=-100.
                    while True:
                        state,rss,size=host.resources(run),tree.rss(),host.artifact_size(run)
                        elapsed=budget.elapsed();peak=max(peak,rss);size_peak=max(size_peak,size)
                        minimum=state['available'] if minimum is None else min(minimum,state['available'])
                        trace.write(json.dumps({'seconds':elapsed,'rss':rss,'artifact_bytes':size,**state})+'\n');trace.flush()
                        stage='starting'
                        try:
                            text=(directory/'progress.jsonl').read_text();lines=text[:text.rfind('\n')+1].splitlines()
                            if lines:stage=json.loads(lines[-1])['stage']
                        except (OSError,ValueError,KeyError):pass
                        if stage!=previous or elapsed-last_report>=30:
                            print(f'[{elapsed/60:.1f}/15 min] {stage} | worker {rss/2**30:.2f}/4 GiB',flush=True)
                            previous,last_report=stage,elapsed
                        reason=policy.stop_reason(state,None,rss,size,elapsed,limits=LIMITS)
                        if reason:break
                        if process.poll() is not None:
                            require(process.returncode==0 and not tree.live(),'Worker failed; see worker.log')
                            require(read(directory/'complete.json')['request_sha256']==file_hash(run/'request.json'),'Worker receipt differs')
                            break
                        time.sleep(LIMITS['interval'])
        except BaseException as exc:reason=f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if tree is not None:tree.stop()
                elif process is not None and process.poll() is None:host.ProcessTree(process.pid).stop()
                if process is not None:process.wait(timeout=10)
            except Exception as exc:reason=f'Cleanup failed: {exc}; prior={reason}'
        comparison=None
        if not reason:
            try:
                receipt=verify_assignment(run)
                comparison=read(safe_artifact(run,receipt['comparison_path']))
                reason=policy.stop_reason(host.resources(run),None,0,host.artifact_size(run),budget.elapsed(),limits=LIMITS)
            except Exception as exc:reason=f'Completion verification failed: {exc}'
        status='STOPPED' if reason else 'COMPLETE' if comparison['all_match'] else 'MISMATCH'
        result={'scope':SCOPE,'status':status,'reason':reason,'comparison':comparison,'elapsed_seconds':budget.elapsed(),
            'peak_tree_rss':peak,'minimum_available':minimum,'max_artifact_bytes':size_peak,
            'worker_exit':None if process is None else process.returncode,'attempt':directory.relative_to(run).as_posix(),'run_id':run.name}
        publish(directory/'supervisor.json',canonical(result))
        if not reason:
            publish(run/'complete.json',canonical({'request_sha256':file_hash(run/'request.json'),
                'assignment_receipt_sha256':file_hash(run/'assignment-complete.json'),
                'summary_path':(directory/'supervisor.json').relative_to(run).as_posix(),
                'summary_sha256':file_hash(directory/'supervisor.json')}))
        return result


def verify_completed(run):
    request(run);verify_assignment(run)
    receipt=read(safe_artifact(run,'complete.json'))
    require(receipt['request_sha256']==file_hash(run/'request.json')
            and receipt['assignment_receipt_sha256']==file_hash(run/'assignment-complete.json')
            and receipt['summary_sha256']==file_hash(safe_artifact(run,receipt['summary_path'])),'Replay completion drift')
    result=read(safe_artifact(run,receipt['summary_path']))
    require(result['status'] in ('COMPLETE','MISMATCH') and result['elapsed_seconds']<LIMITS['seconds'],'Incomplete replay')
    return result


def start(recover=None):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if recover:run=output_location(recover)
    else:
        run=OUTPUT/('seed123-replay-'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz'))
        base=definition();run.mkdir()
        now=time.time();publish(run/'request.json',canonical({'definition':base,'started_epoch':now,'deadline_epoch':now+LIMITS['seconds']}))
    print('Replay run: '+str(run),flush=True)
    return execute(run,recover=recover is not None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--probe',action='store_true');modes.add_argument('--recover',type=Path)
    modes.add_argument('--resume',type=Path);modes.add_argument('--_worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--attempt',help=argparse.SUPPRESS);args=parser.parse_args()
    if args._worker:worker(output_location(args._worker),args.attempt);return 0
    result=verify_completed(output_location(args.resume)) if args.resume else start(args.recover)
    print(json.dumps(result,indent=2));return 0 if result['status']=='COMPLETE' else 3


if __name__=='__main__':sys.exit(main())
