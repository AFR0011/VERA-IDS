"""Approved 2026-11-18 first-support-valid contiguous-unit assignment audit.

Metadata only: no benchmark observations, fitted models or materialized splits.
Five partition seeds, fixed 12/4/4 units, >=200 of every retained family per split,
at most 10,000 permutations per seed. Duplicates/exhaustion are reported unchanged.
"""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

from experiments.c03.producer.plan import ROOT,canonical,digest,file_hash,require,output_location
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.artifacts import safe_artifact
from experiments.c03.producer.locking import file_lock
from experiments.c03.engineering import preparation_probe as host,resource_policy
from . import replay

FAMILIES=('Benign','Botnet','BruteForce','DDoS','DoS','Scan/Recon','Web/App')
SEEDS=(123,124,125,126,127)
METADATA=ROOT.parent/'Codes/ids_eval_framework/outputs/02_prepared_data/processed_V5_cicids17_recovery/B_day_file/CICIDS2017'
FILES=('protocol_b_unit_manifest.csv','protocol_b_file_family_support.csv')
SCOPE='CICIDS_CONTIGUOUS_FIRST_VALID_ASSIGNMENT_METADATA_ONLY'
LIMITS=dict(replay.LIMITS)


def rows(path,maximum):
    require(path.stat().st_size<=2*2**20,'Unexpectedly large metadata file')
    with path.open(encoding='utf-8-sig',newline='') as stream:
        result=[]
        for row in csv.DictReader(stream):
            require(len(result)<maximum,'Too many metadata rows')
            result.append(row)
    return result


def load_metadata(root=METADATA):
    hashes={name:file_hash(root/name) for name in FILES}
    raw=rows(root/FILES[0],20)
    require(len(raw)==20,'Expected original twenty units')
    units=[]
    for index,row in enumerate(raw):
        require(int(row['unit_order'])==index and row['partition_mode']=='contiguous_subfiles','Unit order/construction differs')
        name=row['source_file_name'];segment=int(row['segment_index'])
        require(Path(name).name==name and name.endswith('.csv') and int(row['segment_count'])==4
                and row['unit_id']==f'{name}::part{segment}of4','Unit identity differs')
        start,end=int(row['row_start']),int(row['row_end'])
        kept,dropped=int(row['kept_rows_after_family_drop']),int(row['dropped_rows'])
        require(0<=start<end and end-start==int(row['raw_row_count'])==kept+dropped and min(kept,dropped)>=0,'Unit boundaries differ')
        units.append({'id':row['unit_id'],'source_file':name,'segment':segment,'row_start':start,'row_end':end,'kept_rows':kept})
    require(len({u['id'] for u in units})==20 and len({u['source_file'] for u in units})==5,'Repeated/missing units')
    for name in {u['source_file'] for u in units}:
        group=[u for u in units if u['source_file']==name]
        require([u['segment'] for u in group]==[1,2,3,4] and group[0]['row_start']==0
                and all(a['row_end']==b['row_start'] for a,b in zip(group,group[1:])),'Source segments overlap or have gaps')
    matrix=np.zeros((20,len(FAMILIES)),dtype=np.int64);seen=set()
    for row in rows(root/FILES[1],140):
        i=int(row['unit_order']);family=row['family']
        require(0<=i<20 and family in FAMILIES and (i,family) not in seen
                and row['unit_id']==units[i]['id'],'Family support identity differs')
        require(row['source_file_name']==units[i]['source_file'] and int(row['row_start'])==units[i]['row_start']
                and int(row['row_end'])==units[i]['row_end'],'Support/unit boundary mismatch')
        count=int(row['count']);require(count>=0,'Negative support')
        matrix[i,FAMILIES.index(family)]=count;seen.add((i,family))
    require(matrix.sum(axis=1).tolist()==[u['kept_rows'] for u in units]
            and int(matrix.sum())==2099879,'Unit/family population differs')
    require(hashes=={name:file_hash(root/name) for name in FILES},'Metadata changed during read')
    return {'units':units,'family_order':list(FAMILIES),'family_counts':matrix.tolist(),'input_hashes':hashes}


def definition():
    paths=[Path(__file__),ROOT/'tests/test_cicids_seeded_assignments.py',Path(replay.__file__),
           Path(host.__file__),Path(resource_policy.__file__),ROOT/'experiments/c03/producer/plan.py',
           ROOT/'experiments/c03/producer/checkpoints.py',ROOT/'experiments/c03/producer/locking.py',
           ROOT/'experiments/c03/producer/artifacts.py']
    return {'scope':SCOPE,'authorization_date':'2026-11-18','seeds':list(SEEDS),
        'rng':'numpy.random.RandomState(seed).permutation(20)','numpy':np.__version__,'python':sys.version.split()[0],
        'unit_counts':{'train':12,'val':4,'test':4},'minimum_family_support':200,'max_attempts':10000,
        'selection':'First support-valid assignment in seeded draw order; no support-score or performance maximization',
        'limits':LIMITS,'sources':{p.relative_to(ROOT).as_posix():file_hash(p) for p in paths},
        'metadata':load_metadata(),'model_seed':None,'raw_observations_read':False,
        'interpretation':'Dependent partitions of the same corpus; contiguous units are not independent attack executions'}


def sample(matrix,seed,*,max_attempts=10000,guard=lambda:None):
    require(matrix.shape==(20,7) and np.issubdtype(matrix.dtype,np.integer) and (matrix>=0).all(),'Invalid support matrix')
    require(type(seed) is int and 0<=seed<2**32,'Invalid partition seed')
    require(type(max_attempts) is int and 1<=max_attempts<=10000,'Invalid attempt cap')
    rng=np.random.RandomState(seed)
    for attempt in range(1,max_attempts+1):
        if attempt%128==1:guard()
        draw=rng.permutation(20)
        assignment={'train':sorted(draw[:12].tolist()),'val':sorted(draw[12:16].tolist()),'test':sorted(draw[16:].tolist())}
        support={key:matrix[indices].sum(axis=0).tolist() for key,indices in assignment.items()}
        if all(min(counts)>=200 for counts in support.values()):
            return {'seed':seed,'status':'FEASIBLE','attempts':attempt,'assignment':assignment,
                    'support':support,'assignment_sha256':digest(assignment)}
    return {'seed':seed,'status':'EXHAUSTED','attempts':max_attempts,'assignment':None,
            'support':None,'assignment_sha256':None}


def request(run):
    req=json.loads(safe_artifact(run,'request.json').read_text())
    require(req['definition']==definition() and req['deadline_epoch']==req['started_epoch']+LIMITS['seconds'],'Assignment audit definition/deadline drift')
    return req


def evaluate_seed(plan,seed,guard=lambda:None):
    matrix=np.asarray(plan['metadata']['family_counts'],dtype=np.int64)
    first=sample(matrix,seed,guard=guard)
    second=sample(matrix,seed,guard=guard)
    require(first==second,'Partition seed failed deterministic regeneration')
    first['regeneration_verified']=True
    if first['assignment'] is not None:
        first['unit_ids']={key:[plan['metadata']['units'][i]['id'] for i in indices] for key,indices in first['assignment'].items()}
        first['row_totals']={key:sum(counts) for key,counts in first['support'].items()}
    return first


def summary(results):
    fingerprints=[v['assignment_sha256'] for v in results if v['status']=='FEASIBLE']
    return {'all_five_feasible':len(fingerprints)==len(SEEDS),
        'distinct_feasible_assignments':len(set(fingerprints)),
        'duplicate_seeds':[[v['seed'] for v in results if v['assignment_sha256']==sha]
            for sha in sorted(set(fingerprints)) if fingerprints.count(sha)>1]}


def verify_result(run,req,guard=lambda:None):
    result=json.loads(safe_artifact(run,'result.json').read_text())
    require(result['request_sha256']==file_hash(run/'request.json') and result['scope']==SCOPE
            and result['status']=='COMPLETE' and result['run_id']==run.name
            and result['new_split_files'] is False and result['models_fitted'] is False,'Result identity drift')
    for seed,value in zip(SEEDS,result['results'],strict=True):
        require(value==evaluate_seed(req['definition'],seed,guard),'Completed assignment drift')
        require(json.loads(safe_artifact(run,f'seeds/{seed}.json').read_text())==
                {'request_sha256':file_hash(run/'request.json'),'result':value},'Seed receipt drift')
    require(all(result[key]==value for key,value in summary(result['results']).items()),'Summary drift')
    require(0<=result['elapsed_seconds']<LIMITS['seconds'] and 0<result['peak_rss']<=LIMITS['max_rss'],
            'Audit resource evidence exceeds limits')
    return result


def execute(run,recover=False):
    with file_lock(run/'.writer.guard'):
        req=request(run);plan=req['definition']
        if (run/'complete.json').exists():return verify_completed(run)
        require(recover or not (run/'seeds').exists(),'Explicit checkpoint recovery required')
        monotonic=time.monotonic();base=time.time()-req['started_epoch']
        require(base>=0,'Clock moved before run start')
        peak=0;minimum=None
        def elapsed():return max(time.time()-req['started_epoch'],base+time.monotonic()-monotonic)
        def guard():
            nonlocal peak,minimum
            state=host.resources(run);rss=psutil.Process().memory_info().rss
            peak=max(peak,rss);minimum=state['available'] if minimum is None else min(minimum,state['available'])
            reason=resource_policy.stop_reason(state,None,rss,host.artifact_size(run),elapsed(),limits=LIMITS)
            require(reason is None,reason)
        guard();require(resource_policy.start_reason(host.resources(run),limits=LIMITS) is None,'Disk start gate failed')
        if (run/'result.json').exists():
            result=verify_result(run,req,guard)
            require(request(run)==req,'Metadata/source changed during recovery')
            guard()
            publish(run/'complete.json',canonical({'result_sha256':file_hash(run/'result.json')}))
            return result
        folder=run/'seeds';folder.mkdir(exist_ok=True)
        results=[];reused=[]
        for seed in SEEDS:
            guard();path=folder/f'{seed}.json'
            value=evaluate_seed(plan,seed,guard)
            receipt={'request_sha256':file_hash(run/'request.json'),'result':value}
            if path.exists():
                require(json.loads(safe_artifact(run,f'seeds/{seed}.json').read_text())==receipt,'Seed checkpoint drift')
                reused.append(seed)
            else:publish(path,canonical(receipt))
            results.append(value)
            print(f'Seed {seed}: {value["status"]}, draw {value["attempts"]}; deterministic replay verified',flush=True)
        require(request(run)==req,'Metadata/source changed during audit')
        guard()
        result={'scope':SCOPE,'status':'COMPLETE','request_sha256':file_hash(run/'request.json'),'results':results,
            **summary(results),
            'reused_seeds':reused,'elapsed_seconds':elapsed(),'peak_rss':peak,'minimum_available':minimum,
            'new_split_files':False,'models_fitted':False,'run_id':run.name}
        publish(run/'result.json',canonical(result))
        publish(run/'complete.json',canonical({'result_sha256':file_hash(run/'result.json')}))
        return result


def verify_completed(run):
    req=request(run)
    require(json.loads(safe_artifact(run,'complete.json').read_text())=={'result_sha256':file_hash(run/'result.json')}
            ,'Completion drift')
    return verify_result(run,req)


def start(recover=None):
    if recover:run=output_location(recover)
    else:
        plan=definition();run=replay.OUTPUT/('contiguous-seeds-'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz'))
        run.mkdir(parents=True);now=time.time()
        publish(run/'request.json',canonical({'definition':plan,'started_epoch':now,'deadline_epoch':now+LIMITS['seconds']}))
    print('Metadata audit run: '+str(run),flush=True)
    return execute(run,recover=recover is not None)


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--audit',action='store_true');modes.add_argument('--recover',type=Path);modes.add_argument('--resume',type=Path)
    args=parser.parse_args()
    result=verify_completed(output_location(args.resume)) if args.resume else start(args.recover)
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':sys.exit(main())
