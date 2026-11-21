"""Bounded selected-TRAIN numeric extraction for the approved native EFC pilot.

No fitting, imputation, validation/test observations, or upstream model imports.
Completed source-part checkpoints survive interruption within the original budget.
"""
import argparse
from datetime import datetime,timezone
import json,os,sys,subprocess,threading,time,uuid
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import psutil

from experiments.c03.producer.plan import ROOT,canonical,digest,file_hash,require,output_location
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.artifacts import safe_artifact,array_hash,write_arrays
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer.validation import truth_labels
from experiments.c03.engineering import preparation_probe as host,resource_policy as policy,lifecycle_probe as life

SCOPE='ENGINEERING_EFC_SELECTED_TRAIN_INPUTS_ONLY'
REFERENCE=ROOT.parent/'VERA-IDS-local-runs/c03/pilot-producer-20261117t210546664156z'
OUTPUT=ROOT.parent/'VERA-IDS-local-runs/literature'
LIMITS={k:policy.LIMITS[k] for k in ('seconds','interval','max_rss','start_disk','min_disk','max_artifacts','start_available','min_available')}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def definition():
    receipt=read(REFERENCE/'prepared/complete.json')
    meta=read(REFERENCE/'prepared/preprocessor/preprocessor_meta.json')
    require(receipt['plan_sha256']=='6b54cfd10448ff4f4cf9cc601e3269debe14f95e78999b0378ffba955d057bdd','Wrong fixed C03 case')
    require(meta['n_num']==46 and meta['n_cat']==0 and len(set(meta['num_cols']))==46
            and not any(c.startswith('y_') for c in meta['num_cols']),'Unexpected predictor definition')
    require(file_hash(REFERENCE/'prepared/preprocessor/preprocessor_meta.json')==
            receipt['files']['prepared/preprocessor/preprocessor_meta.json'],'Predictor metadata drift')
    paths=[Path(__file__),ROOT/'tests/test_efc_input_probe.py',Path(host.__file__),Path(policy.__file__),Path(life.__file__),
           ROOT/'experiments/c03/producer/plan.py',ROOT/'experiments/c03/producer/artifacts.py',
           ROOT/'experiments/c03/producer/locking.py',ROOT/'experiments/c03/producer/checkpoints.py',
           ROOT/'experiments/c03/producer/validation.py']
    return {'scope':SCOPE,'dataset':'CICIoT2023','holdout':'Botnet','seed':123,'reference_run_id':REFERENCE.name,
        'reference_receipt_sha256':file_hash(REFERENCE/'prepared/complete.json'),
        'source_array_sha256':receipt['files']['prepared/stage1.npz'],
        'identity_sha256':receipt['bindings']['stage1']['row_ids'],'binary_labels_sha256':receipt['bindings']['stage1']['labels'],
        'rows':1270025,'columns':meta['num_cols'],'inventory':receipt['source_inventory']['train'],
        'limits':LIMITS,'sources':{p.relative_to(ROOT).as_posix():file_hash(p) for p in paths},
        'input_policy':'Preserve raw numeric values and count nonfinite cells; no imputation or preprocessing fit',
        'sampling':'Exact completed C03 known/benign TRAIN identities, no resampling',
        'environment':{'python':sys.version.split()[0],'numpy':np.__version__}}


def request(run):
    value=read(safe_artifact(run,'request.json'))
    require(value['definition']==definition() and value['deadline_epoch']==value['started_epoch']+LIMITS['seconds'],'Input probe definition/deadline drift')
    return value


def load_index(plan):
    path=REFERENCE/'prepared/stage1.npz'
    require(file_hash(path)==plan['source_array_sha256'],'Completed C03 array drift')
    with np.load(path,allow_pickle=False) as archive:
        identity=digest({k:array_hash(archive[k]) for k in ('source_part','source_sha256','row_offset')})
        require(identity==plan['identity_sha256'],'Sample identity differs')
        parts,offsets,binary=archive['source_part'],archive['row_offset'],archive['target']
    require(len(parts)==len(offsets)==len(binary)==plan['rows'] and array_hash(binary)==plan['binary_labels_sha256'],
            'Sample size/binary labels differ')
    require(np.isin(binary,[0,1]).all() and (offsets>=0).all(),'Invalid sample indices/labels')
    require(set(parts)<=set(e['path'] for e in plan['inventory']),'Unregistered sampled part')
    return parts,offsets,binary


def extract_part(path,expected_sha,columns,offsets,binary,emit=lambda _:None):
    require(file_hash(path)==expected_sha,'Selected TRAIN source drift')
    require(len(offsets)>0 and len(np.unique(offsets))==len(offsets) and (offsets>=0).all(),'Repeated/invalid source rows')
    source=pq.ParquetFile(path)
    mapping={str(c).strip().lower().replace(' ','_'):c for c in source.schema_arrow.names}
    require(len(mapping)==len(source.schema_arrow.names),'Canonical source names collide')
    required=[*columns,'y_stage1_attack','y_stage2_family']
    require(set(required)<=set(mapping) and int(max(offsets))<source.metadata.num_rows,'Missing predictors/row range')
    order=np.argsort(offsets);sorted_offsets=offsets[order]
    X=np.empty((len(offsets),len(columns)),dtype=np.float64);y=np.empty(len(offsets),dtype=np.int8)
    start=filled=0
    for batch in source.iter_batches(batch_size=8192,columns=[mapping[c] for c in required],use_threads=False):
        end=start+len(batch)
        lo,hi=np.searchsorted(sorted_offsets,[start,end])
        if hi>lo:
            frame=batch.to_pandas();frame.columns=[str(c).strip().lower().replace(' ','_') for c in frame.columns]
            selected=frame.iloc[sorted_offsets[lo:hi]-start]
            target=truth_labels(selected);destination=order[lo:hi]
            require((target<6).all() and np.array_equal(target!=0,binary[destination]),'Held-out/binary truth mismatch')
            X[destination]=selected[columns].to_numpy(dtype=np.float64);y[destination]=target
            filled+=hi-lo
        start=end
        if filled==len(offsets):break
    require(filled==len(offsets) and file_hash(path)==expected_sha,'Incomplete or changed selected TRAIN source')
    emit('source_verified')
    return X,y


def verify_part(run,path,plan,entry,positions):
    receipt=read(path/'complete.json')
    require(receipt['request_sha256']==file_hash(run/'request.json') and receipt['source']==entry
            and receipt['positions_sha256']==array_hash(positions),'Part checkpoint identity drift')
    require(file_hash(safe_artifact(path,'values.npz'))==receipt['values_sha256'],'Part checkpoint bytes drift')
    return receipt


def calculate(run,emit):
    req=request(run);plan=req['definition'];emit('verify_completed_C03_TRAIN_identities')
    parts,offsets,binary=load_index(plan);folder=run/'parts';folder.mkdir(exist_ok=True)
    receipts=[];reused=[]
    for number,entry in enumerate(plan['inventory']):
        positions=np.flatnonzero(parts==entry['path'])
        if not len(positions):continue
        require(entry['path'].startswith('train/'),'Only TRAIN sources allowed')
        source=safe_artifact(host.DATA_ROOT,entry['path']);target=folder/f'p{number:04d}';target.mkdir(exist_ok=True)
        emit(f'part_{number+1}_of_{len(plan["inventory"])}_rows_{len(positions)}')
        if (target/'complete.json').exists():
            require(file_hash(source)==entry['sha256'],'Previously extracted source drift')
            receipt=verify_part(run,target,plan,entry,positions);reused.append(number)
        else:
            # Incomplete part artifacts are retained in distinct attempt folders.
            attempt=target/('a'+uuid.uuid4().hex);attempt.mkdir()
            X,y=extract_part(source,entry['sha256'],plan['columns'],offsets[positions],binary[positions])
            write_arrays(attempt/'values.npz',X=X,y=y,positions=positions)
            payload={'request_sha256':file_hash(run/'request.json'),'source':entry,'rows':len(y),
                'positions_sha256':array_hash(positions),'nonfinite_by_column':np.sum(~np.isfinite(X),axis=0).tolist(),
                'class_support':np.bincount(y,minlength=6).tolist(),'values_sha256':file_hash(attempt/'values.npz')}
            # Link the completed data once; recover a publication interrupted before its receipt.
            if (target/'values.npz').exists():require(file_hash(target/'values.npz')==payload['values_sha256'],'Uncommitted part differs')
            else:os.link(attempt/'values.npz',target/'values.npz')
            publish(target/'complete.json',canonical(payload));receipt=verify_part(run,target,plan,entry,positions)
            del X,y
        receipts.append({'part':target.relative_to(run).as_posix(),'receipt_sha256':file_hash(target/'complete.json'),**receipt})
    counts=np.sum([r['class_support'] for r in receipts],axis=0)
    invalid=np.sum([r['nonfinite_by_column'] for r in receipts],axis=0)
    require(sum(r['rows'] for r in receipts)==plan['rows'] and (counts>=200).all(),'Incomplete sample/class support')
    require(request(run)==req,'Definition changed during extraction')
    return {'scope':SCOPE,'status':'COMPLETE','ready_for_preprocessing':not bool(invalid.any()),
        'rows':plan['rows'],'columns':plan['columns'],'class_support':counts.tolist(),'nonfinite_by_column':invalid.tolist(),
        'parts':receipts,'reused_parts':reused,'identity_sha256':plan['identity_sha256'],
        'models_fitted':False,'validation_or_test_read':False,'imputation_applied':False}


def watch(pid,created,deadline,done):
    while not done.wait(.3):
        if not life.supervisor_alive(pid,created) or time.time()>=deadline:os._exit(15)


def worker(run,attempt):
    req=request(run);directory=safe_artifact(run,'attempts/'+attempt);job=read(directory/'job.json')
    done=threading.Event()
    threading.Thread(target=watch,args=(job['owner_pid'],job['owner_created'],req['deadline_epoch'],done),daemon=True).start()
    try:
        with file_lock(run/'.worker.guard'),(directory/'stages.jsonl').open('x') as stream:
            def emit(stage):stream.write(json.dumps({'stage':stage})+'\n');stream.flush()
            result=calculate(run,emit)
            require(time.time()<req['deadline_epoch'],'Original deadline exhausted')
            publish(directory/'result.json',canonical(result))
    finally:done.set()


def verify_completed(run):
    req=request(run);receipt=read(safe_artifact(run,'complete.json'))
    result_path=safe_artifact(run,receipt['result_path'])
    require(receipt['request_sha256']==file_hash(run/'request.json') and receipt['result_sha256']==file_hash(result_path),
            'Completion identity drift')
    result=read(result_path)
    require(result['status']=='COMPLETE' and receipt['elapsed_seconds']<LIMITS['seconds'],'Incomplete/late extraction')
    for item in result['parts']:
        path=safe_artifact(run,item['part'])
        require(file_hash(path/'complete.json')==item['receipt_sha256'] and file_hash(path/'values.npz')==item['values_sha256'],
                'Completed part drift')
    return {**result,'resources':receipt}


def execute(run,recover=False):
    with file_lock(run/'.pipeline.guard'):
        req=request(run)
        if (run/'complete.json').exists():return verify_completed(run)
        require(recover or not (run/'attempts').exists(),'Explicit recovery required')
        with file_lock(run/'.worker.guard'):pass
        budget=life.Budget(req);directory=run/'attempts'/('a'+uuid.uuid4().hex);directory.mkdir(parents=True)
        publish(directory/'job.json',canonical({'owner_pid':os.getpid(),'owner_created':psutil.Process().create_time()}))
        process=tree=None;reason=None;peak=size_peak=0;minimum=None
        try:
            reason=policy.start_reason(host.resources(run),limits=LIMITS)
            reason=reason or policy.stop_reason(host.resources(run),None,0,host.artifact_size(run),budget.elapsed(),limits=LIMITS)
            if not reason:
                with (directory/'worker.log').open('x') as log:
                    process=subprocess.Popen([sys.executable,'-B','-u','-m',__name__ if __name__!='__main__' else
                        'experiments.literature.efc_input_probe','--_worker',str(run),'--attempt',directory.name],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    tree=host.ProcessTree(process.pid);last=None;last_time=-30
                    while True:
                        state=host.resources(run);rss=tree.rss();size=host.artifact_size(run);elapsed=budget.elapsed()
                        peak=max(peak,rss);size_peak=max(size_peak,size);minimum=min(minimum or state['available'],state['available'])
                        stage='starting'
                        try:
                            lines=(directory/'stages.jsonl').read_text().splitlines();stage=json.loads(lines[-1])['stage']
                        except (OSError,IndexError,ValueError):pass
                        if stage!=last or elapsed-last_time>=30:
                            print(f'[{elapsed/60:.1f}/15 min] {stage} | {rss/2**30:.2f}/4 GiB',flush=True);last,last_time=stage,elapsed
                        reason=policy.stop_reason(state,None,rss,size,elapsed,limits=LIMITS)
                        if reason:break
                        if process.poll() is not None:
                            require(process.returncode==0 and not tree.live(),'Worker failed; see worker.log');break
                        time.sleep(.3)
        except BaseException as exc:reason=f'{type(exc).__name__}: {exc}'
        finally:
            if tree is not None:tree.stop()
            if process is not None:process.wait(timeout=10)
        summary={'scope':SCOPE,'status':'STOPPED' if reason else 'COMPLETE','reason':reason,'elapsed_seconds':budget.elapsed(),
            'peak_tree_rss':peak,'minimum_available':minimum,'max_artifact_bytes':size_peak,'run_id':run.name,
            'worker_exit':None if process is None else process.returncode}
        if not reason:
            require((directory/'result.json').is_file(),'Missing worker result')
            summary.update(request_sha256=file_hash(run/'request.json'),result_path=(directory/'result.json').relative_to(run).as_posix(),
                           result_sha256=file_hash(directory/'result.json'))
        publish(directory/'supervisor.json',canonical(summary))
        if not reason:publish(run/'complete.json',canonical(summary));return verify_completed(run)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--probe',action='store_true');modes.add_argument('--recover',type=Path)
    modes.add_argument('--resume',type=Path);modes.add_argument('--_worker',type=Path);parser.add_argument('--attempt')
    args=parser.parse_args()
    if args._worker:worker(output_location(args._worker),args.attempt);return 0
    if args.resume:result=verify_completed(output_location(args.resume))
    else:
        if args.recover:run=output_location(args.recover)
        else:
            plan=definition();run=OUTPUT/('efc-train-inputs-'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz'))
            run.mkdir(parents=True);now=time.time();publish(run/'request.json',canonical({'definition':plan,
                'started_epoch':now,'deadline_epoch':now+LIMITS['seconds']}))
        print('Input probe run: '+str(run),flush=True);result=execute(run,recover=args.recover is not None)
    print(json.dumps({k:v for k,v in result.items() if k!='parts'},indent=2));return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':sys.exit(main())
