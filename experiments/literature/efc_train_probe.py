"""One bounded native EFC TRAIN preprocessing or class-fit checkpoint.

No validation/test scoring or selection. Each invocation has an immutable
15-minute deadline; recovery never extends it. Incomplete fits restart.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
from importlib.metadata import version
import json,os,sys,subprocess,threading,time,uuid
from pathlib import Path
import numpy as np
import joblib
import psutil
from . import efc_input_probe as inputs,efc_adapter as adapter
from experiments.c03.producer.plan import ROOT,canonical,digest,file_hash,require,output_location
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.artifacts import safe_artifact,array_hash
from experiments.c03.producer.locking import file_lock

host,policy,life=inputs.host,inputs.policy,inputs.life
LIMITS=dict(inputs.LIMITS)
INPUT_RUN=inputs.OUTPUT/'efc-train-inputs-20261118t170122755261z'
NATIVE_RECORD=ROOT/'experiments/literature/EFC_NATIVE_EQUIVALENCE_SUCCESS.json'
SCOPE='EFC_NATIVE_FIXED_BOTNET_SEED123_TRAIN_ONLY'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def native_runtime(load=False):
    record=read(NATIVE_RECORD);package=inputs.OUTPUT/record['run_id']/'env/Lib/site-packages/efc'
    require(record['adapter_sha256']==file_hash(Path(adapter.__file__)),'Native adapter drift')
    require(record['discretizer_sha256']==file_hash(Path(adapter.__file__).with_name('efc_discretization.py')),'Binning source drift')
    for name,sha in record['installed_native_files'].items():require(file_hash(package/name)==sha,'Native installed source/binary drift')
    for name in ('numpy','scipy','scikit-learn','joblib'):require(version(name)==record['versions'][name],'Native dependency drift: '+name)
    if load:
        sys.path.append(str(package.parent))
        spec=importlib.util.find_spec('efc')
        require(spec is not None and Path(spec.origin).resolve()==(package/'__init__.py').resolve(),'Unexpected EFC installation')
        import efc
    return {'evidence_sha256':file_hash(NATIVE_RECORD),'installed_native_files':record['installed_native_files'],
            'versions':record['versions'],'python':sys.version.split()[0]}


def definition(stage,prepared=None):
    require(stage=='prepare' or stage in [f'class{k}' for k in range(6)],'Invalid TRAIN phase')
    require((stage=='prepare')==(prepared is None),'Class fit requires completed preprocessing')
    reference=read(ROOT/'experiments/literature/EFC_TRAIN_INPUT_SUCCESS.json')
    require(file_hash(INPUT_RUN/'complete.json')==reference['completion_sha256'],'Input completion changed')
    paths=[Path(__file__),ROOT/'tests/test_efc_train_probe.py',Path(adapter.__file__),Path(inputs.__file__),
           ROOT/'experiments/literature/efc_discretization.py',Path(host.__file__),Path(policy.__file__),Path(life.__file__),
           ROOT/'experiments/c03/producer/plan.py',ROOT/'experiments/c03/producer/checkpoints.py',
           ROOT/'experiments/c03/producer/artifacts.py',ROOT/'experiments/c03/producer/locking.py']
    base={'scope':SCOPE,'stage':stage,'dataset':'CICIoT2023','holdout':'Botnet','seed':123,'limits':LIMITS,
          'input_completion_sha256':reference['completion_sha256'],'input_run_id':INPUT_RUN.name,
          'rows':reference['rows'],'class_support':reference['class_support'],'columns':reference['columns'],
          'identity_sha256':reference['identity_sha256'],'native':native_runtime(),
          'parameters':{'pseudocounts':0.5,'cutoff_quantile':0.95,'n_bins':30,'n_jobs':1,
                        'subsample':None,'quantile_method':'linear','blas_threads':1},
          'sources':{p.relative_to(ROOT).as_posix():file_hash(p) for p in paths},
          'prepared_run_id':None,'prepared_completion_sha256':None,'validation_test_reads':False}
    if prepared is not None:
        prepared=output_location(prepared)
        require(prepared.parent==inputs.OUTPUT.resolve(),'Unexpected preprocessing location')
        receipt=verify_completed(prepared)
        require(receipt['stage']=='prepare','Expected preprocessing checkpoint')
        base.update(prepared_run_id=prepared.name,prepared_completion_sha256=file_hash(prepared/'complete.json'))
    return base


def request(run):
    req=read(safe_artifact(run,'request.json'));plan=req['definition']
    prepared=None if plan['prepared_run_id'] is None else inputs.OUTPUT/plan['prepared_run_id']
    require(plan==definition(plan['stage'],prepared) and req['deadline_epoch']==req['started_epoch']+LIMITS['seconds'],
            'Native TRAIN definition/deadline drift')
    return req


def assemble_inputs(directory,result,emit):
    shape=(result['rows'],len(result['columns']))
    X=np.lib.format.open_memmap(directory/'raw.npy',mode='w+',dtype=np.float64,shape=shape)
    y=np.lib.format.open_memmap(directory/'labels.npy',mode='w+',dtype=np.int8,shape=(shape[0],))
    seen=np.zeros(shape[0],dtype=bool)
    for i,item in enumerate(result['parts']):
        path=safe_artifact(INPUT_RUN,item['part'])/'values.npz'
        require(file_hash(path)==item['values_sha256'],'Imported numeric part drift')
        with np.load(path,allow_pickle=False) as arrays:
            positions=arrays['positions'];values=arrays['X'];labels=arrays['y']
            require(positions.dtype==np.int64 and (positions>=0).all() and (positions<shape[0]).all()
                    and len(np.unique(positions))==len(positions) and not seen[positions].any(),'Duplicate/invalid input positions')
            require(values.shape==(len(positions),shape[1]) and labels.shape==(len(positions),)
                    and np.isfinite(values).all() and np.isin(labels,np.arange(6)).all(),'Invalid native TRAIN values')
            X[positions]=values;y[positions]=labels;seen[positions]=True
        if i%20==0:emit(f'assemble_part_{i+1}_of_{len(result["parts"])}')
    require(seen.all() and np.bincount(y,minlength=6).tolist()==result['class_support'],'Incomplete native TRAIN assembly')
    X.flush();y.flush();return X,y


def calculate(run,directory,emit):
    plan=request(run)['definition'];native_runtime(load=True)
    if plan['stage']=='prepare':
        emit('verify_raw_TRAIN_checkpoints');result=inputs.verify_completed(INPUT_RUN)
        require(result['ready_for_preprocessing'],'Input policy unresolved')
        X,y=assemble_inputs(directory,result,emit)
        emit('fit_all_TRAIN_linear_quantiles')
        preprocessor,discretized,max_bin=adapter.preprocess_training(X)
        require(discretized.shape==X.shape and discretized.dtype==np.int64 and 1<=max_bin<=30,'Invalid native discretization')
        np.save(directory/'discrete.npy',discretized,allow_pickle=False)
        joblib.dump(preprocessor,directory/'preprocessor.joblib')
        emit('verify_preprocessor_reload')
        reloaded=joblib.load(directory/'preprocessor.joblib')
        # Check every TRAIN row without another full dense transformed allocation.
        for offset in range(0,len(X),8192):
            np.testing.assert_array_equal(reloaded.transform(X[offset:offset+8192]).astype('int64'),discretized[offset:offset+8192])
        payload={'stage':'prepare','rows':len(X),'max_bin':max_bin,'features':X.shape[1],
                 'class_support':result['class_support'],'exact_reload_transform_verified':True,
                 'identity_sha256':plan['identity_sha256'],'classifier_fitted':False}
        names=['raw.npy','labels.npy','discrete.npy','preprocessor.joblib']
        del X,y,discretized
    else:
        prepare=inputs.OUTPUT/plan['prepared_run_id'];completed=verify_completed(prepare)
        require(file_hash(prepare/'complete.json')==plan['prepared_completion_sha256'],'Preprocessing changed')
        target=safe_artifact(prepare,completed['artifact_directory'])
        X=np.load(target/'discrete.npy',mmap_mode='r',allow_pickle=False)
        y=np.load(target/'labels.npy',mmap_mode='r',allow_pickle=False)
        k=int(plan['stage'][-1]);positions=np.flatnonzero(y==k)
        require(len(positions)==plan['class_support'][k],'Native class support drift')
        emit(f'fit_native_class_{k}_rows_{len(positions)}')
        fitted=adapter.fit_class(X[positions],completed['max_bin'])
        emit('save_native_class');joblib.dump(fitted,directory/'model.joblib')
        reloaded=joblib.load(directory/'model.joblib')
        for name in ('sitefreq_','pairfreq_','coupling_matrix_','local_fields_','cutoff_'):
            np.testing.assert_array_equal(getattr(fitted,name),getattr(reloaded,name))
        payload={'stage':plan['stage'],'class_label':k,'rows':len(positions),'max_bin':completed['max_bin'],
                 'cutoff':float(fitted.cutoff_),'class_positions_sha256':array_hash(positions),
                 'exact_model_reload_verified':True,'classifier_fitted':True}
        names=['model.joblib']
    payload.update(scope=SCOPE,status='COMPLETE',request_sha256=file_hash(run/'request.json'),
        validation_test_read=False,artifact_directory=directory.relative_to(run).as_posix(),
        files={name:file_hash(directory/name) for name in names})
    publish(directory/'result.json',canonical(payload));return payload


def verify_payload(run,receipt):
    req=request(run);path=safe_artifact(run,receipt['result_path'])
    require(receipt['request_sha256']==file_hash(run/'request.json') and receipt['result_sha256']==file_hash(path)
            and receipt['elapsed_seconds']<LIMITS['seconds'],'Native completion drift/deadline')
    result=read(path);directory=safe_artifact(run,result['artifact_directory'])
    require(result['request_sha256']==file_hash(run/'request.json') and result['stage']==req['definition']['stage']
            and result['status']=='COMPLETE' and result['validation_test_read'] is False,'Native result differs')
    expected={'raw.npy','labels.npy','discrete.npy','preprocessor.joblib'} if result['stage']=='prepare' else {'model.joblib'}
    require(set(result['files'])==expected,'Native checkpoint incomplete')
    for name,sha in result['files'].items():require(file_hash(safe_artifact(directory,name))==sha,'Native artifact drift')
    return {**result,'resources':receipt}


def verify_completed(run):
    return verify_payload(run,read(safe_artifact(run,'complete.json')))


def worker(run,attempt):
    req=request(run);directory=safe_artifact(run,'attempts/'+attempt);job=read(directory/'job.json');done=threading.Event()
    require(life.supervisor_alive(job['owner_pid'],job['owner_created']),'No live native TRAIN supervisor')
    threading.Thread(target=inputs.watch,args=(job['owner_pid'],job['owner_created'],req['deadline_epoch'],done),daemon=True).start()
    try:
        with file_lock(run/'.worker.guard'),(directory/'stages.jsonl').open('x') as stream:
            def emit(stage):stream.write(json.dumps({'stage':stage})+'\n');stream.flush()
            calculate(run,directory,emit);require(time.time()<req['deadline_epoch'],'Native TRAIN deadline exhausted')
    finally:done.set()


def execute(run,recover=False):
    with file_lock(run/'.pipeline.guard'):
        req=request(run)
        if (run/'complete.json').exists():return verify_completed(run)
        require(recover or not (run/'attempts').exists(),'Explicit recovery required')
        with file_lock(run/'.worker.guard'):pass
        budget=life.Budget(req)
        # A verified supervisor checkpoint may precede an interrupted final link.
        verified=[p for p in (run/'attempts').glob('*/supervisor.json') if read(p)['status']=='COMPLETE']
        require(len(verified)<=1,'Ambiguous completed native attempts')
        if verified:
            prior=read(verified[0]);verify_payload(run,prior)
            reason=policy.stop_reason(host.resources(run),None,psutil.Process().memory_info().rss,
                                      host.artifact_size(run),budget.elapsed(),limits=LIMITS)
            require(reason is None,reason)
            recovered={**prior,'original_verified_elapsed_seconds':prior['elapsed_seconds'],
                       'elapsed_seconds':budget.elapsed(),'recovered_supervisor_sha256':file_hash(verified[0])}
            publish(run/'complete.json',canonical(recovered));return verify_completed(run)
        directory=run/'attempts'/('a'+uuid.uuid4().hex);directory.mkdir(parents=True)
        publish(directory/'job.json',canonical({'owner_pid':os.getpid(),'owner_created':psutil.Process().create_time()}))
        process=tree=None;reason=None;peak=size_peak=0;minimum=None
        try:
            reason=policy.start_reason(host.resources(run),limits=LIMITS)
            reason=reason or policy.stop_reason(host.resources(run),None,0,host.artifact_size(run),budget.elapsed(),limits=LIMITS)
            if not reason:
                environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                with (directory/'worker.log').open('x') as log:
                    process=subprocess.Popen([sys.executable,'-B','-u','-m','experiments.literature.efc_train_probe',
                        '--_worker',str(run),'--attempt',directory.name],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
                    tree=host.ProcessTree(process.pid);last=None;last_time=-30
                    while True:
                        state=host.resources(run);rss=tree.rss();size=host.artifact_size(run);elapsed=budget.elapsed()
                        peak=max(peak,rss);size_peak=max(size_peak,size);minimum=min(minimum or state['available'],state['available'])
                        stage='starting'
                        try:stage=json.loads((directory/'stages.jsonl').read_text().splitlines()[-1])['stage']
                        except (OSError,IndexError,ValueError):pass
                        if stage!=last or elapsed-last_time>=30:
                            print(f'[{elapsed/60:.1f}/15 min] {stage} | {rss/2**30:.2f}/4 GiB',flush=True);last,last_time=stage,elapsed
                        reason=policy.stop_reason(state,None,rss,size,elapsed,limits=LIMITS)
                        if reason:break
                        if process.poll() is not None:
                            require(process.returncode==0 and not tree.live(),'Native worker failed; see worker.log');break
                        time.sleep(.3)
        except BaseException as exc:reason=f'{type(exc).__name__}: {exc}'
        finally:
            if tree is not None:tree.stop()
            if process is not None:process.wait(timeout=10)
        summary={'scope':SCOPE,'stage':req['definition']['stage'],'status':'STOPPED' if reason else 'COMPLETE','reason':reason,
            'elapsed_seconds':budget.elapsed(),'peak_tree_rss':peak,'minimum_available':minimum,'max_artifact_bytes':size_peak,
            'worker_exit':None if process is None else process.returncode,'run_id':run.name}
        if not reason:
            require((directory/'result.json').is_file(),'Missing native TRAIN result')
            summary.update(request_sha256=file_hash(run/'request.json'),result_path=(directory/'result.json').relative_to(run).as_posix(),
                           result_sha256=file_hash(directory/'result.json'))
        publish(directory/'supervisor.json',canonical(summary))
        if not reason:publish(run/'complete.json',canonical(summary));return verify_completed(run)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true');modes.add_argument('--fit-class',type=int,choices=range(6))
    modes.add_argument('--recover',type=Path);modes.add_argument('--resume',type=Path);modes.add_argument('--_worker',type=Path)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--attempt');args=parser.parse_args()
    if args._worker:worker(output_location(args._worker),args.attempt);return 0
    if args.resume:result=verify_completed(output_location(args.resume))
    else:
        if args.recover:run=output_location(args.recover)
        else:
            stage='prepare' if args.prepare else f'class{args.fit_class}'
            plan=definition(stage,args.prepared);run=inputs.OUTPUT/('efc-native-'+stage+'-'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz'))
            run.mkdir(parents=True);now=time.time();publish(run/'request.json',canonical({'definition':plan,
                'started_epoch':now,'deadline_epoch':now+LIMITS['seconds']}))
        print('Native TRAIN run: '+str(run),flush=True);result=execute(run,recover=args.recover is not None)
    print(json.dumps(result,indent=2));return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':sys.exit(main())
