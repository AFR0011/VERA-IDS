"""Sealed native-EFC fixed-case test evaluation; no validation tuning or refit.

Native training-derived cutoffs are retained. This is not a policy-matched
comparison to VERA. Completed test-part checkpoints are immutable and reusable.
"""
import argparse
from datetime import datetime,timezone
import json,os,sys,subprocess,threading,time,uuid
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq
import psutil
from . import efc_train_probe as train,efc_adapter as adapter
from experiments.c03.postseal.evaluation import PopulationHash
from experiments.c03.producer.validation import truth_labels
from experiments.c03.producer.plan import ROOT,canonical,digest,file_hash,require,output_location
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.artifacts import safe_artifact,array_hash,write_arrays
from experiments.c03.producer.locking import file_lock

SCOPE='EFC_NATIVE_FIXED_BOTNET_SEED123_FIDELITY_TEST'
FIT_RECORD=ROOT/'experiments/literature/EFC_NATIVE_FIT_SUCCESS.json'
C03_RESULT=ROOT/'experiments/c03/postseal/PILOT_SUCCESS.json'
C03_RUN=ROOT.parent/'VERA-IDS-local-runs/c03/postseal-20261118t055250590478z'
LIMITS=dict(train.LIMITS)
host,policy,life=train.host,train.policy,train.life


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(truth,prediction):
    require(truth.shape==prediction.shape and len(truth)>0 and np.issubdtype(truth.dtype,np.integer)
            and np.issubdtype(prediction.dtype,np.integer) and np.isin(truth,np.arange(7)).all()
            and np.isin(prediction,np.arange(7)).all(),'Invalid native prediction population')
    matrix=np.bincount(truth.astype(int)*7+prediction,minlength=49).reshape(7,7)
    support,predicted,tp=matrix.sum(axis=1),matrix.sum(axis=0),matrix.diagonal()
    f1=np.divide(2.*tp,support+predicted,out=np.zeros(7),where=support+predicted>0)
    rejected=prediction==6;benign=truth==0;unknown=truth==6;known=truth!=6;attacks=(truth>0)&known
    def rate(hit,population):
        n,d=int((hit&population).sum()),int(population.sum())
        return {'value':n/d if d else None,'numerator':n,'denominator':d,
                'undefined_reason':None if d else 'No supported true population'}
    return {'confusion_matrix':matrix.tolist(),'label_support':support.tolist(),'macro_f1':float(f1.mean()),
        'accuracy':float(tp.sum()/len(truth)),'udr':rate(rejected,unknown),'fur_known':rate(rejected,known),
        'fur_known_attacks':rate(rejected,attacks),'benign_family_fpr':rate((prediction>0)&(prediction<6),benign),
        'benign_rejection':rate(rejected,benign),'realized_rejection':rate(rejected,np.ones(len(truth),dtype=bool)),
        'auc_not_reported_reason':'Native multiclass EFC has no Stage-1 P(attack); no new energy-AUC definition is adopted'}


def definition():
    fits=read(FIT_RECORD);prior=read(C03_RESULT)
    require(fits['status']=='COMPLETE' and set(fits['classes'])==set(map(str,range(6))),'All native classes required')
    require(file_hash(C03_RUN/'contract.json')==prior['contract_sha256'],'Reference test contract drift')
    spec=read(C03_RUN/'contract.json')['test']
    paths=[Path(__file__),ROOT/'tests/test_efc_fidelity.py',Path(train.__file__),Path(adapter.__file__),
           ROOT/'experiments/c03/postseal/evaluation.py',ROOT/'experiments/c03/producer/validation.py',
           Path(host.__file__),Path(policy.__file__),Path(life.__file__),
           ROOT/'experiments/c03/producer/plan.py',ROOT/'experiments/c03/producer/checkpoints.py',
           ROOT/'experiments/c03/producer/artifacts.py',ROOT/'experiments/c03/producer/locking.py']
    return {'scope':SCOPE,'dataset':'CICIoT2023','holdout':'Botnet','seed':123,'limits':LIMITS,
        'fit_record_sha256':file_hash(FIT_RECORD),'fits':fits,'native':train.native_runtime(),
        'test':spec,'expected_population_sha256':prior['population_sha256'],'expected_truth_sha256':prior['truth_sha256'],
        'columns':read(ROOT/'experiments/literature/EFC_TRAIN_INPUT_SUCCESS.json')['columns'],
        'prediction_rule':'Native minimum class energy; first sorted class on ties; reject iff winning energy > own TRAIN cutoff',
        'unknown_mapping':{'native':-1,'canonical':6},'labels':['Benign','BruteForce','DDoS','DoS','Other','Scan/Recon','Unknown'],
        'selection':'None; no validation reads, tuning, recalibration or refitting',
        'comparison':'Standalone native operating point; not policy-matched to C03; complete-method differences only',
        'uncertainty':'Not estimated; one fixed holdout/seed, dependent prepared corpus',
        'sources':{p.relative_to(ROOT).as_posix():file_hash(p) for p in paths}}


def verify_fit_records(plan):
    fits=plan['fits'];prepared=train.inputs.OUTPUT/fits['prepared_run_id']
    require(file_hash(prepared/'complete.json')==fits['prepared_completion_sha256'],'Native preprocessing receipt drift')
    prep=train.verify_completed(prepared)
    for k,item in fits['classes'].items():
        run=train.inputs.OUTPUT/item['run_id']
        require(file_hash(run/'complete.json')==item['completion_sha256'],'Native class receipt drift')
        result=train.verify_completed(run)
        source_plan=train.request(run)['definition']
        require(result['class_label']==int(k) and result['files']['model.joblib']==item['model_sha256']
                and result['cutoff']==item['cutoff'] and result['rows']==prep['class_support'][int(k)]
                and result['max_bin']==prep['max_bin']
                and source_plan['prepared_run_id']==fits['prepared_run_id']
                and source_plan['prepared_completion_sha256']==fits['prepared_completion_sha256'],'Native fitted class differs')
    return prep


def verify_seal(run):
    req=read(safe_artifact(run,'request.json'));plan=req['definition']
    require(plan==definition() and req['deadline_epoch']==req['started_epoch']+LIMITS['seconds'],'Fidelity source/definition drift')
    expected={'request_sha256':file_hash(run/'request.json'),'definition_sha256':digest(plan),
              'fit_record_sha256':plan['fit_record_sha256'],'prediction_rule':plan['prediction_rule']}
    require(read(safe_artifact(run,'seal.json'))==expected,'Native policy seal differs')
    return req


def load_model(plan):
    prep=verify_fit_records(plan);train.native_runtime(load=True)
    root=train.inputs.OUTPUT/plan['fits']['prepared_run_id']
    preprocessor=joblib.load(safe_artifact(root,prep['artifact_directory'])/'preprocessor.joblib')
    estimators=[]
    for k in range(6):
        item=plan['fits']['classes'][str(k)];run=train.inputs.OUTPUT/item['run_id']
        result=train.verify_completed(run);directory=safe_artifact(run,result['artifact_directory'])
        estimators.append(joblib.load(directory/'model.joblib'))
    return adapter.assemble(preprocessor,estimators,np.arange(6),prep['max_bin'],prep['features'])


def score_frame(model,frame,columns):
    frame=frame.copy();frame.columns=[str(c).strip().lower().replace(' ','_') for c in frame.columns]
    require(frame.columns.is_unique and set(columns)<=set(frame.columns),'Test predictor schema differs')
    X=frame[columns].to_numpy(dtype=np.float64)
    require(np.isfinite(X).all(),'Nonfinite test values; no imputation policy adopted')
    truth=truth_labels(frame)
    pred,energy=model.predict(X,return_energies=True,unknown_class=True)
    require(pred.shape==energy.shape==truth.shape and np.issubdtype(pred.dtype,np.integer)
            and np.isin(pred,[-1,0,1,2,3,4,5]).all()
            and np.isfinite(energy).all(),'Invalid native test outputs')
    return truth,np.where(pred==-1,6,pred).astype(np.int8),energy


def verify_part(run,directory,entry):
    receipt=read(directory/'complete.json')
    require(receipt['seal_sha256']==file_hash(run/'seal.json') and receipt['source']==entry
            and file_hash(directory/'scores.npz')==receipt['scores_sha256'],'Native test part drift')
    with np.load(directory/'scores.npz',allow_pickle=False) as arrays:
        truth,pred,energy=arrays['truth'],arrays['prediction'],arrays['minimum_energy']
    require(len(truth)==receipt['rows'] and truth.shape==pred.shape==energy.shape and np.isfinite(energy).all(),
            'Invalid test part arrays')
    return receipt,truth,pred,energy


def calculate(run,emit):
    req=verify_seal(run);plan=req['definition'];emit('verify_native_fits_before_test')
    model=load_model(plan);spec=plan['test'];root=host.DATA_ROOT
    # This is the first resolution of test sources, after the policy/model seal.
    actual=sorted(p.relative_to(root).as_posix() for p in (root/'test').glob('*.parquet'))
    require(actual==[p['path'] for p in spec['parts']],'Native test inventory differs')
    folder=run/'parts';folder.mkdir(exist_ok=True);truths=[];predictions=[];energies=[];receipts=[]
    population=PopulationHash(spec['rows'],spec['parts'])
    for i,entry in enumerate(spec['parts']):
        emit(f'test_part_{i+1}_of_{len(spec["parts"])}')
        path=safe_artifact(root,entry['path']);require(file_hash(path)==entry['sha256'],'Native test source drift')
        directory=folder/f'p{i:04d}';directory.mkdir(exist_ok=True)
        if not (directory/'complete.json').exists():
            ys=[];ps=[];es=[];source=pq.ParquetFile(path)
            for batch in source.iter_batches(batch_size=spec['block_rows'],use_threads=False):
                y,p,e=score_frame(model,batch.to_pandas(),plan['columns']);ys.append(y);ps.append(p);es.append(e)
            y,p,e=np.concatenate(ys),np.concatenate(ps),np.concatenate(es)
            require(len(y)==source.metadata.num_rows and file_hash(path)==entry['sha256'],'Test changed/incomplete')
            attempt=directory/('a'+uuid.uuid4().hex);attempt.mkdir()
            write_arrays(attempt/'scores.npz',truth=y,prediction=p,minimum_energy=e)
            sha=file_hash(attempt/'scores.npz')
            if (directory/'scores.npz').exists():require(file_hash(directory/'scores.npz')==sha,'Interrupted test part differs')
            else:os.link(attempt/'scores.npz',directory/'scores.npz')
            publish(directory/'complete.json',canonical({'seal_sha256':file_hash(run/'seal.json'),'source':entry,
                'scores_sha256':sha,'rows':len(y)}))
        receipt,y,p,e=verify_part(run,directory,entry)
        population.add(entry,0,len(y));truths.append(y);predictions.append(p);energies.append(e)
        receipts.append({'directory':directory.relative_to(run).as_posix(),'sha256':file_hash(directory/'complete.json')})
    truth,pred,energy=np.concatenate(truths),np.concatenate(predictions),np.concatenate(energies)
    identity=population.finish();require(identity==plan['expected_population_sha256']
        and array_hash(truth)==plan['expected_truth_sha256'],'Test population/labels differ from fixed reference')
    result_metrics=metrics(truth,pred)
    require(min(result_metrics['label_support'])>=spec['minimum_per_class'],'Insufficient fixed-case support')
    require(verify_seal(run)==req,'Native seal changed during scoring');verify_fit_records(plan)
    result={'scope':SCOPE,'status':'COMPLETE','rows':len(truth),'population_sha256':identity,'truth_sha256':array_hash(truth),
        'prediction_sha256':array_hash(pred),'minimum_energy_sha256':array_hash(energy),'metrics':result_metrics,
        'seal_sha256':file_hash(run/'seal.json'),'parts':receipts,'comparison':plan['comparison'],'uncertainty':plan['uncertainty']}
    if (run/'evaluation.json').exists():require(read(run/'evaluation.json')==result,'Interrupted evaluation differs')
    else:publish(run/'evaluation.json',canonical(result))
    return result


def verify_completed(run):
    req=verify_seal(run);verify_fit_records(req['definition'])
    receipt=read(run/'complete.json');result=read(run/'evaluation.json')
    require(receipt['evaluation_sha256']==file_hash(run/'evaluation.json') and receipt['elapsed_seconds']<LIMITS['seconds']
            and receipt['status']==result['status']=='COMPLETE' and result['scope']==SCOPE
            and result['seal_sha256']==file_hash(run/'seal.json'),'Native evaluation completion drift')
    truths=[];predictions=[];energies=[];spec=req['definition']['test'];population=PopulationHash(spec['rows'],spec['parts'])
    for item,entry in zip(result['parts'],read(run/'request.json')['definition']['test']['parts'],strict=True):
        directory=safe_artifact(run,item['directory'])
        require(file_hash(directory/'complete.json')==item['sha256'],'Native completed test receipt drift')
        _,y,p,e=verify_part(run,directory,entry);truths.append(y);predictions.append(p);energies.append(e)
        population.add(entry,0,len(y))
    y,p,e=np.concatenate(truths),np.concatenate(predictions),np.concatenate(energies)
    require(population.finish()==result['population_sha256']==req['definition']['expected_population_sha256']
            and array_hash(y)==result['truth_sha256']==req['definition']['expected_truth_sha256']
            and array_hash(p)==result['prediction_sha256'] and array_hash(e)==result['minimum_energy_sha256']
            and metrics(y,p)==result['metrics'],'Native aggregate reconciliation failed')
    return {**result,'resources':receipt}


def worker(run,attempt):
    req=verify_seal(run);directory=safe_artifact(run,'attempts/'+attempt);job=read(directory/'job.json');done=threading.Event()
    require(life.supervisor_alive(job['owner_pid'],job['owner_created']),'No live fidelity supervisor')
    threading.Thread(target=train.inputs.watch,args=(job['owner_pid'],job['owner_created'],req['deadline_epoch'],done),daemon=True).start()
    try:
        with file_lock(run/'.worker.guard'),(directory/'stages.jsonl').open('x') as stream:
            def emit(stage):stream.write(json.dumps({'stage':stage})+'\n');stream.flush()
            calculate(run,emit);require(time.time()<req['deadline_epoch'],'Native evaluation deadline exhausted')
    finally:done.set()


def execute(run,recover=False):
    with file_lock(run/'.pipeline.guard'):
        req=verify_seal(run)
        if (run/'complete.json').exists():return verify_completed(run)
        require(recover or not (run/'attempts').exists(),'Explicit fidelity recovery required')
        with file_lock(run/'.worker.guard'):pass
        budget=life.Budget(req);directory=run/'attempts'/('a'+uuid.uuid4().hex);directory.mkdir(parents=True)
        publish(directory/'job.json',canonical({'owner_pid':os.getpid(),'owner_created':psutil.Process().create_time()}))
        process=tree=None;reason=None;peak=size_peak=0;minimum=None
        try:
            reason=policy.start_reason(host.resources(run),limits=LIMITS)
            reason=reason or policy.stop_reason(host.resources(run),None,0,host.artifact_size(run),budget.elapsed(),limits=LIMITS)
            if not reason:
                environment={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                with (directory/'worker.log').open('x') as log:
                    process=subprocess.Popen([sys.executable,'-B','-u','-m','experiments.literature.efc_fidelity','--_worker',
                        str(run),'--attempt',directory.name],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
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
                            require(process.returncode==0 and not tree.live(),'Native fidelity worker failed; see worker.log');break
                        time.sleep(.3)
        except BaseException as exc:reason=f'{type(exc).__name__}: {exc}'
        finally:
            if tree is not None:tree.stop()
            if process is not None:process.wait(timeout=10)
        summary={'scope':SCOPE,'status':'STOPPED' if reason else 'COMPLETE','reason':reason,'elapsed_seconds':budget.elapsed(),
            'peak_tree_rss':peak,'minimum_available':minimum,'max_artifact_bytes':size_peak,'run_id':run.name,
            'worker_exit':None if process is None else process.returncode}
        if not reason:summary['evaluation_sha256']=file_hash(run/'evaluation.json')
        publish(directory/'supervisor.json',canonical(summary))
        if not reason:publish(run/'complete.json',canonical(summary));return verify_completed(run)
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--evaluate',action='store_true');modes.add_argument('--recover',type=Path)
    modes.add_argument('--resume',type=Path);modes.add_argument('--_worker',type=Path);parser.add_argument('--attempt')
    args=parser.parse_args()
    if args._worker:worker(output_location(args._worker),args.attempt);return 0
    if args.resume:result=verify_completed(output_location(args.resume))
    else:
        if args.recover:run=output_location(args.recover)
        else:
            plan=definition();verify_fit_records(plan)
            run=train.inputs.OUTPUT/('efc-native-fidelity-'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz'))
            run.mkdir(parents=True);now=time.time();publish(run/'request.json',canonical({'definition':plan,
                'started_epoch':now,'deadline_epoch':now+LIMITS['seconds']}))
            publish(run/'seal.json',canonical({'request_sha256':file_hash(run/'request.json'),'definition_sha256':digest(plan),
                'fit_record_sha256':plan['fit_record_sha256'],'prediction_rule':plan['prediction_rule']}))
        print('Native fidelity run: '+str(run),flush=True);result=execute(run,recover=args.recover is not None)
    print(json.dumps({k:v for k,v in result.items() if k!='parts'},indent=2));return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':sys.exit(main())
