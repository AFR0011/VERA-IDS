"""Authorized 15-minute DOC TRAIN/known-validation preparation and one-epoch probe."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

from . import doc_runtime as rt, doc_preparation as prep, efc_input_probe as inputs
from experiments.c03.producer.plan import ROOT, canonical, file_hash, require, output_location
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.locking import file_lock
from experiments.c03.engineering import preparation_probe as host, resource_policy as policy
from experiments.c03.engineering.cuda_readiness_probe import query_gpu
from experiments.c03.engineering.lifecycle_probe import Budget

LIMITS=dict(policy.LIMITS)
TRAIN_RUN=inputs.OUTPUT/'efc-train-inputs-20261118t170122755261z'
PYTHON=ROOT/'.release-audit/doc-env/Scripts/python.exe'


def definition():
    source=rt.read(inputs.REFERENCE/'prepared/complete.json')
    imported=rt.read(ROOT/'experiments/literature/EFC_TRAIN_INPUT_SUCCESS.json')
    require(source['plan_sha256']=='6b54cfd10448ff4f4cf9cc601e3269debe14f95e78999b0378ffba955d057bdd','Wrong C03 case')
    require(file_hash(TRAIN_RUN/'complete.json')==imported['completion_sha256'],'Wrong EFC input run')
    paths=[Path(__file__),Path(prep.__file__),Path(rt.__file__),Path(inputs.__file__),Path(host.__file__),Path(policy.__file__),
           ROOT/'experiments/literature/doc_core.py',ROOT/'experiments/literature/doc_training.py',
           ROOT/'experiments/literature/doc_readiness_fit.py',ROOT/'experiments/literature/doc-requirements.txt',
           ROOT/'tests/test_doc_readiness.py',ROOT/'tests/test_doc_readiness_fit.py',
           ROOT/'experiments/c03/producer/artifacts.py',ROOT/'experiments/c03/producer/checkpoints.py',
           ROOT/'experiments/c03/producer/plan.py',ROOT/'experiments/c03/producer/locking.py',
           ROOT/'experiments/c03/producer/validation.py',ROOT/'experiments/c03/engineering/cuda_readiness_probe.py',
           ROOT/'experiments/c03/engineering/lifecycle_probe.py',
           ROOT/'docs/RECKONING_DOC_TABULAR_CONTRACT_DRAFT.md',ROOT/'pyproject.toml',ROOT/'uv.lock']
    return {'columns':imported['columns'],'sources':{p.relative_to(ROOT).as_posix():file_hash(p) for p in paths},
            'train':{'run_id':TRAIN_RUN.name,'completion_sha256':imported['completion_sha256'],
                     'identity_sha256':imported['identity_sha256'],'class_support':imported['class_support']},
            'reference_completion_sha256':file_hash(inputs.REFERENCE/'prepared/complete.json'),
            'validation':{'array_sha256':source['files']['prepared/validation.npz'],
                          'identity_sha256':source['validation']['rows'],'truth_sha256':source['validation']['truth'],
                          'inventory':source['source_inventory']['val'],'expected_known':599278},
            'limits':LIMITS,'approved_scope':'Full known inputs, TRAIN scaler, one epoch, validation BCE and reload only',
            'fit':{'model_seed':123,'batch_size':128,'epochs':1,'device':'cuda:0','cpu_threads':2},
            'cutoffs_fitted':False,'test_read':False}


def verify_request(run):
    plan=rt.request(run)
    require(all(plan[k]==v for k,v in definition().items()),'Readiness definition drift')
    return plan


def verify_completed(run):
    plan=verify_request(run);result=rt.read(run/'complete.json')
    require(result['scope']==rt.SCOPE and result['status']=='COMPLETE'
            and result['request_sha256']==file_hash(run/'request.json'), 'Completion drift')
    for name,sha in result['artifacts'].items():require(file_hash(run/name)==sha,'Completed receipt drift')
    prepared=rt.read(run/'prepare/complete.json');rt.verify_files(run/'prepare',prepared)
    rt.verify_journal(run,result['fit']['journal_commit'])
    require(prepared['request_sha256']==result['request_sha256'],'Preparation binding drift')
    require(result['elapsed_seconds']<900 and result['fit']['epochs']==1 and result['fit']['reload_prediction_exact'], 'Incomplete readiness')
    return result


def execute(run, recover=False):
    run=Path(run)
    with file_lock(run/'.supervisor.guard'):
        plan=verify_request(run)
        if (run/'complete.json').exists():return verify_completed(run)
        require(recover or not (run/'attempts').exists(),'Explicit recovery required')
        rt.guard(plan)
        budget=Budget({'started_epoch':plan['started']})
        gpu=policy.query_gpu_start()
        hardware={k:gpu[k] for k in ('index','name','total','driver')}
        require(policy.start_reason(host.resources(run),gpu,limits=LIMITS) is None,'Disk/GPU start gate failed')
        with file_lock(run/'.worker.guard'):pass
        records=[]
        for stage in ('prepare','fit'):
            rt.guard(plan)
            attempt='a'+uuid.uuid4().hex;directory=run/'attempts'/attempt;directory.mkdir(parents=True)
            publish(directory/'owner.json',canonical({'pid':os.getpid(),'created':psutil.Process().create_time()}))
            command=([sys.executable,'-B','-u','-m','experiments.literature.doc_readiness','--_prepare',str(run),'--attempt',attempt]
                     if stage=='prepare' else [str(PYTHON),'-B','-u','-m','experiments.literature.doc_readiness_fit','--run',str(run),'--attempt',attempt])
            env=os.environ.copy();env['PYTHONPATH']=str(ROOT);env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            process=tree=None;reason=None;peak=peak_gpu=size_peak=0;minimum=None;last=-30;last_stage=None
            try:
                with (directory/'worker.log').open('x') as log:
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    tree=host.ProcessTree(process.pid)
                    while True:
                        elapsed=budget.elapsed();state=host.resources(run);rss=tree.rss();size=host.artifact_size(run)
                        peak=max(peak,rss);size_peak=max(size_peak,size);minimum=min(minimum or state['available'],state['available'])
                        gpu=query_gpu();peak_gpu=max(peak_gpu,gpu['used'])
                        require({k:gpu[k] for k in hardware}==hardware,'GPU identity changed during readiness')
                        current={'stage':stage}
                        try:current=json.loads((directory/'progress.jsonl').read_text().splitlines()[-1])
                        except (OSError,IndexError,ValueError):pass
                        if current!=last_stage or elapsed-last>=30:
                            print(f'[{elapsed/60:.1f}/15 min] {current} | worker {rss/2**30:.2f} GiB | GPU {gpu["used"]/2**30:.2f} GiB',flush=True)
                            last,last_stage=elapsed,current
                        reason=policy.stop_reason(state,gpu,rss,size,elapsed,limits=LIMITS)
                        if reason or process.poll() is not None:break
                        time.sleep(.3)
            finally:
                if tree is not None:tree.stop()
                if process is not None:process.wait(timeout=10)
            record={'stage':stage,'attempt':attempt,'worker_exit':process.returncode,'reason':reason,
                    'peak_tree_rss':peak,'peak_gpu_used':peak_gpu,'minimum_available':minimum,'peak_artifacts':size_peak,
                    'batch_elapsed_seconds':budget.elapsed()}
            publish(directory/'result.json',canonical(record));records.append(record)
            if reason or process.returncode!=0:
                return {'scope':rt.SCOPE,'status':'STOPPED','reason':reason or 'Worker failed; inspect attempt log',
                        'run_directory':str(run),'attempts':records}
        verify_request(run);rt.guard(plan)
        prepared=rt.read(run/'prepare/complete.json');rt.verify_files(run/'prepare',prepared)
        fit=rt.read(run/'fit-result.json')
        require(fit['request_sha256']==file_hash(run/'request.json') and fit['prepared_sha256']==file_hash(run/'prepare/complete.json'), 'Fit binding drift')
        rt.verify_journal(run,fit['journal_commit'])
        require(budget.elapsed()<900,'Shared deadline exhausted before completion')
        result={'scope':rt.SCOPE,'status':'COMPLETE','run_id':run.name,'request_sha256':file_hash(run/'request.json'),
                'elapsed_seconds':budget.elapsed(),'attempts':records,'fit':fit,'hardware':hardware,
                'artifacts':{p:file_hash(run/p) for p in ('prepare/complete.json','fit-result.json')},
                'cutoffs_fitted':False,'test_read':False}
        publish(run/'complete.json',canonical(result))
        return verify_completed(run)


def main():
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--probe',action='store_true');modes.add_argument('--recover',type=Path)
    modes.add_argument('--verify',type=Path);modes.add_argument('--_prepare',type=Path);parser.add_argument('--attempt')
    args=parser.parse_args()
    if args._prepare:
        plan,done=rt.supervise_owner(args._prepare,args.attempt)
        try:
            with file_lock(args._prepare/'.worker.guard'):
                prep.prepare(args._prepare,plan,emit=lambda stage:rt.emit(args._prepare,args.attempt,stage))
        finally:done.set()
        return
    if args.verify:result=verify_completed(output_location(args.verify))
    elif args.recover:result=execute(output_location(args.recover),True)
    else:
        plan=definition();run=inputs.OUTPUT/('doc-readiness-'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz'))
        run.mkdir(parents=True);now=time.time()
        publish(run/'request.json',canonical({'scope':rt.SCOPE,'started':now,'deadline':now+900,**plan}))
        print('DOC readiness run: '+str(run),flush=True);result=execute(run)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
