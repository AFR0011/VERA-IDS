"""Bounded post-seal pipeline. --recover retains original deadline and checkpoints."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from . import contract as c, selection, evaluation
from experiments.c03.pilot import contract as producer
from experiments.c03.engineering import lifecycle_probe as old, resource_policy as policy
from experiments.c03.engineering import preparation_probe as host, xgb_full_row_sizing_probe as gpu
from experiments.c03.producer.plan import ROOT, OUTPUT_ROOT, canonical, digest, file_hash, require
from experiments.c03.producer.artifacts import safe_artifact
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.locking import file_lock


def request(store):
    value = store.read('request.json')
    require(value['contract_sha256'] == digest(store.plan) and value['limits'] == c.LIMITS
            and value['deadline_epoch'] == value['started_epoch']+c.LIMITS['seconds'], 'Execution request drift')
    return value


def watch(pid, created, budget, done):
    while not done.wait(.3):
        if not old.supervisor_alive(pid, created) or budget.elapsed() >= c.LIMITS['seconds']:
            os._exit(15)


def worker(run, stage, attempt):
    store = c.open_store(run)
    req = request(store)
    directory = safe_artifact(run, f'attempts/{stage}/{attempt}')
    job = store.read((directory/'job.json').relative_to(run).as_posix())
    require(job['request_sha256'] == file_hash(run/'request.json') and job['stage'] == stage
            and old.supervisor_alive(job['owner_pid'], job['owner_created']), 'Worker owner/request differs')
    budget, done = old.Budget(req), threading.Event()
    threading.Thread(target=watch, args=(job['owner_pid'], job['owner_created'], budget, done), daemon=True).start()
    try:
        with file_lock(run/'.worker.guard'), (directory/'progress.jsonl').open('x') as stream:
            def emit(message):
                stream.write(json.dumps({'stage': message, 'seconds': budget.elapsed()})+'\n')
                stream.flush()
            require(budget.elapsed() < c.LIMITS['seconds'], 'Expired worker request')
            if stage == 'selection':
                selection.select(store, emit)
            elif stage == 'analysis':
                evaluation.analyze(store, emit)
            else:
                evaluation.score_component(store, stage, directory, emit)
            require(budget.elapsed() < c.LIMITS['seconds'], 'Deadline reached')
            emit('stage_complete')
    finally:
        done.set()


def completed(store, stage):
    if stage == 'selection':
        if not (store.path/'selection-seal.json').exists(): return False
        selection.verify_seal(store)
    elif stage == 'analysis':
        if not (store.path/'evaluation-complete.json').exists(): return False
        evaluation.verify_result(store)
    elif evaluation.completed_component(store, stage) is None:
        return False
    return True


def command(run, stage, attempt):
    return [sys.executable, '-B', '-u', '-m', 'experiments.c03.postseal.supervisor',
            '--_worker', str(run), '--stage', stage, '--attempt', attempt]


def run_stage(store, stage, req, budget):
    import psutil
    directory = store.path/'attempts'/stage/('a'+uuid.uuid4().hex)
    directory.mkdir(parents=True)
    publish(directory/'job.json', canonical({'request_sha256': file_hash(store.path/'request.json'),
        'stage': stage, 'owner_pid': os.getpid(), 'owner_created': psutil.Process().create_time()}))
    process = tree = None
    reason, peak, gpu_peak, max_size, minimum = None, 0, 0, 0, None
    try:
        telemetry, state = gpu.query_gpu(), host.resources(store.path)
        require(gpu.gpu_identity(telemetry) == req['gpu'], 'GPU identity differs')
        reason = policy.start_reason(state, telemetry if '_xgb_' in stage else None, limits=c.LIMITS)
        reason = reason or policy.stop_reason(state, telemetry, 0, host.artifact_size(store.path), budget.elapsed(), limits=c.LIMITS)
        if not reason:
            with (directory/'worker.log').open('x') as log, (directory/'monitor.jsonl').open('x') as trace:
                process = subprocess.Popen(command(store.path, stage, directory.name), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                tree = host.ProcessTree(process.pid)
                last_gpu, last_report, previous = time.monotonic(), -100., None
                while True:
                    state, rss, size = host.resources(store.path), tree.rss(), host.artifact_size(store.path)
                    if time.monotonic()-last_gpu >= c.LIMITS['gpu_interval']:
                        telemetry = gpu.query_gpu()
                        require(gpu.gpu_identity(telemetry) == req['gpu'], 'GPU identity differs')
                        last_gpu = time.monotonic()
                    elapsed = budget.elapsed()
                    peak, gpu_peak, max_size = max(peak,rss), max(gpu_peak,telemetry['used']), max(max_size,size)
                    minimum = state['available'] if minimum is None else min(minimum,state['available'])
                    trace.write(json.dumps({'seconds':elapsed,'rss':rss,'artifact_bytes':size,'gpu':telemetry,**state})+'\n')
                    trace.flush()
                    progress = stage+':verifying_dependencies'
                    try:
                        text = (directory/'progress.jsonl').read_text()
                        lines = text[:text.rfind('\n')+1].splitlines()
                        if lines: progress = json.loads(lines[-1])['stage']
                    except (OSError,ValueError,KeyError): pass
                    if previous != progress or elapsed-last_report >= 30:
                        print(f'[{elapsed/60:.1f}/60 min] {progress} | worker {rss/2**30:.2f}/4 GiB | GPU {telemetry["used"]/2**30:.2f}/4 GiB', flush=True)
                        previous,last_report = progress,elapsed
                    reason = policy.stop_reason(state,telemetry,rss,size,elapsed,limits=c.LIMITS)
                    if reason: break
                    if process.poll() is not None:
                        require(not tree.live() and process.returncode == 0, f'Worker failed ({process.returncode}); see worker.log')
                        require(completed(store,stage),'Worker completion missing')
                        break
                    time.sleep(c.LIMITS['interval'])
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            if tree is not None: tree.stop()
            elif process is not None and process.poll() is None: host.ProcessTree(process.pid).stop()
            if process is not None: process.wait(timeout=10)
        except Exception as exc: reason = f'Cleanup failure: {exc}; prior={reason}'
    if not reason:
        try:
            telemetry = gpu.query_gpu()
            require(gpu.gpu_identity(telemetry) == req['gpu'], 'GPU identity differs')
            reason = policy.stop_reason(host.resources(store.path),telemetry,0,host.artifact_size(store.path),budget.elapsed(),limits=c.LIMITS)
        except Exception as exc: reason = f'Final monitor failure: {exc}'
    result = {'stage':stage,'status':'STOPPED' if reason else 'COMPLETE','reason':reason,
        'peak_tree_rss':peak,'peak_gpu_used':gpu_peak,'max_artifact_bytes':max_size,'minimum_available':minimum,
        'elapsed_seconds':budget.elapsed(),'worker_exit':None if process is None else process.returncode,
        'attempt':directory.relative_to(store.path).as_posix()}
    publish(directory/'supervisor.json',canonical(result))
    return result


def execute(store, recover=False):
    with file_lock(store.path/'.pipeline.guard'):
        req = request(store)
        if (store.path/'complete.json').exists(): return verify_completed(store)
        budget = old.Budget(req)
        require(budget.elapsed()<c.LIMITS['seconds'],'Original one-hour budget exhausted')
        with file_lock(store.path/'.worker.guard'): pass
        require(recover or not (store.path/'attempts').exists(),'Explicit recovery required')
        stages, reused, reason = {}, [], None
        try:
            queue = ['selection']
            while queue:
                stage = queue.pop(0)
                if completed(store,stage):
                    reused.append(stage)
                    print('Verified checkpoint reused: '+stage,flush=True)
                else:
                    stages[stage] = run_stage(store,stage,req,budget)
                    require(stages[stage]['status']=='COMPLETE',stages[stage]['reason'])
                require(budget.elapsed()<c.LIMITS['seconds'],'Shared deadline reached')
                if stage == 'selection':
                    queue = selection.selected_components(store,selection.verify_seal(store))+['analysis']
            evaluation.verify_result(store)
            telemetry=gpu.query_gpu()
            require(gpu.gpu_identity(telemetry)==req['gpu'],'GPU identity differs at completion')
            require(policy.stop_reason(host.resources(store.path),telemetry,0,
                    host.artifact_size(store.path),budget.elapsed(),limits=c.LIMITS) is None,
                    'Resource/deadline guard reached at completion')
        except BaseException as exc: reason = f'{type(exc).__name__}: {exc}'
        summary = {'scope':c.SCOPE,'status':'STOPPED' if reason else 'COMPLETE','reason':reason,
            'elapsed_seconds':budget.elapsed(),'stages':stages,'reused_stages':reused,'run_directory':str(store.path)}
        folder = store.path/'summaries'; folder.mkdir(exist_ok=True)
        path = folder/('a'+uuid.uuid4().hex+'.json')
        publish(path,canonical(summary))
        if reason is None:
            publish(store.path/'complete.json',canonical({'contract_sha256':digest(store.plan),
                'evaluation_receipt_sha256':file_hash(store.path/'evaluation-complete.json'),
                'summary_path':path.relative_to(store.path).as_posix(),'summary_sha256':file_hash(path)}))
        return summary


def verify_completed(store):
    evaluation.verify_result(store)
    receipt = store.read('complete.json')
    require(receipt['contract_sha256']==digest(store.plan)
            and receipt['evaluation_receipt_sha256']==file_hash(store.path/'evaluation-complete.json')
            and receipt['summary_sha256']==file_hash(safe_artifact(store.path,receipt['summary_path'])),'Completion drift')
    summary=store.read(receipt['summary_path'])
    require(summary['status']=='COMPLETE' and summary['elapsed_seconds']<c.LIMITS['seconds'],'Unsuccessful supervision')
    return summary


def start(source=None,recover=None):
    telemetry = policy.query_gpu_start()
    require(gpu.gpu_identity(telemetry)==gpu.load_readiness()['selected_gpu'],'Unqualified GPU')
    reason=policy.start_reason(host.resources(OUTPUT_ROOT),telemetry,limits=c.LIMITS)
    if reason: return {'status':'START_REFUSED','reason':reason}
    if recover:
        store=c.open_store(recover)
    else:
        name='postseal-'+datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%fz')
        store=c.create(OUTPUT_ROOT/name,producer.open_store(source))
        started=time.time()
        publish(store.path/'request.json',canonical({'contract_sha256':digest(store.plan),'limits':c.LIMITS,
            'started_epoch':started,'deadline_epoch':started+c.LIMITS['seconds'],'gpu':gpu.gpu_identity(telemetry)}))
    print('Post-seal run: '+str(store.path),flush=True)
    return execute(store,recover=recover is not None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run',type=Path,help='Completed scientific producer directory')
    modes.add_argument('--recover',type=Path)
    modes.add_argument('--resume',type=Path)
    modes.add_argument('--_worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--stage',choices=('selection','analysis',*producer.COMPONENTS),help=argparse.SUPPRESS)
    parser.add_argument('--attempt',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args._worker:
        worker(args._worker,args.stage,args.attempt); return 0
    result=verify_completed(c.open_store(args.resume)) if args.resume else start(args.run,args.recover)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='COMPLETE' else 3


if __name__=='__main__': sys.exit(main())
