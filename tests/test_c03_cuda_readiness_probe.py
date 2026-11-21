"""CPU-generated fits and mocked GPU telemetry; never opens benchmark arrays."""
import copy
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from experiments.c03.engineering import cuda_readiness_probe as cuda
from experiments.c03.producer.plan import build_plan, canonical, file_hash
from experiments.c03.producer.locking import file_lock


def gpu():
    return {'index':0,'name':'fixture GPU','total':8*cuda.GIB,'used':cuda.GIB//2,
            'free':7*cuda.GIB,'driver':'fixture'}


def host():
    return {'available':5*cuda.GIB,'disk_free':20*cuda.GIB}


def config(name,device='cuda:0'):
    return {'learner':{'generic_param':{'device':device,'nthread':'2'},
        'learner_model_param':{'num_feature':'46','num_class':'0' if name==cuda.COMPONENTS[0] else '5'},
        'objective':{'name':'binary:logistic' if name==cuda.COMPONENTS[0] else 'multi:softprob'},
        'gradient_booster':{'gbtree_train_param':{'tree_method':'hist'}}}}


def test_gpu_parser_units_and_reserved_memory():
    reading=cuda.parse_gpu('0, "GPU, fixture", 8151, 249, 7651, 610.88\n')
    assert reading['name']=='GPU, fixture'
    assert reading['total']==8151*1024**2
    assert reading['used']+reading['free'] < reading['total']


@pytest.mark.parametrize('text', ['', '0,GPU,8000,N/A,7000,driver', '1,GPU,8000,500,7000,driver',
    '0,GPU,8000,500,9000,driver','0,GPU,8000,2000,7000,driver',
    '0,GPU,8000,500,7000,driver\n1,GPU,8000,500,7000,driver'])
def test_invalid_gpu_readings_fail_closed(text):
    with pytest.raises(ValueError):
        cuda.parse_gpu(text)


def test_gpu_query_has_timeout_and_rejects_remapping(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    calls=[]
    def run(command,**kwargs):
        calls.append((command,kwargs))
        return SimpleNamespace(stdout='0,GPU,8000,500,7000,driver')
    monkeypatch.setattr(cuda.subprocess,'run',run)
    assert cuda.query_gpu()['index']==0
    assert calls[0][1]['timeout']==2 and calls[0][1]['check'] is True
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    with pytest.raises(ValueError,match='remapping'):
        cuda.query_gpu()


@pytest.mark.parametrize('key,value,expected', [
    ('available',2*cuda.GIB,'3 GiB'),('disk_free',cuda.GIB,'2 GiB'),('free',5*cuda.GIB,'6 GiB')])
def test_start_limits(key,value,expected):
    state,device=host(),gpu()
    (state if key in state else device)[key]=value
    assert expected in cuda.start_reason(state,device)


@pytest.mark.parametrize('change,expected', [
    ({'elapsed':300},'300-second'),({'rss':2*cuda.GIB+1},'RSS'),
    ({'available':2*cuda.GIB-1},'Available RAM'),({'disk_free':cuda.GIB-1},'Free disk'),
    ({'size':64*1024**2+1},'artifacts'),({'used':4*cuda.GIB+1},'device-used'),
    ({'free':2*cuda.GIB-1},'Free VRAM')])
def test_stop_limits(change,expected):
    state,device,args=host(),gpu(),{'rss':0,'size':0,'elapsed':0}
    for key,value in change.items():
        (args if key in args else state if key in state else device)[key]=value
    assert expected in cuda.stop_reason(state,device,**args)


@pytest.mark.parametrize('name',cuda.COMPONENTS)
def test_real_cpu_generated_objective_preserves_parameters_without_data_or_model_io(monkeypatch,name):
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import joblib
    def forbidden(*_a,**_k):
        pytest.fail('Unexpected benchmark/scoring/model I/O')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(pd,'read_parquet',forbidden)
    monkeypatch.setattr(joblib,'load',forbidden)
    monkeypatch.setattr(joblib,'dump',forbidden)
    monkeypatch.setattr(xgb.XGBClassifier,'predict',forbidden)
    monkeypatch.setattr(xgb.XGBClassifier,'predict_proba',forbidden)
    monkeypatch.setattr(xgb.Booster,'save_model',forbidden)
    plan=build_plan()
    original=copy.deepcopy(plan)
    stages=[]
    result=cuda.fit_generated(plan,name,stages.append,device='cpu')
    assert plan==original and result['rows']==4096 and result['features']==46
    assert result['boosted_rounds']==8 and result['fit_seconds']>0
    assert result['effective_parameters']['n_jobs']==2
    assert result['effective_parameters']['n_estimators']==8
    assert result['effective_parameters']['random_state']==(123 if name==cuda.COMPONENTS[0] else 133)
    assert result['model_retained'] is False and result['quality_metrics'] is None
    assert [x for x in stages if x.startswith('round_')]==[f'round_{i}_of_8' for i in range(1,9)]
    with pytest.raises(ValueError,match='CPU fallback'):
        cuda.check_booster(result['booster_config'],name)


@pytest.mark.parametrize('change', ['device','objective','classes','threads','features','method'])
def test_fitted_configuration_rejection(change):
    name=cuda.COMPONENTS[1]
    value=config(name)
    learner=value['learner']
    if change=='device': learner['generic_param']['device']='cuda:1'
    if change=='objective': learner['objective']['name']='binary:logistic'
    if change=='classes': learner['learner_model_param']['num_class']='4'
    if change=='threads': learner['generic_param']['nthread']='24'
    if change=='features': learner['learner_model_param']['num_feature']='45'
    if change=='method': learner['gradient_booster']['gbtree_train_param']['tree_method']='exact'
    with pytest.raises(ValueError):
        cuda.check_booster(value,name)


def fake_command(run,name,*,fail=False,sleep=0,cpu=False):
    result={'scope':cuda.SCOPE,'status':'CUDA_COMPONENT_COMPLETE','component':name,'scientific_checkpoint':False,
        'rows':4096,'features':46,'boosted_rounds':8,'model_retained':False,'quality_metrics':None,
        'input_kind':'GENERATED_ONLY','booster_config':config(name,'cpu' if cpu else 'cuda:0')}
    code='''import json,sys,time
from pathlib import Path
p=Path(sys.argv[1])
(p/'worker.guard').write_bytes(b'0')
(p/'stages.jsonl').write_text(json.dumps({'stage':'fixture'})+'\\n')
time.sleep(float(sys.argv[2]))
if sys.argv[3]=='yes': raise RuntimeError('fixture interruption')
(p/'worker-result.json').write_text(sys.argv[4])
'''
    return [sys.executable,'-B','-u','-c',code,str(run/'profiles'/name),str(sleep),'yes' if fail else 'no',json.dumps(result)]


def fixture_run(tmp_path,monkeypatch):
    monkeypatch.setattr(cuda,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(cuda.host,'resources',lambda _:host())
    monkeypatch.setattr(cuda,'query_gpu',gpu)
    run=cuda.new_run()
    (run/'request.json').write_bytes(canonical({**cuda.request_record(),'selected_gpu':cuda.gpu_identity(gpu())}))
    return run


def test_actual_sequential_processes_share_deadline_and_completed_resume_needs_no_gpu(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(cuda,'command',fake_command)
    starts=[]
    original=cuda.run_profile
    def recorded(run,name,start,selected):
        starts.append(start)
        return original(run,name,start,selected)
    monkeypatch.setattr(cuda,'run_profile',recorded)
    result=cuda.execute(run)
    assert result['status']=='COMPLETE' and starts[0]==starts[1]
    monkeypatch.setattr(cuda,'command',lambda *_:pytest.fail('Worker on resume'))
    monkeypatch.setattr(cuda,'query_gpu',lambda:pytest.fail('GPU on completed resume'))
    assert cuda.verify_completed(run)['reused_completed_result']


def test_partial_recovery_reuses_first_and_preserves_failed_attempt(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(cuda,'command',lambda r,n:fake_command(r,n,fail=n==cuda.COMPONENTS[1]))
    result=cuda.execute(run)
    assert result['status']=='STOPPED' and result['completed_components']==[cuda.COMPONENTS[0]]
    before={str(p.relative_to(run)):file_hash(p) for p in run.rglob('*') if p.is_file()}
    target=cuda.fork_completed(run,cuda.gpu_identity(gpu()))
    calls=[]
    def command(r,n):
        calls.append(n)
        return fake_command(r,n)
    monkeypatch.setattr(cuda,'command',command)
    assert cuda.execute(target)['status']=='COMPLETE' and calls==[cuda.COMPONENTS[1]]
    assert {str(p.relative_to(run)):file_hash(p) for p in run.rglob('*') if p.is_file()}==before
    assert cuda.verify_completed(target)['reused_components']==[cuda.COMPONENTS[0]]


def test_cpu_fallback_fails_first_component_and_skips_second(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    calls=[]
    def command(r,n):
        calls.append(n)
        return fake_command(r,n,cpu=True)
    monkeypatch.setattr(cuda,'command',command)
    result=cuda.execute(run)
    assert result['status']=='STOPPED' and 'CPU fallback' in result['reason']
    assert calls==[cuda.COMPONENTS[0]] and not (run/'complete.json').exists()


@pytest.mark.parametrize('failure',['timeout','memory','identity'])
def test_gpu_monitor_failure_stops_live_worker(tmp_path,monkeypatch,failure):
    run=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(cuda,'command',lambda r,n:fake_command(r,n,sleep=30))
    count=0
    def query():
        nonlocal count
        count+=1
        reading=gpu()
        if count>1:
            if failure=='timeout': raise subprocess.TimeoutExpired('nvidia-smi',2)
            if failure=='memory': reading.update(used=5*cuda.GIB,free=3*cuda.GIB)
            if failure=='identity': reading['name']='other GPU'
        return reading
    monkeypatch.setattr(cuda,'query_gpu',query)
    result=cuda.run_profile(run,cuda.COMPONENTS[0],time.monotonic(),cuda.gpu_identity(gpu()))
    assert result['status']=='STOPPED' and result['worker_exit'] is not None
    assert not (run/'profiles'/cuda.COMPONENTS[0]/'complete.json').exists()


def test_expired_budget_prevents_launch(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(cuda,'command',lambda *_:pytest.fail('Started after deadline'))
    result=cuda.run_profile(run,cuda.COMPONENTS[0],time.monotonic()-301,cuda.gpu_identity(gpu()))
    assert result['status']=='STOPPED' and '300-second' in result['reason']


def test_second_component_checks_start_gate_again(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    calls=[]
    def command(r,n):
        calls.append(n)
        return fake_command(r,n)
    def resources(_):
        reading=host()
        if (run/'profiles'/cuda.COMPONENTS[0]/'complete.json').exists():
            reading['available']=int(2.5*cuda.GIB)
        return reading
    monkeypatch.setattr(cuda,'command',command)
    monkeypatch.setattr(cuda.host,'resources',resources)
    assert cuda.execute(run)['status']=='STOPPED'
    assert calls==[cuda.COMPONENTS[0]]


def test_live_locks_corrupt_receipts_and_wrong_gpu_block_recovery(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(cuda,'command',fake_command)
    assert cuda.execute(run)['status']=='COMPLETE'
    profile=run/'profiles'/cuda.COMPONENTS[0]
    with file_lock(profile/'worker.guard'):
        with pytest.raises(FileExistsError): cuda.fork_completed(run,cuda.gpu_identity(gpu()))
    with pytest.raises(ValueError,match='GPU identity'):
        cuda.fork_completed(run,{**cuda.gpu_identity(gpu()),'driver':'changed'})
    (profile/'worker-result.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='drift'): cuda.verify_completed(run)
    with pytest.raises(ValueError,match='drift'): cuda.fork_completed(run,cuda.gpu_identity(gpu()))


def test_incomplete_recovery_and_changed_worker_request_refused(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    request=cuda.read(run/'request.json')
    request['recovery']='0'*64
    (run/'request.json').write_bytes(canonical(request))
    with pytest.raises(FileNotFoundError): cuda.verify_request(run)
    profile=run/'profiles'/cuda.COMPONENTS[0]
    profile.mkdir(parents=True)
    (profile/'request.json').write_bytes(canonical({}))
    monkeypatch.setattr(cuda,'fit_generated',lambda *_:pytest.fail('Fit after drift'))
    with pytest.raises(ValueError,match='source/request'): cuda.worker(run,cuda.COMPONENTS[0])


@pytest.mark.parametrize('failure',['ram','gpu_query'])
def test_probe_refuses_before_new_run_or_fit(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(cuda,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(cuda.host,'resources',lambda _:{**host(),'available':cuda.GIB if failure=='ram' else 5*cuda.GIB})
    def query():
        if failure=='gpu_query': raise subprocess.TimeoutExpired('nvidia-smi',2)
        return gpu()
    monkeypatch.setattr(cuda,'query_gpu',query)
    monkeypatch.setattr(cuda,'execute',lambda *_:pytest.fail('Fit after gate refusal'))
    assert cuda.probe()['status']=='START_REFUSED'
    assert not list(tmp_path.iterdir())
