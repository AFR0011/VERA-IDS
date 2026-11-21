"""Generated TRAIN fixtures and mocked CUDA processes only; no benchmark reads."""
import copy
import json
import subprocess
import sys
import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from experiments.c03.engineering import xgb_full_row_sizing_probe as xgb
from experiments.c03.producer.artifacts import read_arrays, tensor_hash, array_hash
from experiments.c03.producer.plan import canonical, file_hash
from experiments.c03.producer.locking import file_lock
from test_c03_rf_stage1_sizing_probe import fixture_source
from test_c03_cuda_readiness_probe import gpu, config


def host():
    return {'available':8*xgb.GIB,'disk_free':30*xgb.GIB}


def training_source(tmp_path):
    source, success, plan, *_=fixture_source(tmp_path)
    prep_path=source/'diagnostic/prepared/complete.json'
    prep=xgb.read(prep_path)
    result=xgb.read(source/'worker-result.json')
    for stage,name,old_name in [('stage1',xgb.COMPONENTS[0],'stage1_rf_inv_family_clipped'),
                                ('stage2',xgb.COMPONENTS[1],'stage2_rf_shared')]:
        path=source/f'diagnostic/prepared/{stage}.npz'
        values=read_arrays(path)
        X=csr_matrix(np.random.RandomState(73).normal(size=(100,46)))
        values.update(data=X.data,indices=X.indices,indptr=X.indptr,shape=np.array(X.shape))
        values['weight_'+name]=values['weight_'+old_name].copy()
        with path.open('wb') as stream:
            np.savez(stream,**values)
        binding=prep['bindings'][stage]
        binding['tensor']=tensor_hash(X)
        binding['weights'][name]=array_hash(values['weight_'+name])
        result['identities']['bindings'][stage]=binding
        success['identities']['bindings'][stage]=binding
        prep['files'][f'prepared/{stage}.npz']=file_hash(path)
        result['files'][f'diagnostic/prepared/{stage}.npz']=file_hash(path)
    prep_path.write_bytes(canonical(prep))
    result['files']['diagnostic/prepared/complete.json']=file_hash(prep_path)
    (source/'worker-result.json').write_bytes(canonical(result))
    completion=xgb.read(source/'complete.json')
    completion['worker_result_sha256']=file_hash(source/'worker-result.json')
    (source/'complete.json').write_bytes(canonical(completion))
    success['evidence_hashes']={path:file_hash(source/path) for path in success['evidence_hashes']}
    return source,success,plan


@pytest.mark.parametrize('name',xgb.COMPONENTS)
def test_only_target_training_stage_is_read_and_cpu_fit_preserves_inputs(tmp_path,monkeypatch,name):
    source,success,plan=training_source(tmp_path)
    stage=plan['graph']['components'][name]['stage']
    allowed=source/f'diagnostic/prepared/{stage}.npz'
    original_hash=xgb.file_hash
    original_load=np.load
    def checked_hash(path):
        if str(path).endswith('.npz'): assert path==allowed
        return original_hash(path)
    def checked_load(path,*args,**kwargs):
        assert path==allowed
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(xgb,'file_hash',checked_hash)
    monkeypatch.setattr(np,'load',checked_load)
    imported=xgb.verify_import(source,success,plan,name)
    X,y,w=xgb.load_training(allowed,imported,name)
    assert array_hash(w)==imported['expected_binding']['weights'][name]
    import xgboost
    import joblib
    import pandas as pd
    def forbidden(*_a,**_k): pytest.fail('Unexpected raw/model/scoring access')
    monkeypatch.setattr(joblib,'load',forbidden)
    monkeypatch.setattr(joblib,'dump',forbidden)
    monkeypatch.setattr(pd,'read_parquet',forbidden)
    monkeypatch.setattr(xgboost.Booster,'save_model',forbidden)
    monkeypatch.setattr(xgboost.XGBClassifier,'predict',forbidden)
    monkeypatch.setattr(xgboost.XGBClassifier,'predict_proba',forbidden)
    before=copy.deepcopy(plan)
    events=[]
    result=xgb.fit_inputs(X,y,w,plan,name,events.append,device='cpu')
    assert plan==before and result['rows']==100 and result['boosted_rounds']==8
    assert result['effective_parameters']['random_state']==(123 if name==xgb.COMPONENTS[0] else 133)
    assert result['effective_parameters']['n_jobs']==2
    assert result['model_retained'] is False and result['quality_metrics'] is None
    assert result['input_hashes']==[imported['expected_binding'][k] for k in ('tensor','labels')]+[array_hash(w)]
    assert [e for e in events if e.startswith('round_')]==[f'round_{i}_of_8' for i in range(1,9)]


@pytest.mark.parametrize('target',['request.json','diagnostic/prepared/complete.json','diagnostic/prepared/stage1.npz'])
def test_import_corruption_rejected_before_decoding(tmp_path,monkeypatch,target):
    source,success,plan=training_source(tmp_path)
    with (source/target).open('ab') as stream: stream.write(b'drift')
    monkeypatch.setattr(np,'load',lambda *_a,**_k:pytest.fail('Decoded corrupt input'))
    with pytest.raises(ValueError,match='drift'):
        xgb.verify_import(source,success,plan,xgb.COMPONENTS[0])


@pytest.mark.parametrize('kind',['helper','model','labels','row_ids','weights','shape'])
def test_definition_and_identity_drift_fail(tmp_path,kind):
    source,success,plan=training_source(tmp_path)
    name=xgb.COMPONENTS[1]
    if kind=='helper':
        plan['sources']['src/ids_eval_framework/_native/protocol_b_grid.py']='0'*64
    elif kind=='model':
        plan['graph']['components'][name]['seed']=134
    if kind in ('helper','model'):
        with pytest.raises(ValueError): xgb.verify_import(source,success,plan,name)
        return
    imported=xgb.verify_import(source,success,plan,name)
    columns=46
    if kind=='weights': imported['expected_binding']['weights'][name]='0'*64
    elif kind=='shape': columns=45
    else: imported['expected_binding'][kind]='0'*64
    with pytest.raises(ValueError):
        xgb.load_training(source/imported['array_path'],imported,name,columns)


def test_historical_stage2_weight_recomputation_is_checked(tmp_path,monkeypatch):
    source,success,plan=training_source(tmp_path)
    name=xgb.COMPONENTS[1]
    imported=xgb.verify_import(source,success,plan,name)
    from experiments.c03.producer.validation import historical
    _,grid,_=historical()
    monkeypatch.setattr(grid,'stage2_sample_weights',lambda y,*_args:np.ones(len(y)))
    # Fixture has equal class counts, so use a deliberately different positive weight.
    monkeypatch.setattr(grid,'stage2_sample_weights',lambda y,*_args:np.full(len(y),2.))
    with pytest.raises(ValueError,match='Historical Stage-2 weighting'):
        xgb.load_training(source/imported['array_path'],imported,name)


def diagnostic(name,*,cpu=False,wrong_updater=False):
    cfg=config(name,'cpu' if cpu else 'cuda:0')
    cfg['learner']['gradient_booster']['gbtree_train_param']['updater']='grow_quantile_histmaker' if wrong_updater else 'grow_gpu_hist'
    cfg['learner']['gradient_booster']['gbtree_model_param']={'num_trees':'8' if name==xgb.COMPONENTS[0] else '40'}
    binding={'rows':xgb.DEFINITION['rows'][name],'tensor':'a'*64,'labels':'b'*64,'weights':{name:'c'*64}}
    imported={'scope':xgb.SCOPE,'scientific_checkpoint':False,'component':name,'definition':xgb.DEFINITION,
        'approved_success_sha256':xgb.prep_source.SUCCESS_SHA,'cuda_success_sha256':xgb.CUDA_SUCCESS_SHA,
        'expected_binding':binding,'fixture':True}
    result={'scope':xgb.SCOPE,'status':'XGB_SIZING_COMPONENT_COMPLETE','component':name,'scientific_checkpoint':False,
        'rows':binding['rows'],'features':46,'boosted_rounds':8,'model_retained':False,'quality_metrics':None,
        'input_kind':'VERIFIED_TRAIN_IMPORT','input_hashes':['a'*64,'b'*64,'c'*64],'booster_config':cfg}
    return imported,result


def fake_command(run,name,*,fail=False,sleep=0,cpu=False,wrong_updater=False):
    imported,result=diagnostic(name,cpu=cpu,wrong_updater=wrong_updater)
    code='''import json,sys,time
from pathlib import Path
from experiments.c03.producer.plan import canonical,file_hash
p=Path(sys.argv[1])
(p/'worker.guard').write_bytes(b'0')
(p/'stages.jsonl').write_text(json.dumps({'stage':'fixture'})+'\\n')
time.sleep(float(sys.argv[2]))
if sys.argv[3]=='yes': raise RuntimeError('fixture interruption')
(p/'import.json').write_bytes(canonical(json.loads(sys.argv[4])))
result=json.loads(sys.argv[5]); result['files']={'import.json':file_hash(p/'import.json')}
(p/'worker-result.json').write_bytes(canonical(result))
'''
    return [sys.executable,'-B','-u','-c',code,str(run/'profiles'/name),str(sleep),
            'yes' if fail else 'no',json.dumps(imported),json.dumps(result)]


def fixture_run(tmp_path,monkeypatch):
    monkeypatch.setattr(xgb,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(xgb.host,'resources',lambda _:host())
    monkeypatch.setattr(xgb,'query_gpu',gpu)
    run=xgb.new_run()
    (run/'request.json').write_bytes(canonical({**xgb.request_record(),'selected_gpu':xgb.gpu_identity(gpu())}))
    return run


def test_sequential_workers_share_budget_and_resume_does_not_fit(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    starts=[]
    original=xgb.run_profile
    def recorded(r,n,s,g):
        starts.append(s)
        return original(r,n,s,g)
    monkeypatch.setattr(xgb,'run_profile',recorded)
    monkeypatch.setattr(xgb,'command',fake_command)
    result=xgb.execute(run)
    assert result['status']=='COMPLETE' and starts[0]==starts[1]
    monkeypatch.setattr(xgb,'command',lambda *_:pytest.fail('Worker on resume'))
    monkeypatch.setattr(xgb,'query_gpu',lambda:pytest.fail('GPU on resume'))
    assert xgb.verify_completed(run)['reused_completed_result']


def test_partial_recovery_preserves_first_component_and_original_attempt(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(xgb,'command',lambda r,n:fake_command(r,n,fail=n==xgb.COMPONENTS[1]))
    result=xgb.execute(run)
    assert result['status']=='STOPPED' and result['completed_components']==[xgb.COMPONENTS[0]]
    before={str(p.relative_to(run)):file_hash(p) for p in run.rglob('*') if p.is_file()}
    target=xgb.fork_completed(run,xgb.gpu_identity(gpu()))
    calls=[]
    def command(r,n):
        calls.append(n)
        return fake_command(r,n)
    monkeypatch.setattr(xgb,'command',command)
    assert xgb.execute(target)['status']=='COMPLETE' and calls==[xgb.COMPONENTS[1]]
    assert {str(p.relative_to(run)):file_hash(p) for p in run.rglob('*') if p.is_file()}==before


@pytest.mark.parametrize('failure',['cpu','updater'])
def test_wrong_execution_backend_stops_batch_before_second_fit(tmp_path,monkeypatch,failure):
    run=fixture_run(tmp_path,monkeypatch)
    calls=[]
    def command(r,n):
        calls.append(n)
        return fake_command(r,n,cpu=failure=='cpu',wrong_updater=failure=='updater')
    monkeypatch.setattr(xgb,'command',command)
    assert xgb.execute(run)['status']=='STOPPED'
    assert calls==[xgb.COMPONENTS[0]]
    assert not (run/'profiles'/xgb.COMPONENTS[0]/'complete.json').exists()


@pytest.mark.parametrize('failure',['timeout','vram','ram','deadline'])
def test_live_worker_resource_stops(tmp_path,monkeypatch,failure):
    run=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(xgb,'command',lambda r,n:fake_command(r,n,sleep=30))
    count=0
    def query():
        nonlocal count
        count+=1
        value=gpu()
        if count>1:
            if failure=='timeout': raise subprocess.TimeoutExpired('nvidia-smi',2)
            if failure=='vram': value.update(used=5*xgb.GIB,free=3*xgb.GIB)
        return value
    def resources(_):
        value=host()
        if failure=='ram' and (run/'profiles'/xgb.COMPONENTS[0]/'worker.log').exists():
            value['available']=xgb.GIB
        return value
    monkeypatch.setattr(xgb,'query_gpu',query)
    monkeypatch.setattr(xgb.host,'resources',resources)
    start=time.monotonic()-(899 if failure=='deadline' else 0)
    result=xgb.run_profile(run,xgb.COMPONENTS[0],start,xgb.gpu_identity(gpu()))
    assert result['status']=='STOPPED' and result['worker_exit'] is not None
    assert not (run/'profiles'/xgb.COMPONENTS[0]/'complete.json').exists()


def test_full_row_gates_apply_before_each_worker_and_expired_budget(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    calls=[]
    def command(r,n):
        calls.append(n)
        return fake_command(r,n)
    def resources(_):
        value=host()
        if (run/'profiles'/xgb.COMPONENTS[0]/'complete.json').exists(): value['available']=5*xgb.GIB
        return value
    monkeypatch.setattr(xgb,'command',command)
    monkeypatch.setattr(xgb.host,'resources',resources)
    result=xgb.execute(run)
    assert result['status']=='STOPPED' and '6 GiB' in result['reason']
    assert calls==[xgb.COMPONENTS[0]]
    fresh=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(xgb,'command',lambda *_:pytest.fail('Started after deadline'))
    result=xgb.run_profile(fresh,xgb.COMPONENTS[0],time.monotonic()-901,xgb.gpu_identity(gpu()))
    assert result['status']=='STOPPED' and '900-second' in result['reason']


def test_active_lock_and_corrupt_checkpoint_block_recovery(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(xgb,'command',fake_command)
    assert xgb.execute(run)['status']=='COMPLETE'
    profile=run/'profiles'/xgb.COMPONENTS[0]
    with file_lock(profile/'worker.guard'):
        with pytest.raises(FileExistsError): xgb.fork_completed(run,xgb.gpu_identity(gpu()))
    (profile/'import.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='drift'): xgb.verify_completed(run)
    with pytest.raises(ValueError,match='drift'): xgb.fork_completed(run,xgb.gpu_identity(gpu()))


def test_prestart_gate_prevents_array_import_and_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(xgb,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(xgb,'query_gpu',gpu)
    monkeypatch.setattr(xgb,'load_readiness',lambda:{'selected_gpu':xgb.gpu_identity(gpu())})
    monkeypatch.setattr(xgb.host,'resources',lambda _:{**host(),'available':5*xgb.GIB})
    monkeypatch.setattr(xgb,'verify_import',lambda *_:pytest.fail('Array import before gate'))
    monkeypatch.setattr(xgb,'execute',lambda *_:pytest.fail('Worker before gate'))
    assert xgb.probe()['status']=='START_REFUSED'
    assert not list(tmp_path.iterdir())


def test_request_drift_and_uncommitted_recovery_rejected(tmp_path,monkeypatch):
    run=fixture_run(tmp_path,monkeypatch)
    request=xgb.read(run/'request.json'); request['recovery']='0'*64
    (run/'request.json').write_bytes(canonical(request))
    with pytest.raises(FileNotFoundError): xgb.verify_request(run)
    profile=run/'profiles'/xgb.COMPONENTS[0]; profile.mkdir(parents=True)
    (profile/'request.json').write_bytes(canonical({}))
    monkeypatch.setattr(xgb,'load_readiness',lambda:pytest.fail('Import after request drift'))
    with pytest.raises(ValueError,match='source/request'): xgb.worker(run,xgb.COMPONENTS[0])
