"""Generated-array and disposable-process checks for the Stage-1 RF batch."""
import copy
import json
import sys
import time

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier

from experiments.c03.engineering import rf_stage1_sizing_probe as rf
from experiments.c03.producer.artifacts import array_hash, tensor_hash, identity_hash, write_arrays
from experiments.c03.producer.plan import canonical, digest, file_hash
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer.validation import historical
from test_c03_rf_stage2_sizing_probe import fake_source
from test_c03_preparation_probe import ample


def fixture_source(tmp_path):
    source, success, plan, *_ = fake_source(tmp_path)
    rng = np.random.RandomState(8)
    X = csr_matrix(rng.normal(size=(100, 3)))
    y = np.array([0]*30+[1]*70, dtype=np.int64)
    families = np.array(['Benign']*30+['A']*60+['B']*10)
    _, grid, _ = historical()
    weights = grid.make_stage1_sample_weights(y, families, 'inv_family_clipped', float(grid.CFG['inv_family_clip_max']))
    ids = pd.DataFrame({'source_part':['train/fixture.parquet']*100,
        'source_sha256':['c'*64]*100,'row_offset':np.arange(100,dtype=np.int64)})
    path = source/rf.ARRAY
    write_arrays(path,data=X.data,indices=X.indices,indptr=X.indptr,shape=np.array(X.shape),target=y,
        weight_stage1_rf_inv_family_clipped=weights,source_part=ids.source_part.to_numpy(dtype=str),
        source_sha256=ids.source_sha256.to_numpy(dtype=str),row_offset=ids.row_offset.to_numpy())
    binding = {'rows':100,'tensor':tensor_hash(X),'labels':array_hash(y),'row_ids':identity_hash(ids),
        'weights':{rf.COMPONENTS[0]:digest(None),rf.COMPONENTS[1]:array_hash(weights)}}
    prep_path = source/'diagnostic/prepared/complete.json'
    prep = rf.read(prep_path)
    prep['bindings']['stage1'] = binding
    prep['files']['prepared/stage1.npz'] = file_hash(path)
    prep_path.write_bytes(canonical(prep))
    result = rf.read(source/'worker-result.json')
    result['identities']['bindings']['stage1'] = binding
    result['files'].update({rf.ARRAY:file_hash(path),'diagnostic/prepared/complete.json':file_hash(prep_path)})
    (source/'worker-result.json').write_bytes(canonical(result))
    receipt = rf.read(source/'complete.json')
    receipt['worker_result_sha256'] = file_hash(source/'worker-result.json')
    (source/'complete.json').write_bytes(canonical(receipt))
    success['identities']['bindings']['stage1'] = binding
    success['evidence_hashes'] = {name:file_hash(source/name) for name in success['evidence_hashes']}
    return source, success, plan, X, y, weights


@pytest.mark.parametrize('name', rf.COMPONENTS)
def test_verified_import_and_both_actual_weighting_paths(tmp_path, monkeypatch, name):
    source, success, plan, X, y, weights = fixture_source(tmp_path)
    # Stage-1 import must not read the existing Stage-2 or any validation arrays.
    original_hash = rf.file_hash
    def checked_hash(path):
        assert not str(path).endswith(('stage2.npz','validation.npz'))
        return original_hash(path)
    monkeypatch.setattr(rf, 'file_hash', checked_hash)
    imported = rf.verify_import(source, success, plan)
    loaded_X, loaded_y, loaded_w = rf.load_stage1(source/rf.ARRAY, imported, name, columns=3)
    assert tensor_hash(loaded_X) == tensor_hash(X)
    np.testing.assert_array_equal(loaded_y,y)
    if name == rf.COMPONENTS[0]:
        assert loaded_w is None
    else:
        np.testing.assert_array_equal(loaded_w,weights)
        assert np.all(loaded_w[y==0] == 1) and len(np.unique(loaded_w[y==1])) == 2
    original_fit = RandomForestClassifier.fit
    calls = []
    def fit(model, tensor, target, **kwargs):
        calls.append(kwargs)
        return original_fit(model,tensor,target,**kwargs)
    monkeypatch.setattr(RandomForestClassifier,'fit',fit)
    def forbidden(*_a,**_k):
        pytest.fail('Unexpected scoring/model persistence')
    monkeypatch.setattr(RandomForestClassifier,'predict',forbidden)
    monkeypatch.setattr(RandomForestClassifier,'predict_proba',forbidden)
    import joblib
    monkeypatch.setattr(joblib,'dump',forbidden)
    before = copy.deepcopy(plan)
    result = rf.fit_diagnostic(loaded_X,loaded_y,loaded_w,plan,name,lambda _:None)
    assert plan == before and len(result['tree_node_counts']) == 8
    assert result['effective_parameters']['n_jobs'] == 2
    assert result['effective_parameters']['random_state'] == 123
    assert ('sample_weight' in calls[0]) == (name == rf.COMPONENTS[1])
    assert result['quality_metrics'] is None and result['model_retained'] is False


@pytest.mark.parametrize('target', ['request.json',rf.ARRAY,'diagnostic/prepared/complete.json'])
def test_import_corruption_fails_before_load(tmp_path,monkeypatch,target):
    source, success, plan, *_ = fixture_source(tmp_path)
    with (source/target).open('ab') as stream:
        stream.write(b'drift')
    monkeypatch.setattr(np,'load',lambda *_a,**_k:pytest.fail('Decoded corrupt source'))
    with pytest.raises(ValueError,match='drift'):
        rf.verify_import(source,success,plan)


@pytest.mark.parametrize('mode', ['family_weight','native_weight','labels','shape','helper'])
def test_identity_weight_and_helper_drift_rejected(tmp_path,mode):
    source, success, plan, *_ = fixture_source(tmp_path)
    if mode == 'helper':
        plan['sources']['src/ids_eval_framework/_native/protocol_b_grid.py']='0'*64
        with pytest.raises(ValueError,match='helper'):
            rf.verify_import(source,success,plan)
        return
    imported = rf.verify_import(source,success,plan)
    name,columns=rf.COMPONENTS[1],3
    if mode == 'family_weight':
        imported['expected_binding']['weights'][name]='0'*64
    elif mode == 'native_weight':
        name=rf.COMPONENTS[0]
        imported['expected_binding']['weights'][name]='0'*64
    elif mode == 'labels':
        imported['expected_binding']['labels']='0'*64
    else:
        columns=46
    with pytest.raises(ValueError):
        rf.load_stage1(source/rf.ARRAY,imported,name,columns)


def fake_command(run,name,*,fail=False,sleep=0):
    # Writes diagnostic fixtures only; no input arrays or estimator imports/fits.
    code = '''import json,sys,time
from pathlib import Path
from experiments.c03.producer.plan import canonical,file_hash
p=Path(sys.argv[1]); name=sys.argv[2]
(p/'worker.guard').write_bytes(b'0')
(p/'stages.jsonl').write_text(json.dumps({'stage':name+'_fixture'})+'\\n')
time.sleep(float(sys.argv[3]))
if sys.argv[4]=='yes': raise RuntimeError('fixture interruption')
(p/'import.json').write_bytes(canonical({'fixture':True}))
(p/'worker-result.json').write_bytes(canonical({'scope':'ENGINEERING_RF_STAGE1_SIZING_ONLY',
 'status':'PROFILE_COMPLETE','component':name,'scientific_checkpoint':False,
 'quality_metrics':None,'model_retained':False,'files':{'import.json':file_hash(p/'import.json')}}))
'''
    return [sys.executable,'-B','-u','-c',code,str(run/'profiles'/name),name,str(sleep),'yes' if fail else 'no']


def new_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(rf.guard,'resources',lambda _:ample())
    monkeypatch.setattr(rf,'OUTPUT_ROOT',tmp_path)
    run=rf.new_run()
    (run/'request.json').write_bytes(canonical(rf.request_record()))
    return run


def test_actual_sequential_workers_complete_and_resume_without_refit(tmp_path,monkeypatch):
    run=new_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(rf,'command',fake_command)
    calls=[]
    original=rf.run_profile
    def recorded(run,name,started):
        calls.append(started)
        return original(run,name,started)
    monkeypatch.setattr(rf,'run_profile',recorded)
    result=rf.execute(run)
    assert result['status']=='COMPLETE' and calls[0]==calls[1]
    assert result['completed_profiles']==list(rf.COMPONENTS)
    monkeypatch.setattr(rf,'command',lambda *_:pytest.fail('Worker on completed resume'))
    assert rf.verify_completed(run)['reused_completed_result']


def test_interrupted_second_profile_recovers_first_without_refit(tmp_path,monkeypatch):
    run=new_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(rf,'command',lambda r,n:fake_command(r,n,fail=n==rf.COMPONENTS[1]))
    result=rf.execute(run)
    assert result['status']=='STOPPED' and result['completed_profiles']==[rf.COMPONENTS[0]]
    original_files={str(p.relative_to(run)):file_hash(p) for p in run.rglob('*') if p.is_file()}
    target=rf.fork_completed(run)
    calls=[]
    def worker(r,n):
        calls.append(n)
        return fake_command(r,n)
    monkeypatch.setattr(rf,'command',worker)
    assert rf.execute(target)['status']=='COMPLETE'
    assert calls==[rf.COMPONENTS[1]]
    assert {str(p.relative_to(run)):file_hash(p) for p in run.rglob('*') if p.is_file()}==original_files
    assert rf.verify_completed(target)['reused_profiles']==[rf.COMPONENTS[0]]


def test_first_failure_does_not_start_second_and_no_empty_recovery(tmp_path,monkeypatch):
    run=new_fixture(tmp_path,monkeypatch)
    calls=[]
    def command(r,n):
        calls.append(n)
        return fake_command(r,n,fail=True)
    monkeypatch.setattr(rf,'command',command)
    assert rf.execute(run)['status']=='STOPPED'
    assert calls==[rf.COMPONENTS[0]]
    with pytest.raises(ValueError,match='No completed profile'):
        rf.fork_completed(run)


def test_second_profile_rechecks_start_memory(tmp_path,monkeypatch):
    run=new_fixture(tmp_path,monkeypatch)
    calls=[]
    def command(r,n):
        calls.append(n)
        return fake_command(r,n)
    def resources(_):
        state=ample()
        if (run/'profiles'/rf.COMPONENTS[0]/'complete.json').exists():
            state['available']=5*rf.guard.GIB
        return state
    monkeypatch.setattr(rf,'command',command)
    monkeypatch.setattr(rf.guard,'resources',resources)
    result=rf.execute(run)
    assert result['status']=='STOPPED' and '6 GiB' in result['reason']
    assert calls==[rf.COMPONENTS[0]]
    assert result['completed_profiles']==[rf.COMPONENTS[0]]


def test_shared_budget_blocks_later_worker_and_stops_running_worker(tmp_path,monkeypatch):
    run=new_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(rf,'command',lambda *_:pytest.fail('Launched after budget'))
    result=rf.run_profile(run,rf.COMPONENTS[0],time.monotonic()-901)
    assert result['status']=='STOPPED' and '900-second' in result['reason']
    monkeypatch.setattr(rf,'command',lambda r,n:fake_command(r,n,sleep=30))
    result=rf.run_profile(run,rf.COMPONENTS[1],time.monotonic()-899)
    assert result['status']=='STOPPED' and '900-second' in result['reason']
    assert result['worker_exit'] is not None
    assert not (run/'profiles'/rf.COMPONENTS[1]/'complete.json').exists()


def test_corrupt_checkpoint_and_live_worker_block_fork(tmp_path,monkeypatch):
    run=new_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(rf,'command',fake_command)
    assert rf.execute(run)['status']=='COMPLETE'
    profile=run/'profiles'/rf.COMPONENTS[0]
    with file_lock(profile/'worker.guard'):
        with pytest.raises(FileExistsError):
            rf.fork_completed(run)
    (profile/'worker-result.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='drift'):
        rf.fork_completed(run)
    with pytest.raises(ValueError,match='drift'):
        rf.verify_completed(run)


def test_incomplete_recovery_rejected(tmp_path,monkeypatch):
    run=new_fixture(tmp_path,monkeypatch)
    request=rf.request_record()
    request['recovery']='0'*64
    (run/'request.json').write_bytes(canonical(request))
    with pytest.raises(FileNotFoundError):
        rf.verify_request(run)


def test_start_gate_precedes_worker_import_and_recovery(tmp_path,monkeypatch):
    monkeypatch.setattr(rf,'OUTPUT_ROOT',tmp_path)
    monkeypatch.setattr(rf.guard,'resources',lambda _:{'available':5*rf.guard.GIB,'disk_free':30*rf.guard.GIB})
    monkeypatch.setattr(rf,'execute',lambda *_:pytest.fail('Worker below gate'))
    monkeypatch.setattr(rf,'fork_completed',lambda *_:pytest.fail('Recovery below gate'))
    monkeypatch.setattr(rf,'verify_import',lambda *_:pytest.fail('Import below gate'))
    assert rf.probe()['status']=='START_REFUSED'
    assert rf.probe(tmp_path/'parent')['status']=='START_REFUSED'
    assert not list(tmp_path.iterdir())


def test_worker_rejects_changed_request_before_import(tmp_path,monkeypatch):
    profile=tmp_path/'profiles'/rf.COMPONENTS[0]
    profile.mkdir(parents=True)
    (profile/'request.json').write_bytes(canonical({}))
    monkeypatch.setattr(rf.base,'load_success',lambda:pytest.fail('Import after request drift'))
    with pytest.raises(ValueError,match='request/source'):
        rf.worker(tmp_path,rf.COMPONENTS[0])
