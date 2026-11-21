"""Disposable generated fixtures; no benchmark test files are opened."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import joblib
import numpy as np
import pandas as pd
import pytest

from test_c03_pilot import prepared, completed
from experiments.c03.postseal import contract as c, selection as s, evaluation as e, supervisor as p
from experiments.c03.producer import validation as v, models as native
from experiments.c03.producer.plan import canonical, digest, file_hash
from experiments.c03.producer.artifacts import identity_hash, read_arrays
from experiments.c03.producer.checkpoints import publish


@pytest.fixture(scope='module')
def generated(completed, prepared, tmp_path_factory):
    source, stages = completed
    _, sources, _, _ = prepared
    root = tmp_path_factory.mktemp('postseal-generated')
    testroot = root/'inputs'; (testroot/'test').mkdir(parents=True)
    frames = [pd.read_parquet(sources.root/x['path']) for x in sources.inventory()['val']]
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 1750
    parts = []
    for i, piece in enumerate((frame.iloc[:713], frame.iloc[713:])):
        path=testroot/'test'/f'part-{i:03d}.parquet'
        piece.to_parquet(path,index=False)
        parts.append({'path':path.relative_to(testroot).as_posix(),'sha256':file_hash(path)})
    spec={'parts':parts,'rows':len(frame),'minimum_per_class':200,'block_rows':127,'order':'lexicographic part path, then original zero-based row offset'}
    paths={'plan.json','run.json','inputs.json','prepared/complete.json'}
    paths.update(source.read_json(source.path/'prepared/complete.json')['files'])
    components={}
    for name,(fit,score) in stages.items():
        components[name]={'fit':fit.relative_to(source.path).as_posix(),'score':score.relative_to(source.path).as_posix()}
        for folder,key in ((fit,'model_file'),(score,'score_file')):
            result=source.read_json(folder/'complete.json')['result']
            paths.update(x.relative_to(source.path).as_posix() for x in (folder/'complete.json',folder/result[key]))
    plan={'scope':c.SCOPE,'producer_run_id':source.path.name,'producer_plan_sha256':digest(source.plan),
          'producer_dependencies':{x:file_hash(source.path/x) for x in paths},'components':components,
          'sources':c.sources(),'test':spec,'limits':c.LIMITS,'uncertainty':'Generated test only'}
    return root,source,testroot,frame,plan


@pytest.fixture
def store(generated,tmp_path,monkeypatch):
    _,source,testroot,_,plan=generated
    path=tmp_path/'extension';path.mkdir()
    publish(path/'contract.json',canonical(plan))
    monkeypatch.setattr(c.producer,'frozen_plan',lambda:source.plan)
    monkeypatch.setattr(c,'OUTPUT_ROOT',source.path.parent)
    monkeypatch.setattr(c,'test_inventory',lambda:plan['test'])
    monkeypatch.setattr(e.host,'DATA_ROOT',testroot)
    return c.Store(path,copy.deepcopy(plan),source)


def generated_bundle(store):
    with np.load(store.source.path/'prepared/validation.npz',allow_pickle=False) as data: truth=data['target']
    core,_,_=v.historical()
    rng=np.random.RandomState(551)
    p1=np.where(truth==0,rng.uniform(.10,.40,len(truth)),rng.uniform(.6,.9,len(truth)))
    p2=rng.uniform(.02,.06,(len(truth),5))
    for i in range(1,6): p2[truth==i,i-1]=.85
    p2/=p2.sum(axis=1,keepdims=True)
    return truth,{name:core.CandidateScores(p1.copy(),p2.copy()) for name in core.EXPECTED}


def seal(store,monkeypatch):
    monkeypatch.setattr(s,'bundle',generated_bundle)
    return s.select(store,lambda _:None)


def test_streamed_population_hash_equals_original_identity_hash():
    parts=[{'path':'test/a.parquet','sha256':'a'*64},{'path':'test/longer.parquet','sha256':'b'*64}]
    h=e.PopulationHash(11,parts);frames=[]
    for part,n in zip(parts,(4,7)):
        for offset in range(0,n,3): h.add(part,offset,min(3,n-offset))
        frames.append(pd.DataFrame({'source_part':[part['path']]*n,'source_sha256':[part['sha256']]*n,'row_offset':np.arange(n,dtype=np.int64)}))
    assert h.finish()==identity_hash(pd.concat(frames,ignore_index=True))
    with pytest.raises(ValueError,match='count'): e.PopulationHash(1,parts).finish()


def test_bundle_reads_only_targets_and_verified_score_arrays(store):
    store.verify()
    truth,scores=s.bundle(store)
    assert truth.shape==(1750,) and len(scores)==3


def test_selection_seals_both_arms_before_test_access_and_recovers(store,monkeypatch):
    monkeypatch.setattr(e,'test_batches',lambda _:pytest.fail('Selection read test'))
    record=seal(store,monkeypatch)
    assert set(record['decisions']['arms'])=={'blind','visible'}
    assert all(len(a['selections'])==4 for a in record['decisions']['arms'].values())
    sha=file_hash(store.path/'selection.json')
    (store.path/'selection-seal.json').unlink()
    monkeypatch.setattr(s,'bundle',lambda _:pytest.fail('Recovery repeated selection'))
    assert s.select(store,lambda _:None)==record
    assert file_hash(store.path/'selection.json')==sha


@pytest.mark.parametrize('mutation',['missing','selection','source','producer'])
def test_bad_seal_blocks_test_resolution_and_model_load(store,monkeypatch,mutation):
    seal(store,monkeypatch)
    if mutation=='missing': (store.path/'selection-seal.json').unlink()
    elif mutation=='selection':
        with (store.path/'selection.json').open('ab') as f:f.write(b' ')
    elif mutation=='source': monkeypatch.setattr(c,'sources',lambda:{})
    else:
        monkeypatch.setattr(c,'file_hash',lambda _: '0'*64)
    monkeypatch.setattr(e,'test_batches',lambda _:pytest.fail('Unsealed test access'))
    monkeypatch.setattr(e.native,'load_verified_local_model',lambda *a:pytest.fail('Unsafe model load'))
    with pytest.raises((ValueError,FileNotFoundError)):
        e.score_component(store,c.producer.COMPONENTS[0],store.path,lambda _:None)


def score_all(store):
    record=s.verify_seal(store)
    for name in s.selected_components(store,record):
        directory=store.path/'attempts'/name/'agenerated';directory.mkdir(parents=True)
        e.score_component(store,name,directory,lambda _:None)


@pytest.mark.parametrize('family',['rf','xgb'])
def test_streaming_scores_equal_full_transform_and_paired_analysis(store,generated,monkeypatch,family):
    def bundle(store):
        truth,scores=generated_bundle(store)
        if family=='xgb':
            core,_,_=v.historical()
            for name,value in list(scores.items()):
                if name.startswith('rf'):
                    scores[name]=core.CandidateScores(value.attack_prob,np.roll(value.family_prob,1,axis=1))
        return truth,scores
    monkeypatch.setattr(s,'bundle',bundle)
    record=s.select(store,lambda _:None)
    assert all('_'+family+'_' in name for name in s.selected_components(store,record))
    score_all(store)
    _,_,_,frame,_=generated
    prep=joblib.load(store.source.path/'prepared/preprocessor/preprocessor.joblib')
    X=prep.transform(frame.drop(columns=['y_stage1_attack','y_stage2_family']))
    identities=[]
    for part in store.plan['test']['parts']:
        n=len(pd.read_parquet(e.host.DATA_ROOT/part['path']))
        identities.append(pd.DataFrame({'source_part':[part['path']]*n,'source_sha256':[part['sha256']]*n,'row_offset':np.arange(n,dtype=np.int64)}))
    for name in s.selected_components(store,record):
        directory=e.completed_component(store,name)
        receipt,truth,actual=e.verify_component(store,name,directory)
        fitdir=store.source.path/store.plan['components'][name]['fit']
        fit=store.source.read_json(fitdir/'complete.json')['result']
        model=native.load_verified_local_model(fitdir/fit['model_file'],fit['model_family'])
        if fit['model_family']=='xgb':model.set_params(device='cpu',n_jobs=2,tree_method='hist')
        expected=native.probabilities(model,X,store.source.plan['graph']['components'][name]['stage'])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        assert receipt['population_sha256']==identity_hash(pd.concat(identities,ignore_index=True))
        np.testing.assert_array_equal(truth,v.truth_labels(frame))
    result=e.analyze(store,lambda _:None)
    assert result['rows']==1750 and len(result['records'])==16 and result['test_accessed']
    for pair in result['records'].values():
        for arm in pair['arms'].values():
            if arm['status']=='feasible':assert sum(map(sum,arm['metrics']['confusion_matrix']))==1750
    assert e.verify_result(store)==result
    sha=file_hash(store.path/'evaluation.json')
    (store.path/'evaluation-complete.json').unlink()
    assert e.analyze(store,lambda _:None)==result
    assert file_hash(store.path/'evaluation.json')==sha


def test_score_corruption_and_duplicate_receipts_rejected(store,monkeypatch):
    record=seal(store,monkeypatch)
    name=s.selected_components(store,record)[0]
    directory=store.path/'attempts'/name/'agenerated';directory.mkdir(parents=True)
    e.score_component(store,name,directory,lambda _:None)
    with (directory/'test.npz').open('ab') as f:f.write(b'drift')
    with pytest.raises(ValueError,match='artifact drift'):e.completed_component(store,name)
    duplicate=directory.parent/'aduplicate';duplicate.mkdir()
    shutil.copyfile(directory/'complete.json',duplicate/'complete.json')
    with pytest.raises(ValueError,match='Duplicate'):e.completed_component(store,name)


def test_all_infeasible_never_reads_test(store,monkeypatch):
    core,_,_=v.historical()
    original=core.select_both_arms
    def infeasible(*a,**kw):
        result=original(*a,**kw)
        for arm in result['arms'].values():
            for budget in arm['selections'].values():
                for method in budget:budget[method]={'status':'infeasible'}
        return result
    monkeypatch.setattr(core,'select_both_arms',infeasible)
    monkeypatch.setattr(e,'test_batches',lambda _:pytest.fail('Infeasible policies read test'))
    seal(store,monkeypatch)
    result=e.analyze(store,lambda _:None)
    assert result['test_accessed'] is False and result['rows'] is None
    assert all(x['visible_minus_blind'] is None for x in result['records'].values())


def test_test_inventory_drift_rejected(store,monkeypatch):
    seal(store,monkeypatch)
    store.plan['test']['parts'][0]['sha256']='0'*64
    # Direct reader test; source contract/receipt tests cover the earlier boundary.
    with pytest.raises(ValueError,match='before read'): next(e.test_batches(store.plan['test']))


def test_orphan_watchdog_exits_real_process(tmp_path):
    script='''
import threading,time
from experiments.c03.postseal.supervisor import watch
class B:
 def elapsed(self): return 0
threading.Thread(target=watch,args=(99999999,0,B(),threading.Event()),daemon=True).start()
time.sleep(10)
'''
    proc=subprocess.run([sys.executable,'-B','-c',script],capture_output=True,timeout=20)
    assert proc.returncode==15


@pytest.mark.parametrize('trigger',['rss','disk','artifacts','gpu_used','gpu_free','deadline','query'])
def test_supervisor_stop_conditions(store,monkeypatch,trigger):
    telemetry={**p.gpu.load_readiness()['selected_gpu'],'free':7*2**30,'used':0}
    req={'gpu':p.gpu.gpu_identity(telemetry)}
    publish(store.path/'request.json',canonical(req))
    launched,stopped=[],[]
    class Process:
        pid=1234;returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=15;return 15
    process=Process()
    def launch(*a,**k):launched.append(True);return process
    class Tree:
        def __init__(self,pid):pass
        def rss(self):return 5*2**30 if trigger=='rss' else 1000
        def live(self):return []
        def stop(self):stopped.append(True);process.returncode=15
    def gpu():
        value=telemetry.copy()
        if launched:
            if trigger=='query':raise TimeoutError('query timeout')
            if trigger=='gpu_used':value['used']=5*2**30
            if trigger=='gpu_free':value['free']=2**30
        return value
    monkeypatch.setattr(p.subprocess,'Popen',launch)
    monkeypatch.setattr(p.host,'ProcessTree',Tree)
    monkeypatch.setattr(p.host,'resources',lambda _:{'available':0,'disk_free':(9 if launched and trigger=='disk' else 30)*2**30})
    monkeypatch.setattr(p.host,'artifact_size',lambda _:6*2**30 if launched and trigger=='artifacts' else 0)
    monkeypatch.setattr(p.gpu,'query_gpu',gpu)
    monkeypatch.setitem(c.LIMITS,'gpu_interval',0)
    class Budget:
        def elapsed(self):return 3601 if launched and trigger=='deadline' else 0
    result=p.run_stage(store,'selection',req,Budget())
    assert result['status']=='STOPPED' and launched==[True] and stopped


def test_original_deadline_cannot_reset_on_recovery(store):
    now=time.time()-3601
    publish(store.path/'request.json',canonical({'contract_sha256':digest(store.plan),'limits':c.LIMITS,
        'started_epoch':now,'deadline_epoch':now+3600}))
    with pytest.raises(ValueError,match='budget exhausted'):p.execute(store,recover=True)


def test_real_isolated_score_worker(store,generated,monkeypatch):
    record=seal(store,monkeypatch)
    name=s.selected_components(store,record)[0]
    assert '_rf_' in name
    now=time.time();telemetry={**p.gpu.load_readiness()['selected_gpu'],'free':7*2**30,'used':0}
    req={'contract_sha256':digest(store.plan),'limits':c.LIMITS,'started_epoch':now,'deadline_epoch':now+3600,'gpu':p.gpu.gpu_identity(telemetry)}
    publish(store.path/'request.json',canonical(req))
    monkeypatch.setattr(p.gpu,'query_gpu',lambda:telemetry)
    monkeypatch.setattr(p.host,'resources',lambda _:{'available':0,'disk_free':30*2**30})
    script='''
import sys,json
from pathlib import Path
from experiments.c03.postseal import contract as c,supervisor as p,evaluation as e
run=Path(sys.argv[1]);source=Path(sys.argv[5])
c.OUTPUT_ROOT=source.parent
c.producer.frozen_plan=lambda:json.loads((source/'plan.json').read_text())
c.test_inventory=lambda:json.loads((run/'contract.json').read_text())['test']
e.host.DATA_ROOT=Path(sys.argv[6])
p.worker(run,sys.argv[2],sys.argv[3])
'''
    monkeypatch.setattr(p,'command',lambda run,stage,attempt:[sys.executable,'-B','-c',script,str(run),stage,attempt,'unused',str(store.source.path),str(generated[2])])
    result=p.run_stage(store,name,req,p.old.Budget(req))
    assert result['status']=='COMPLETE',result
    assert result['peak_tree_rss']>0 and result['worker_exit']==0
    assert e.completed_component(store,name) is not None
