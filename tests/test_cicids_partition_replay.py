"""Generated-data checks for the read-only historical partition replay."""
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.cicids_partition import replay as r
from experiments.c03.producer.plan import canonical,file_hash
from experiments.c03.producer.checkpoints import publish


def test_generator_is_exact_and_destructive_entrypoints_disabled(tmp_path):
    module=r.load_generator()
    assert file_hash(r.GENERATOR)==r.GENERATOR_SHA
    for fn in (module.main,module.materialize):
        with pytest.raises(RuntimeError,match='cannot materialize'):fn(tmp_path)
    assert not list(tmp_path.iterdir())


def test_drifted_generator_never_executes(tmp_path,monkeypatch):
    path=tmp_path/'bad.py';path.write_text('raise RuntimeError("executed")')
    monkeypatch.setattr(r,'GENERATOR',path)
    with pytest.raises(ValueError,match='bytes differ'):r.load_generator()


def test_historical_assignment_is_deterministic_and_keeps_groups_whole(tmp_path):
    module=r.load_generator()
    labels=np.repeat(['Benign','Botnet','DoS'],60)
    x=np.arange(len(labels),dtype=float);x[1]=x[0];x[61]=x[60]
    frame=pd.DataFrame({'feature':x,'other':np.zeros(len(x)),
        'y_stage1_attack':(labels!='Benign').astype(int),'y_stage2_family':labels,'y_stage2_fine':labels})
    hashes=pd.util.hash_pandas_object(frame[['feature','other']],index=False).to_numpy(dtype=np.uint64)
    module.EXPECTED_ROWS=len(frame);module.EXPECTED_GROUPS=len(np.unique(hashes))
    for split,positions in zip(module.SPLITS,np.array_split(np.arange(len(frame)),3)):
        folder=tmp_path/split;folder.mkdir();frame.iloc[positions].to_parquet(folder/'part.parquet',index=False)
    before={p:file_hash(p) for p in tmp_path.rglob('*.parquet')}
    a,meta,support,moves,features,count=module.build_assignment(tmp_path,123,5,20)
    b,other,*_=module.build_assignment(tmp_path,123,5,37)
    pd.testing.assert_series_equal(a,b)
    assert len(a)==178 and features==['feature','other'] and count==3
    assigned=pd.Series(hashes).map(a)
    assert assigned.notna().all() and assigned[0]==assigned[1] and assigned[60]==assigned[61]
    assert (support.min(axis=1)>=5).all() and sum(meta['n'])==180
    assert before=={p:file_hash(p) for p in tmp_path.rglob('*.parquet')}
    codes=meta['split'].map({'train':0,'val':1,'test':2}).to_numpy(dtype=np.uint8)
    groups=meta['h'].to_numpy(dtype=np.uint64)
    assert r.assignment_digest(groups,codes)==r.assignment_digest(groups[::-1],codes[::-1])


def test_comparison_reports_discrepancy_without_changing_reference(tmp_path):
    meta=pd.DataFrame({'n':[200,200,200],'nfam':[1,1,1],'split':['train','val','test']})
    support=pd.DataFrame({'train':[200],'val':[200],'test':[200]},index=pd.Index(['Benign'],name='family'))
    moves=[{'family':'Benign','from':'train','to':'val','rows':1,'group_family_count':1}]
    expected={'source_part_files':3,'seed':123,'minimum_family_support':200,'feature_columns':1,'rows':600,
        'unique_exact_feature_groups':3,'family_conflict_groups':0,'repair_moves':1,'repair_rows':1,
        'assignment_row_totals':{'train':200,'val':200,'test':200}}
    (tmp_path/'group_safe_surface_summary.json').write_bytes(canonical(expected))
    table=support.copy();table['min_support']=200;table.to_csv(tmp_path/'materialized_family_support.csv')
    pd.DataFrame(moves).to_csv(tmp_path/'materialized_support_repair_moves.csv',index=False)
    (tmp_path/'feature_columns.txt').write_text('feature\n')
    before={p:file_hash(p) for p in tmp_path.iterdir()}
    assert r.compare(meta,support,moves,['feature'],3,tmp_path)['all_match']
    meta.loc[0,'n']=201
    result=r.compare(meta,support,moves,['feature'],3,tmp_path)
    assert not result['all_match'] and not result['checks']['assignment_row_totals']
    assert before=={p:file_hash(p) for p in tmp_path.iterdir()}


def make_request(run,seconds_ago=0):
    now=time.time()-seconds_ago
    req={'definition':r.definition(),'started_epoch':now,'deadline_epoch':now+r.LIMITS['seconds']}
    publish(run/'request.json',canonical(req));return req


def test_recovery_reuses_assignment_without_recomputing(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();make_request(run)
    entries=[{'path':str(i),'rows':1,'sha256':'a'*64} for i in range(24)]
    publish(run/'inventory.json',canonical(entries))
    publish(run/'comparison.json',canonical({'all_match':True}))
    publish(run/'assignment-complete.json',canonical({'scope':r.SCOPE,'request_sha256':file_hash(run/'request.json'),
        'inventory_sha256':file_hash(run/'inventory.json'),'files':{'comparison.json':file_hash(run/'comparison.json')},
        'comparison_path':'comparison.json'}))
    monkeypatch.setattr(r,'load_generator',lambda:SimpleNamespace(EXPECTED_ROWS=24,build_assignment=lambda *a,**k:pytest.fail('Refit assignment')))
    monkeypatch.setattr(r,'inventory',lambda *a:entries)
    assert r.calculate(run,run,lambda _:None)=={'all_match':True}
    (run/'comparison.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact drift'):r.calculate(run,run,lambda _:None)


@pytest.mark.parametrize('trigger',['rss','disk','artifacts','deadline','worker'])
def test_supervisor_cleans_up_on_stop(tmp_path,monkeypatch,trigger):
    run=tmp_path/'run';run.mkdir();make_request(run)
    launched,stopped=[],[]
    class Process:
        pid=1234;returncode=None
        def poll(self):
            if trigger=='worker':self.returncode=15
            return self.returncode
        def wait(self,timeout):self.returncode=15;return 15
    process=Process()
    def launch(*a,**kw):launched.append(True);return process
    class Tree:
        def __init__(self,pid):pass
        def rss(self):return 5*2**30 if trigger=='rss' else 1000
        def live(self):return []
        def stop(self):stopped.append(True);process.returncode=15
    monkeypatch.setattr(r.subprocess,'Popen',launch)
    monkeypatch.setattr(r.host,'ProcessTree',Tree)
    monkeypatch.setattr(r.host,'resources',lambda _:{'available':0,'disk_free':(9 if launched and trigger=='disk' else 30)*2**30})
    monkeypatch.setattr(r.host,'artifact_size',lambda _:6*2**30 if launched and trigger=='artifacts' else 0)
    class Budget:
        def __init__(self,request):pass
        def elapsed(self):return 901 if launched and trigger=='deadline' else 0
    monkeypatch.setattr(r.lifecycle,'Budget',Budget)
    result=r.execute(run)
    assert result['status']=='STOPPED' and launched==[True] and stopped
    assert not (run/'complete.json').exists()


def test_original_deadline_and_definition_are_enforced(tmp_path):
    make_request(tmp_path,seconds_ago=901)
    with pytest.raises(ValueError,match='deadline exhausted'):r.execute(tmp_path,recover=True)
    request=json.loads((tmp_path/'request.json').read_text());request['definition']['seed']=124
    (tmp_path/'request.json').write_bytes(canonical(request))
    with pytest.raises(ValueError,match='drift'):r.request(tmp_path)


def test_orphan_worker_exits_actual_process():
    code='''
import threading,time
from experiments.cicids_partition.replay import watch
class Budget:
 def elapsed(self):return 0
threading.Thread(target=watch,args=(99999999,0,Budget(),threading.Event()),daemon=True).start()
time.sleep(10)
'''
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,timeout=20)
    assert result.returncode==15
