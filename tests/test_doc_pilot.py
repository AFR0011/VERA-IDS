"""Synthetic supervision/deadline and post-seal integrity checks; no benchmark reads."""
import json
import time
from pathlib import Path
import subprocess

import pytest
from experiments.literature import doc_pilot as pilot, doc_pilot_io as rt, doc_pilot_test as evaluation


def fixture_run(root):
    now=time.time(); root.mkdir()
    plan={'started':now,'fit_deadline':now+7200,'deadline':now+8100}
    (root/'request.json').write_text(json.dumps(plan))
    return plan


def test_phase_deadline_recovery_keeps_original_postfit_limit(tmp_path):
    run=tmp_path/'run'; plan=fixture_run(run)
    assert rt.deadline(run,plan)==plan['fit_deadline']
    ended=plan['started']+100
    rt.bound_json(run,'fit-complete.json',{'completed_epoch':ended})
    assert rt.deadline(run,plan)==ended+900
    value=rt.read(run/'fit-complete.json'); value['completed_epoch']=plan['fit_deadline']+1
    (run/'fit-complete.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='deadline'): rt.deadline(run,plan)


def test_export_rejects_unsealed_before_resolving_sources(tmp_path,monkeypatch):
    run=tmp_path/'run'; plan=fixture_run(run)
    with pytest.raises((ValueError,FileNotFoundError)):
        evaluation.export(run,plan,lambda *a,**k:None)
    assert not (run/'test-inputs').exists()


def test_telemetry_timeout_stops_worker_and_records_attempt(tmp_path,monkeypatch):
    run=tmp_path/'run'; plan=fixture_run(run); stopped=[]
    class Process:
        pid=12345; returncode=None
        def __init__(self,*a,**k): pass
        def poll(self): return None
        def wait(self,timeout): self.returncode=15
    class Tree:
        def __init__(self,pid): pass
        def rss(self): return 100
        def stop(self): stopped.append(True)
    def failed(): raise subprocess.TimeoutExpired('nvidia-smi',2)
    monkeypatch.setattr(pilot.subprocess,'Popen',Process)
    monkeypatch.setattr(pilot.host,'ProcessTree',Tree)
    monkeypatch.setattr(pilot.host,'resources',lambda run:{'available':1,'disk_free':30*2**30})
    monkeypatch.setattr(pilot,'query_gpu',failed)
    result=pilot.run_stage(run,plan,'fit',pilot.Budget({'started_epoch':plan['started']}),{})
    assert stopped and result['worker_exit']==15 and 'TimeoutExpired' in result['reason']
    assert rt.read(run/'attempts'/result['attempt']/'result.json')==result
    assert not (run/'complete.json').exists()


def test_test_export_exact_population_and_recovery(tmp_path,monkeypatch):
    import numpy as np
    import pandas as pd
    from experiments.c03.engineering import preparation_probe as host
    from experiments.c03.producer.artifacts import array_hash
    from experiments.c03.postseal.evaluation import PopulationHash
    from experiments.c03.producer.validation import FAMILIES
    run=tmp_path/'run'; plan=fixture_run(run); raw=tmp_path/'raw'; (raw/'test').mkdir(parents=True)
    columns=['x'+str(i) for i in range(46)]
    x=np.arange(7*46,dtype='float64').reshape(7,46)/100
    frame=pd.DataFrame(x,columns=columns); frame['y_stage1_attack']=[0,1,1,1,1,1,1]
    frame['y_stage2_family']=['Benign',*FAMILIES,'Botnet']
    source=raw/'test/generated.parquet'; frame.to_parquet(source,index=False)
    entry={'path':'test/generated.parquet','sha256':rt.file_hash(source)}
    pop=PopulationHash(7,[entry]); pop.add(entry,0,7)
    plan.update(columns=columns,test={'rows':7,'parts':[entry]},
                expected_population_sha256=pop.finish(),expected_truth_sha256=array_hash(np.arange(7,dtype='int8')))
    (run/'request.json').write_text(json.dumps(plan)); (run/'seal.json').write_text('{}')
    stats=run/'generated'; stats.mkdir()
    (stats/'scaler.json').write_text(json.dumps({'count':1,'mean':[0.]*46,'m2':[1.]*46}))
    monkeypatch.setattr(host,'DATA_ROOT',raw)
    monkeypatch.setattr(rt,'verify_seal',lambda *a:{})
    monkeypatch.setattr(rt,'preparation',lambda *a:(stats,{}))
    evaluation.export(run,plan,lambda *a,**k:None)
    before=(run/'test-inputs/p0000.npz').read_bytes()
    evaluation.export(run,plan,lambda *a,**k:None)
    assert (run/'test-inputs/p0000.npz').read_bytes()==before
    with np.load(run/'test-inputs/p0000.npz') as arrays:
        np.testing.assert_array_equal(arrays['X'],x.astype('float32'))
        np.testing.assert_array_equal(arrays['truth'],np.arange(7))
    source.write_bytes(source.read_bytes()+b'drift')
    with pytest.raises(ValueError,match='source drift'): evaluation.export(run,plan,lambda *a,**k:None)
