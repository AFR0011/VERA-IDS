"""Generated-array preparation integrity; no benchmark observations."""
import json
import time
import numpy as np
import pytest

from experiments.literature import doc_preparation as p, doc_runtime as rt


@pytest.fixture
def run(tmp_path):
    (tmp_path/'request.json').write_text(json.dumps({'fixture':'generated only'}))
    (tmp_path/'prepare').mkdir()
    return tmp_path


def test_partial_part_replay_and_mutation_rejection(run):
    x=p.mapped(run/'prepare/raw.npy',(10,46),np.float64)
    y=p.mapped(run/'prepare/y.npy',(10,),np.int64)
    positions=np.array([1,7,9]); values=np.arange(138,dtype=np.float64).reshape(3,46);labels=np.array([0,2,5])
    p.save_rows(run,'train',0,x,y,positions,values,labels,{'fixture':1})
    receipt=(run/'prepare/parts/train-0000.json').read_bytes()
    p.save_rows(run,'train',0,x,y,positions,values,labels,{'fixture':1})
    assert (run/'prepare/parts/train-0000.json').read_bytes()==receipt
    x[7,0]+=1;p.flush(x)
    with pytest.raises(ValueError,match='array rows drift'):
        p.save_rows(run,'train',0,x,y,positions,values,labels,{'fixture':1})


def test_scaling_recovery_preserves_train_statistics(run):
    rng=np.random.default_rng(47)
    x=rng.normal(size=(35,46)); v=np.full((19,46),1000.)
    x[:,0]=7
    calls=0
    def interrupt():
        nonlocal calls
        calls+=1
        if calls==8:raise RuntimeError('interruption')
    with pytest.raises(RuntimeError,match='interruption'):
        p.standardize(run,x,v,guard=interrupt,block=8)
    original={q.relative_to(run):q.read_bytes() for q in (run/'prepare').rglob('*.json')}
    stats=p.standardize(run,x,v,block=8)
    assert all((run/name).read_bytes()==data for name,data in original.items())
    np.testing.assert_allclose(stats.mean,x.mean(0),atol=1e-14)
    assert stats.scale[0]==1
    transformed=np.load(run/'prepare/validation-X.npy',allow_pickle=False)
    np.testing.assert_array_equal(transformed,stats.transform(v))
    bad=p.mapped(run/'prepare/validation-X.npy',v.shape,np.float32);bad[0,2]+=1;p.flush(bad)
    with pytest.raises(ValueError,match='Transformed checkpoint drift'):
        p.standardize(run,x,v,block=8)


def test_unknown_and_position_errors_fail_before_writes(run):
    x=p.mapped(run/'prepare/raw.npy',(10,46),np.float64)
    y=p.mapped(run/'prepare/y.npy',(10,),np.int64)
    for positions,labels in [(np.array([1,1]),np.array([0,1])),(np.array([1,2]),np.array([0,6]))]:
        with pytest.raises(ValueError):
            p.save_rows(run,'train',0,x,y,positions,np.ones((2,46)),labels,{})
    assert not (run/'prepare/parts/train-0000.json').exists()


def test_shared_deadline_has_no_stage_reset():
    with pytest.raises(ValueError,match='deadline'):
        rt.guard({'started':time.time()-1000,'deadline':time.time()-100})


def test_verification_rejects_escape_and_file_drift(run):
    path=run/'prepare/file.txt';path.write_text('data')
    record={'files':{'file.txt':rt.file_hash(path)}}
    rt.verify_files(run/'prepare',record)
    path.write_text('changed')
    with pytest.raises(ValueError,match='drift'):rt.verify_files(run/'prepare',record)
    with pytest.raises(ValueError,match='drift'):rt.verify_files(run/'prepare',{'files':{'../request.json':'0'*64}})
