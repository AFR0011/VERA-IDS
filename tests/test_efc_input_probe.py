"""Generated Parquet and checkpoint tests; no benchmark observations."""
import json,time
import numpy as np
import pandas as pd
import pytest
from experiments.literature import efc_input_probe as probe


@pytest.fixture
def source(tmp_path):
    frame=pd.DataFrame({'Feature A':np.arange(12,dtype=float),'Feature B':np.arange(12,dtype=float)*2,
        'y_stage1_attack':[0,1,1,1,1,1]*2,
        'y_stage2_family':['Benign','BruteForce','DDoS','DoS','Other','Scan/Recon']*2})
    path=tmp_path/'train/part.parquet';path.parent.mkdir();frame.to_parquet(path,index=False)
    return path,frame


def test_selected_rows_preserve_order_values_and_family_labels(source):
    path,frame=source;offsets=np.array([9,0,5,2]);binary=frame.y_stage1_attack.to_numpy()[offsets]
    X,y=probe.extract_part(path,probe.file_hash(path),['feature_a','feature_b'],offsets,binary)
    np.testing.assert_array_equal(X,frame[['Feature A','Feature B']].to_numpy()[offsets])
    np.testing.assert_array_equal(y,[3,0,5,2])


@pytest.mark.parametrize('damage',['sha','duplicate','range','label','heldout','column'])
def test_invalid_source_or_identity_fails_closed(source,damage):
    path,frame=source;offsets=np.array([0,1]);binary=np.array([0,1]);cols=['feature_a','feature_b']
    if damage=='heldout':frame.loc[1,'y_stage2_family']='Botnet';frame.to_parquet(path,index=False)
    sha=probe.file_hash(path)
    if damage=='sha':sha='0'*64
    elif damage=='duplicate':offsets=np.array([1,1])
    elif damage=='range':offsets=np.array([0,12])
    elif damage=='label':binary=np.array([1,1])
    elif damage=='column':cols=['missing']
    with pytest.raises(ValueError):probe.extract_part(path,sha,cols,offsets,binary)


def test_nonfinite_values_remain_visible_without_imputation(source):
    path,frame=source;frame.loc[1,'Feature A']=np.nan;frame.loc[2,'Feature B']=np.inf
    frame.to_parquet(path,index=False)
    X,y=probe.extract_part(path,probe.file_hash(path),['feature_a','feature_b'],np.array([1,2]),np.array([1,1]))
    assert np.isnan(X[0,0]) and np.isposinf(X[1,1]) and np.sum(~np.isfinite(X))==2


def test_sparse_offsets_across_batch_boundaries_keep_sample_order(tmp_path):
    size=20000
    frame=pd.DataFrame({'feature':np.arange(size,dtype=float),'y_stage1_attack':np.ones(size,dtype=int),
                        'y_stage2_family':['DDoS']*size})
    path=tmp_path/'part.parquet';frame.to_parquet(path,index=False)
    offsets=np.array([19999,8192,0,16384,8191])
    X,y=probe.extract_part(path,probe.file_hash(path),['feature'],offsets,np.ones(len(offsets),dtype=int))
    np.testing.assert_array_equal(X[:,0],offsets)
    np.testing.assert_array_equal(y,np.repeat(2,len(offsets)))


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    root=tmp_path/'data';(root/'train').mkdir(parents=True)
    labels=['Benign','BruteForce','DDoS','DoS','Other','Scan/Recon']
    frame=pd.DataFrame({'feature_a':np.arange(1200,dtype=float),'y_stage1_attack':np.tile([0,1,1,1,1,1],200),
                        'y_stage2_family':labels*200})
    path=root/'train/part.parquet';frame.to_parquet(path,index=False)
    run=tmp_path/'run';run.mkdir();(run/'request.json').write_text('{}')
    plan={'rows':1200,'columns':['feature_a'],'identity_sha256':'synthetic',
          'inventory':[{'path':'train/part.parquet','sha256':probe.file_hash(path)}]}
    req={'definition':plan}
    monkeypatch.setattr(probe,'request',lambda run:req)
    monkeypatch.setattr(probe.host,'DATA_ROOT',root)
    monkeypatch.setattr(probe,'load_index',lambda plan:(np.repeat('train/part.parquet',1200),np.arange(1200),frame.y_stage1_attack.to_numpy()))
    return run,plan


def test_completed_parts_reuse_and_corruption_rejection(prepared,monkeypatch):
    run,plan=prepared;first=probe.calculate(run,lambda _:None)
    assert first['class_support']==[200]*6 and first['ready_for_preprocessing']
    assert first['models_fitted'] is False and first['validation_or_test_read'] is False
    before=(run/'parts/p0000/values.npz').read_bytes()
    monkeypatch.setattr(probe,'extract_part',lambda *args,**kw:pytest.fail('Completed extraction repeated'))
    second=probe.calculate(run,lambda _:None)
    assert second['reused_parts']==[0] and (run/'parts/p0000/values.npz').read_bytes()==before
    (run/'parts/p0000/values.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checkpoint bytes drift'):probe.calculate(run,lambda _:None)


def test_interrupted_part_publication_recovers_without_overwrite(prepared,monkeypatch):
    run,plan=prepared;original=probe.publish
    def interrupted(path,data):
        if path.name=='complete.json':raise RuntimeError('power interruption')
        original(path,data)
    monkeypatch.setattr(probe,'publish',interrupted)
    with pytest.raises(RuntimeError,match='power interruption'):probe.calculate(run,lambda _:None)
    before=(run/'parts/p0000/values.npz').read_bytes()
    monkeypatch.setattr(probe,'publish',original)
    result=probe.calculate(run,lambda _:None)
    assert result['ready_for_preprocessing'] and (run/'parts/p0000/values.npz').read_bytes()==before


def test_expired_original_budget_does_not_launch_worker(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();now=time.time()
    req={'started_epoch':now-1000,'deadline_epoch':now-1000+900,'definition':{}}
    monkeypatch.setattr(probe,'request',lambda run:req)
    monkeypatch.setattr(probe.subprocess,'Popen',lambda *a,**k:pytest.fail('Expired worker launched'))
    monkeypatch.setattr(probe.host,'resources',lambda p:{'available':1,'disk_free':100*2**30})
    result=probe.execute(run,recover=True)
    assert result['status']=='STOPPED' and 'time budget' in result['reason']
