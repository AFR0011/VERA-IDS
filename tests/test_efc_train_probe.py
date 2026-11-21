import json,time
from pathlib import Path
import numpy as np
import pytest
from experiments.literature import efc_train_probe as train
from experiments.c03.producer.artifacts import write_arrays


@pytest.fixture
def numeric_parts(tmp_path,monkeypatch):
    root=tmp_path/'inputs';root.mkdir();parts=[]
    for i,positions in enumerate((np.array([3,0,5]),np.array([2,4,1]))):
        folder=root/f'part{i}';folder.mkdir()
        write_arrays(folder/'values.npz',positions=positions,X=positions[:,None].astype(float),y=positions.astype(np.int8))
        parts.append({'part':folder.name,'values_sha256':train.file_hash(folder/'values.npz')})
    monkeypatch.setattr(train,'INPUT_RUN',root)
    return {'rows':6,'columns':['x'],'class_support':[1]*6,'parts':parts,'ready_for_preprocessing':True,'identity_sha256':'fixture'}


def test_native_input_assembly_restores_order_and_labels(tmp_path,numeric_parts):
    directory=tmp_path/'assembly';directory.mkdir()
    X,y=train.assemble_inputs(directory,numeric_parts,lambda _:None)
    np.testing.assert_array_equal(X[:,0],np.arange(6));np.testing.assert_array_equal(y,np.arange(6))
    del X,y


def test_duplicate_parts_cannot_overwrite_rows(tmp_path,numeric_parts):
    directory=tmp_path/'assembly';directory.mkdir();numeric_parts['parts'][1]=numeric_parts['parts'][0]
    with pytest.raises(ValueError,match='Duplicate/invalid'):train.assemble_inputs(directory,numeric_parts,lambda _:None)


def test_corrupted_part_rejected_before_values_are_used(tmp_path,numeric_parts):
    directory=tmp_path/'assembly';directory.mkdir();numeric_parts['parts'][0]['values_sha256']='0'*64
    with pytest.raises(ValueError,match='numeric part drift'):train.assemble_inputs(directory,numeric_parts,lambda _:None)


@pytest.fixture
def preprocessed(tmp_path,monkeypatch,numeric_parts):
    run=tmp_path/'native';directory=run/'attempts/aone';directory.mkdir(parents=True)
    now=time.time();plan={'stage':'prepare','identity_sha256':'fixture'}
    req={'definition':plan,'started_epoch':now,'deadline_epoch':now+900}
    (run/'request.json').write_text(json.dumps(req))
    monkeypatch.setattr(train,'request',lambda path:req)
    monkeypatch.setattr(train,'native_runtime',lambda load=False:None)
    monkeypatch.setattr(train.inputs,'verify_completed',lambda path:numeric_parts)
    monkeypatch.setattr(train.host,'resources',lambda path:{'available':1,'disk_free':100*2**30})
    result=train.calculate(run,directory,lambda _:None)
    receipt={'request_sha256':train.file_hash(run/'request.json'),'result_sha256':train.file_hash(directory/'result.json'),
             'result_path':'attempts/aone/result.json','status':'COMPLETE','elapsed_seconds':1.}
    return run,directory,receipt,result


def test_preprocessing_checkpoint_and_reload(preprocessed):
    run,directory,receipt,result=preprocessed
    assert result['exact_reload_transform_verified'] and result['classifier_fitted'] is False
    assert result['validation_test_read'] is False
    assert train.verify_payload(run,receipt)['rows']==6
    with (directory/'discrete.npy').open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(ValueError,match='artifact drift'):train.verify_payload(run,receipt)


def test_verified_completion_receipt_recovery_never_repeats_preprocessing(preprocessed,monkeypatch):
    run,directory,receipt,result=preprocessed
    train.publish(directory/'supervisor.json',train.canonical(receipt))
    before=(directory/'preprocessor.joblib').read_bytes()
    monkeypatch.setattr(train.subprocess,'Popen',lambda *a,**k:pytest.fail('Verified checkpoint launched a worker'))
    recovered=train.execute(run,recover=True)
    assert recovered['stage']=='prepare' and (directory/'preprocessor.joblib').read_bytes()==before
    assert 'recovered_supervisor_sha256' in recovered['resources']


def test_recovery_keeps_original_deadline(preprocessed,monkeypatch):
    run,directory,receipt,result=preprocessed
    req=train.request(run);req['started_epoch']-=1000;req['deadline_epoch']-=1000
    train.publish(directory/'supervisor.json',train.canonical(receipt))
    with pytest.raises(ValueError,match='time budget'):train.execute(run,recover=True)
    assert not (run/'complete.json').exists()


@pytest.mark.parametrize('failure',['rss','disk','artifacts'])
def test_recovery_retains_resource_guards(preprocessed,monkeypatch,failure):
    run,directory,receipt,result=preprocessed
    train.publish(directory/'supervisor.json',train.canonical(receipt))
    if failure=='rss':monkeypatch.setattr(train.psutil.Process,'memory_info',lambda self:type('Memory',(),{'rss':5*2**30})())
    elif failure=='disk':monkeypatch.setattr(train.host,'resources',lambda path:{'available':1,'disk_free':1})
    else:monkeypatch.setattr(train.host,'artifact_size',lambda path:6*2**30)
    with pytest.raises(ValueError):train.execute(run,recover=True)
    assert not (run/'complete.json').exists()


def test_unknown_class_or_unbound_preparation_rejected():
    with pytest.raises(ValueError):train.definition('class6',Path('unused'))
    with pytest.raises(ValueError):train.definition('class1')
    with pytest.raises(ValueError):train.definition('prepare',Path('unused'))
