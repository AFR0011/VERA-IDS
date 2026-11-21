import json,time
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from experiments.literature import efc_fidelity as f


def test_native_metrics_match_label_semantics_and_exact_denominators():
    truth=np.array([0,0,1,2,3,4,5,6,6],dtype=np.int8)
    pred=np.array([6,1,6,2,3,4,5,6,0],dtype=np.int8)
    result=f.metrics(truth,pred)
    assert result['macro_f1']==pytest.approx(f1_score(truth,pred,labels=list(range(7)),average='macro',zero_division=0))
    assert result['udr']['numerator']==1 and result['udr']['denominator']==2
    assert result['fur_known']['numerator']==2 and result['fur_known']['denominator']==7
    assert result['fur_known_attacks']['numerator']==1 and result['fur_known_attacks']['denominator']==5
    assert result['benign_family_fpr']['value']==.5 and result['benign_rejection']['value']==.5
    assert 'stage1_attack_auroc' not in result and result['auc_not_reported_reason']


def test_absent_unknown_support_is_undefined_not_zero():
    result=f.metrics(np.array([0,1]),np.array([0,6]))
    assert result['udr']['value'] is None and result['udr']['denominator']==0


class Model:
    def predict(self,X,return_energies,unknown_class):
        assert X.shape==(3,1) and return_energies and unknown_class
        return np.array([-1,0,3]),np.array([9.,1.,2.])


def frame():
    return pd.DataFrame({'Feature':[1.,2.,3.],'y_stage1_attack':[1,0,1],
                         'y_stage2_family':['Botnet','Benign','DoS']})


def test_native_unknown_mapping_and_no_labels_in_predictors():
    y,p,e=f.score_frame(Model(),frame(),['feature'])
    np.testing.assert_array_equal(y,[6,0,3]);np.testing.assert_array_equal(p,[6,0,3])


def test_nonfinite_query_fails_without_imputation():
    data=frame();data.loc[0,'Feature']=np.inf
    with pytest.raises(ValueError,match='Nonfinite'):f.score_frame(Model(),data,['feature'])


def test_invalid_seal_blocks_model_loading_and_test_resolution(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();plan={'fit_record_sha256':'fixture','prediction_rule':'native'};now=time.time()
    req={'definition':plan,'started_epoch':now,'deadline_epoch':now+f.LIMITS['seconds']}
    (run/'request.json').write_text(json.dumps(req));(run/'seal.json').write_text('{}')
    monkeypatch.setattr(f,'definition',lambda:plan)
    monkeypatch.setattr(f,'load_model',lambda *a:pytest.fail('Invalid seal loaded a model'))
    with pytest.raises(ValueError,match='seal differs'):f.calculate(run,lambda _:None)


@pytest.fixture
def evaluated(tmp_path,monkeypatch):
    root=tmp_path/'data';(root/'test').mkdir(parents=True)
    data=frame();path=root/'test/part.parquet';data.to_parquet(path,index=False)
    entry={'path':'test/part.parquet','sha256':f.file_hash(path)}
    spec={'rows':3,'parts':[entry],'block_rows':8192,'minimum_per_class':0}
    population=f.PopulationHash(3,[entry]);population.add(entry,0,3)
    plan={'test':spec,'columns':['feature'],'expected_population_sha256':population.finish(),
          'expected_truth_sha256':f.array_hash(np.array([6,0,3],dtype=np.int8)),
          'comparison':'native only','uncertainty':'none'}
    run=tmp_path/'run';run.mkdir();(run/'seal.json').write_text('{}')
    (run/'request.json').write_text(json.dumps({'definition':plan}))
    req={'definition':plan,'started_epoch':time.time(),'deadline_epoch':time.time()+900}
    monkeypatch.setattr(f,'verify_seal',lambda p:req)
    monkeypatch.setattr(f,'load_model',lambda p:Model())
    monkeypatch.setattr(f,'verify_fit_records',lambda p:None)
    monkeypatch.setattr(f.host,'DATA_ROOT',root)
    result=f.calculate(run,lambda _:None)
    receipt={'evaluation_sha256':f.file_hash(run/'evaluation.json'),'elapsed_seconds':1.,'status':'COMPLETE'}
    (run/'complete.json').write_text(json.dumps(receipt))
    return run,result


def test_checkpoint_replay_reconciles_all_aggregates(evaluated,monkeypatch):
    run,result=evaluated
    assert f.verify_completed(run)['metrics']==result['metrics']
    before=(run/'parts/p0000/scores.npz').read_bytes()
    monkeypatch.setattr(f,'score_frame',lambda *a:pytest.fail('Completed test part rescored'))
    assert f.calculate(run,lambda _:None)==result
    assert (run/'parts/p0000/scores.npz').read_bytes()==before


def test_modified_aggregate_detected_even_with_new_outer_hash(evaluated):
    run,result=evaluated;result['metrics']['macro_f1']=.123
    (run/'evaluation.json').write_text(json.dumps(result))
    (run/'complete.json').write_text(json.dumps({'evaluation_sha256':f.file_hash(run/'evaluation.json'),
                                              'elapsed_seconds':1.,'status':'COMPLETE'}))
    with pytest.raises(ValueError,match='reconciliation'):f.verify_completed(run)


def test_corrupted_part_checkpoint_rejected(evaluated):
    run,result=evaluated;(run/'parts/p0000/scores.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='part drift'):f.verify_completed(run)


def test_expired_budget_prevents_inference_launch(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir();now=time.time()
    monkeypatch.setattr(f,'verify_seal',lambda p:{'started_epoch':now-1000,'deadline_epoch':now-100,'definition':{}})
    monkeypatch.setattr(f.host,'resources',lambda p:{'available':1,'disk_free':100*2**30})
    monkeypatch.setattr(f.subprocess,'Popen',lambda *a,**k:pytest.fail('Expired inference launched'))
    result=f.execute(run,recover=True)
    assert result['status']=='STOPPED' and 'time budget' in result['reason']
