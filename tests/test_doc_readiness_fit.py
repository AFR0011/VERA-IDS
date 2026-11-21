"""Generated full-adapter fit fixtures in the isolated neural environment."""
import json
import os
import time

import numpy as np
import pytest
torch=pytest.importorskip('torch',reason='Isolated DOC neural environment required')
from experiments.literature import doc_readiness_fit as fit, doc_runtime as rt
from experiments.literature import doc_training as training

DEVICE=os.environ.get('DOC_TEST_DEVICE','cpu')


def fixture_run(root):
    root.mkdir();(root/'prepare').mkdir()
    now=time.time();plan={'started':now,'deadline':now+900}
    (root/'request.json').write_text(json.dumps(plan))
    rng=np.random.default_rng(73)
    for prefix,rows in [('train',390),('validation',131)]:
        np.save(root/'prepare'/f'{prefix}-X.npy',rng.normal(size=(rows,46)).astype('float32'),allow_pickle=False)
        np.save(root/'prepare'/f'{prefix}-y.npy',np.arange(rows,dtype=np.int64)%6,allow_pickle=False)
    record={'request_sha256':rt.file_hash(root/'request.json'),'rows':{'train':390,'validation':131},
            'class_support':{'train':[65]*6,'validation':[22,22,22,22,22,21]},
            'files':{p.name:rt.file_hash(p) for p in (root/'prepare').glob('*.npy')}}
    (root/'prepare/complete.json').write_text(json.dumps(record))
    return plan


@pytest.mark.parametrize('event',['after_update','validation_block','payload_published','commit_published'])
def test_full_adapter_recovery_without_cutoffs_or_test(tmp_path,event):
    baseline=tmp_path/'baseline';p=fixture_run(baseline)
    expected=fit.run_epoch(baseline,p,device=DEVICE,diagnostic=True)
    run=tmp_path/'resumed';p=fixture_run(run);count=0
    def interrupt(actual):
        nonlocal count
        if actual==event:
            count+=1
            if count==(3 if event=='after_update' else 2 if event.endswith('published') else 1):
                raise RuntimeError('interruption')
    with pytest.raises(RuntimeError,match='interruption'):
        fit.run_epoch(run,p,device=DEVICE,diagnostic=True,hook=interrupt)
    value=fit.run_epoch(run,p,device=DEVICE,diagnostic=True)
    assert value['updates']==4 and value['known_validation_bce']==expected['known_validation_bce']
    assert value['reload_prediction_exact'] and not value['cutoffs_fitted'] and not value['test_read']
    def last(path):
        commit=sorted((path/'fit').glob('commit-*.json'))[-1]
        payload=commit.with_name(commit.name.replace('commit-','payload-').replace('.json','.pt'))
        return torch.load(payload,weights_only=True,map_location='cpu')
    a,b=last(baseline),last(run)
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
    assert a['cursor']==b['cursor']==390 and a['val_cursor']==b['val_cursor']==131


def test_prepared_array_tamper_and_real_surface_reject_small_fixture(tmp_path):
    run=tmp_path/'run';plan=fixture_run(run)
    with pytest.raises(ValueError,match='Full-row GPU'):
        fit.run_epoch(run,plan,device=DEVICE)
    path=run/'prepare/train-X.npy'
    path.write_bytes(path.read_bytes()+b'drift')
    with pytest.raises(ValueError,match='Prepared file drift'):
        fit.run_epoch(run,plan,device=DEVICE,diagnostic=True)


def test_shared_outer_deadline_blocks_fit_before_new_journal(tmp_path):
    run=tmp_path/'run';plan=fixture_run(run)
    plan['started']-=1000;plan['deadline']-=1000
    with pytest.raises(ValueError,match='deadline'):
        fit.run_epoch(run,plan,device=DEVICE,diagnostic=True)
    assert not (run/'fit').exists()
