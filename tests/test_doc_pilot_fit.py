"""Generated fixtures for compact DOC checkpoint equivalence and fault recovery."""
import json
import os

import numpy as np
import pytest
torch = pytest.importorskip('torch', reason='Isolated DOC environment required')
from experiments.literature import doc_pilot_fit as fit, doc_training as training, doc_core as core

DEVICE = os.environ.get('DOC_TEST_DEVICE', 'cpu')


def data():
    rng = np.random.default_rng(17)
    return (rng.normal(size=(390,46)).astype('float32'), np.arange(390, dtype='int64') % 6,
            rng.normal(size=(131,46)).astype('float32'), np.arange(131, dtype='int64') % 6)


def call(root, **kwargs):
    return fit.fit_arrays(root, *data(), binding={'generated': True}, device=DEVICE, fixture=True,
                          epochs=3, checkpoint_updates=2, block_rows=128, **kwargs)


@pytest.mark.parametrize('event', ['after_update', 'weights_published', 'payload_published', 'commit_published', 'validation_block'])
def test_compact_fit_recovery_matches_existing_schedule(tmp_path, event):
    x, y, v, vy = data()
    expected_model, expected = training.synthetic_fit(tmp_path/'reference', x, y, v, vy,
        device=DEVICE, epochs=3, checkpoint_updates=2, block_rows=128)
    root = tmp_path/'recovery'; count = 0
    def interrupt(actual):
        nonlocal count
        if actual == event:
            count += 1
            if count == (3 if event == 'after_update' else 2):
                raise RuntimeError('injected interruption')
    with pytest.raises(RuntimeError, match='injected'):
        call(root, hook=interrupt)
    original_plan = (root/'plan.json').read_bytes()
    model, state, commit = call(root)
    assert (root/'plan.json').read_bytes() == original_plan
    assert state['history'] == expected['history'] and state['best_epoch'] == expected['best_epoch']
    assert state['updates'] == expected['updates']
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected_model.state_dict()[key])
    moments = core.CutoffMoments.empty(6)
    for start in range(0, len(x), 128):
        moments.update(training.probabilities(model, x[start:start+128], DEVICE), y[start:start+128])
    np.testing.assert_array_equal(moments.thresholds(), expected['thresholds'].numpy())
    assert len(list(root.glob('order-*.npy'))) == state['epoch']
    for path in root.glob('payload-*.pt'):
        payload = torch.load(path, weights_only=True)
        assert not {'order','model','optimizer','rng','best_model'} & payload.keys()
        assert path.stat().st_size < 10000
    loaded, again, final = call(root)
    assert final == commit and again['history'] == state['history']
    assert np.array_equal(training.probabilities(loaded, v, DEVICE), training.probabilities(model, v, DEVICE))


def test_reference_payload_and_order_drift_rejected(tmp_path):
    _, state, _ = call(tmp_path/'weights')
    path = tmp_path/'weights'/state['current_ref']['path']
    path.write_bytes(path.read_bytes()+b'drift')
    with pytest.raises(ValueError, match='payload drift'): call(tmp_path/'weights')
    root = tmp_path/'order'
    def stop(event):
        if event == 'after_update': raise RuntimeError('stop')
    with pytest.raises(RuntimeError): call(root, hook=stop)
    path = root/'order-000.npy'; values = np.load(path); values[0] = values[1]
    np.save(path, values)
    with pytest.raises(ValueError, match='order drift'): call(root)


def test_input_drift_and_expired_guard(tmp_path):
    call(tmp_path/'inputs')
    x,y,v,vy=data(); x[0,0] += 1
    with pytest.raises(ValueError, match='drift'):
        fit.fit_arrays(tmp_path/'inputs', x,y,v,vy, binding={'generated':True}, device=DEVICE,
                       fixture=True, epochs=3, checkpoint_updates=2, block_rows=128)
    def expired(): raise ValueError('deadline')
    with pytest.raises(ValueError, match='deadline'): call(tmp_path/'expired', guard=expired)
    assert not (tmp_path/'expired').exists()
    with pytest.raises(ValueError, match='scientific surface'):
        fit.fit_arrays(tmp_path/'real', *data(), binding={}, device=DEVICE)


def test_cutoff_checkpoints_replay_and_tamper(tmp_path):
    model, _, _ = call(tmp_path/'fit')
    (tmp_path/'request.json').write_text('{}'); (tmp_path/'fit-complete.json').write_text('{}')
    x,y,_,_=data()
    first=fit.cutoffs(tmp_path,model,x,y,device=DEVICE,guard=lambda:None,block=128)
    np.testing.assert_array_equal(first,fit.cutoffs(tmp_path,model,x,y,device=DEVICE,guard=lambda:None,block=128))
    path=tmp_path/'cutoffs/0000.json'; value=json.loads(path.read_text()); value['squares'][0]+=.1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='drift'):
        fit.cutoffs(tmp_path,model,x,y,device=DEVICE,guard=lambda:None,block=128)


@pytest.mark.skipif(DEVICE != 'cuda:0', reason='Integrated production inference requires GPU0')
def test_fit_seal_score_and_recovery_on_generated_arrays(tmp_path, monkeypatch):
    import time
    from experiments.literature import doc_pilot_test as scoring, doc_pilot_io as rt
    run=tmp_path/'integrated'; run.mkdir(); folder=run/'generated'; folder.mkdir()
    x,y,v,vy=data()
    for name,a in zip(('train-X','train-y','validation-X','validation-y'),(x,y,v,vy)):
        np.save(folder/(name+'.npy'),a)
    now=time.time(); plan={'started':now,'fit_deadline':now+7200,'deadline':now+8100,
                          'prepared_sha256':'generated','test':{'parts':[{'path':'test/generated.parquet','sha256':'generated'}]}}
    (run/'request.json').write_text(json.dumps(plan))
    monkeypatch.setattr(rt,'preparation',lambda p:(folder,{}))
    original=fit.fit_arrays
    def small(*args,**kwargs):
        return original(*args,**kwargs,fixture=True,epochs=3,checkpoint_updates=2,block_rows=128)
    monkeypatch.setattr(fit,'fit_arrays',small)
    fit.run_fit(run,plan,lambda *a,**k:None)
    sealed=rt.verify_seal(run,plan); seal_bytes=(run/'seal.json').read_bytes()
    fit.run_fit(run,plan,lambda *a,**k:None)
    assert (run/'seal.json').read_bytes()==seal_bytes
    (run/'test-inputs').mkdir(); scoring.save_arrays(run/'test-inputs/p0000.npz',X=v,truth=vy.astype('int8'))
    rt.bound_json(run,'test-inputs/p0000.json',{'seal_sha256':rt.file_hash(run/'seal.json'),
        'source':plan['test']['parts'][0],'rows':len(v),'arrays_sha256':rt.file_hash(run/'test-inputs/p0000.npz')})
    rt.bound_json(run,'test-inputs/complete.json',{'seal_sha256':rt.file_hash(run/'seal.json'),
        'parts':[rt.file_hash(run/'test-inputs/p0000.json')]})
    scoring.score(run,plan,lambda *a,**k:None)
    saved=(run/'test-scores/p0000.npz').read_bytes()
    scoring.score(run,plan,lambda *a,**k:None)
    assert (run/'test-scores/p0000.npz').read_bytes()==saved
    with np.load(run/'test-scores/p0000.npz') as a:
        np.testing.assert_array_equal(a['prediction'],core.predict(a['probabilities'],sealed['thresholds']))
    value=rt.read(run/'seal.json'); value['thresholds'][0]=.9
    (run/'seal.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='cutoff drift'): scoring.score(run,plan,lambda *a,**k:None)
