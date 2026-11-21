"""Synthetic DOC network/recovery checks; run in the dedicated PyTorch env."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch', reason='DOC synthetic checks require isolated PyTorch environment')
from experiments.literature import doc_training as doc
from experiments.literature.doc_core import CutoffMoments, predict

DEVICE = os.environ.get('DOC_TEST_DEVICE', 'cpu')


@pytest.fixture
def data():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(390, 46)).astype('float32')
    y = (np.arange(390) % 6).astype('int64')
    v = rng.normal(size=(131, 46)).astype('float32')
    vy = (np.arange(131) % 6).astype('int64')
    return x, y, v, vy


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def test_network_loss_and_best_model_cutoffs(tmp_path, data):
    model, state = doc.synthetic_fit(tmp_path, *data, device=DEVICE)
    assert sum(isinstance(x, torch.nn.Linear) for x in model) == 2
    assert not any(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in model)
    assert_same(doc.snapshot(model), state['best_model'])
    probs = doc.probabilities(model, data[0], DEVICE)
    m = CutoffMoments.empty(6); m.update(probs, data[1])
    np.testing.assert_array_equal(m.thresholds(), state['thresholds'].numpy())
    small = doc.probabilities(model, data[0], DEVICE, block=17)
    np.testing.assert_allclose(probs, small, atol=2e-7, rtol=2e-6)
    np.testing.assert_array_equal(predict(probs, m.thresholds()), predict(small, m.thresholds()))
    logits = torch.zeros((3, 6), device=DEVICE)
    np.testing.assert_allclose(doc.loss_rows(logits, torch.tensor([0,1,2], device=DEVICE)).cpu(), 6*np.log(2), rtol=1e-6)


@pytest.mark.parametrize('event', ['after_update', 'payload_published', 'commit_published', 'validation_block', 'cutoff_block'])
def test_interrupted_fit_recovers_identically(tmp_path, data, event):
    _, expected = doc.synthetic_fit(tmp_path/'baseline', *data, device=DEVICE, block_rows=128)
    target = tmp_path/'recovery'
    calls = 0
    def interrupt(actual):
        nonlocal calls
        if actual == event:
            calls += 1
            # Interrupt after uncheckpointed stochastic progress where possible.
            if calls == (3 if event == 'after_update' else 2):
                raise RuntimeError('injected interruption')
    with pytest.raises(RuntimeError, match='injected'):
        doc.synthetic_fit(target, *data, device=DEVICE, block_rows=128, hook=interrupt)
    plan_bytes = (target/'plan.json').read_bytes()
    _, actual = doc.synthetic_fit(target, *data, device=DEVICE, block_rows=128)
    assert (target/'plan.json').read_bytes() == plan_bytes
    assert_same(expected, actual)
    _, loaded = doc.synthetic_fit(target, *data, device=DEVICE, block_rows=128)
    assert_same(actual, loaded)


def test_input_and_payload_drift_fail(tmp_path, data):
    doc.synthetic_fit(tmp_path, *data, device=DEVICE)
    changed = data[0].copy(); changed[0,0] += 1
    with pytest.raises(ValueError, match='drift'):
        doc.synthetic_fit(tmp_path, changed, *data[1:], device=DEVICE)
    last = sorted(tmp_path.glob('payload-*.pt'))[-1]
    last.write_bytes(last.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='payload drift'):
        doc.synthetic_fit(tmp_path, *data, device=DEVICE)


def test_original_deadline_not_extended(tmp_path):
    j = doc.Journal(tmp_path, {'scope':'generated'}, seconds=900)
    with j.lock():
        j.open()
    p = tmp_path/'plan.json'; value = json.loads(p.read_text())
    value['started'] -= 1000; value['deadline'] -= 1000
    p.write_text(json.dumps(value))
    with j.lock():
        j.open()
        with pytest.raises(ValueError, match='deadline'):
            j.guard()


def test_exclusive_lock(tmp_path):
    j = doc.Journal(tmp_path, {'scope':'generated'})
    with j.lock():
        with pytest.raises(FileExistsError):
            with j.lock():
                pass


def test_early_stopping_keeps_earliest_tie_and_short_batch_weight(tmp_path, data, monkeypatch):
    # Frozen generated model isolates selection semantics from optimization.
    monkeypatch.setattr(torch.optim.Adam, 'step', lambda self: None)
    _, state = doc.synthetic_fit(tmp_path, *data, device=DEVICE, epochs=20, block_rows=128)
    assert state['epoch'] == 6 and state['best_epoch'] == 0 and state['bad_epochs'] == 5
    assert len(set(state['history'])) == 1
    model = doc.network(DEVICE); model.load_state_dict(state['best_model']); model.eval()
    with torch.no_grad():
        expected = doc.loss_rows(model(torch.as_tensor(data[2], device=DEVICE)),
                                 torch.as_tensor(data[3], device=DEVICE)).double().mean().item()
    assert abs(expected - state['best_loss']) < 1e-6
