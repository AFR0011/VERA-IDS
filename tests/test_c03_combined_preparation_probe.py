"""Engineering scope boundaries and shared preparation/reload on tiny fixtures."""
import json
import os
import weakref

import pandas as pd
import pytest

from experiments.c03.engineering import combined_preparation_probe as c
from experiments.c03.engineering import preparation_probe as guard
from experiments.c03.producer import validation as v, sealing, evaluator
from experiments.c03.producer.checkpoints import Checkpoints, ENGINEERING_SCOPE
from experiments.c03.producer.plan import canonical
from test_c03_producer_validation import fixture_plan, fixture_sources
from test_c03_preparation_probe import ample, command


def engineering_fixture(tmp_path):
    plan = fixture_plan()
    plan.update(status='ENGINEERING_PREPARATION_PLAN', evidence_scope=ENGINEERING_SCOPE)
    return Checkpoints.create(tmp_path/'runs', 'fixture', plan), fixture_sources(tmp_path/'sources')


def test_shared_preparation_roundtrip_without_fit_selection_or_test(tmp_path, monkeypatch):
    store, sources = engineering_fixture(tmp_path)
    def forbidden(*_a, **_k):
        pytest.fail('Classifier, selection or test access')
    _, grid, _ = v.historical()
    monkeypatch.setattr(grid, 'build_stage1_model', forbidden)
    monkeypatch.setattr(grid, 'build_stage2_model', forbidden)
    monkeypatch.setattr(sealing, 'select_and_seal', forbidden)
    monkeypatch.setattr(evaluator, 'read_test', forbidden)
    native_read = pd.read_parquet
    def read(path, **kwargs):
        assert str(path).replace('\\', '/').split('/')[-2] in ('train', 'val')
        return native_read(path, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', read)
    native_prepare, native_load = v.prepare_engineering, v.load_prepared
    references = []
    def prepare(*args, **kwargs):
        data = native_prepare(*args, **kwargs)
        references.extend(weakref.ref(x) for x in [data, *data.tensors.values(), data.validation])
        return data
    def load(*args, **kwargs):
        assert references and all(ref() is None for ref in references)
        return native_load(*args, **kwargs)
    monkeypatch.setattr(v, 'prepare_engineering', prepare)
    monkeypatch.setattr(v, 'load_prepared', load)
    stages = []
    result = c.prepare_and_reload(store, sources, stages.append)
    assert result['reload_verified'] is True
    assert result['identities']['validation']['count'] == 1750
    assert result['tensors']['stage1']['shape'] == [1525, 2]
    assert stages[-2:] == ['reload', 'reload_verified']
    assert v.verify_prepared(store)['scope'] == ENGINEERING_SCOPE
    assert not list(store.path.glob('stage*')) and not (store.path/'selection.json').exists()


def test_engineering_scope_cannot_enter_scientific_stages(tmp_path):
    store, sources = engineering_fixture(tmp_path)
    with pytest.raises(ValueError, match='synthetic'):
        v.produce_validation(store, sources)
    with pytest.raises(ValueError, match='Full fit'):
        v.fit_component(store, store.plan['graph']['fit_order'][0], None)
    with pytest.raises(ValueError, match='not authorized'):
        sealing.select_and_seal(store)
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluator.evaluate(store)
    synthetic = Checkpoints.create(tmp_path/'synthetic', 'fixture', fixture_plan())
    with pytest.raises(ValueError, match='scope'):
        v.prepare_engineering(synthetic, sources, lambda _: None)


@pytest.mark.parametrize('mutation', ['array', 'receipt_scope', 'input'])
def test_reload_rejects_corruption_and_scope_drift(tmp_path, mutation):
    store, sources = engineering_fixture(tmp_path)
    data = v.prepare_engineering(store, sources, lambda _: None)
    del data
    if mutation == 'array':
        with (store.path/'prepared/validation.npz').open('ab') as f:
            f.write(b'drift')
    else:
        path = store.path/('prepared/complete.json' if mutation == 'receipt_scope' else 'inputs.json')
        record = json.loads(path.read_text())
        if mutation == 'receipt_scope':
            record['scope'] = 'SYNTHETIC_ONLY'
        else:
            record['bindings_sha256'] = '0'*64
        path.write_bytes(canonical(record))
    with pytest.raises(ValueError):
        v.load_prepared(store)


def test_interrupted_serialization_has_no_complete_receipt(tmp_path, monkeypatch):
    store, sources = engineering_fixture(tmp_path)
    native = v.save_tensor
    def interrupted(path, *args):
        if path.name == 'stage2.npz':
            raise OSError('simulated write interruption')
        return native(path, *args)
    monkeypatch.setattr(v, 'save_tensor', interrupted)
    with pytest.raises(OSError, match='interruption'):
        c.prepare_and_reload(store, sources, lambda _: None)
    assert not (store.path/'prepared/complete.json').exists()
    assert not (store.path/'inputs.json').exists()
    assert not (store.path/'.writer-lock').exists()
    with pytest.raises(FileExistsError):
        v.prepare_engineering(store, sources, lambda _: None)


def test_combined_start_gate_never_starts_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'source_record', lambda: {})
    monkeypatch.setattr(c, 'engineering_plan', lambda: {})
    monkeypatch.setattr(c, 'inventory', lambda: {})
    monkeypatch.setattr(c, 'OUTPUT_ROOT', tmp_path/'runs')
    monkeypatch.setattr(guard, 'resources', lambda _: {'available': 5*guard.GIB, 'disk_free': 30*guard.GIB})
    monkeypatch.setattr(guard, 'supervise', lambda *_a, **_k: pytest.fail('Launched below gate'))
    assert c.probe()['status'] == 'START_REFUSED'
    assert not (tmp_path/'runs').exists()


def test_worker_rejects_unapproved_request_before_sources(tmp_path, monkeypatch):
    (tmp_path/'request.json').write_bytes(canonical({'scope': ENGINEERING_SCOPE, 'limits': guard.LIMITS,
        'source_record': {'drift': True}, 'plan': {}}))
    monkeypatch.setattr(c, 'source_record', lambda: {'current': True})
    monkeypatch.setattr(c, 'inventory', lambda: pytest.fail('Resolved sources after bad request'))
    with pytest.raises(ValueError, match='drift'):
        c.worker(tmp_path)


def test_disk_accounting_during_atomic_publication(tmp_path):
    temporary = tmp_path/'temporary'
    temporary.write_bytes(b'x'*123)
    os.link(temporary, tmp_path/'published')
    assert guard.artifact_size(tmp_path) == 123
    temporary.unlink()
    assert guard.artifact_size(tmp_path) == 123


def test_disk_accounting_tolerates_removed_temporary(tmp_path):
    class Missing:
        def stat(self):
            raise FileNotFoundError()
    class Snapshot:
        def rglob(self, _):
            return [Missing()]
    assert guard.artifact_size(Snapshot()) == 0


@pytest.mark.parametrize('reported_scope,success', [(ENGINEERING_SCOPE, True), (guard.SCOPE, False)])
def test_supervisor_binds_combined_scope(tmp_path, monkeypatch, reported_scope, success):
    monkeypatch.setattr(guard, 'resources', lambda _: ample())
    result = {'scope': reported_scope, 'status': 'PREPARATION_COMPLETE', 'files': {}}
    code = 'import sys\nfrom pathlib import Path\n(Path(sys.argv[1])/"worker-result.json").write_text('+repr(json.dumps(result))+')\n'
    summary = guard.supervise(tmp_path, command(tmp_path, code), scope=ENGINEERING_SCOPE)
    assert (summary['status'] == 'COMPLETE') == success
    assert (tmp_path/'complete.json').exists() == success
    assert summary['scope'] == ENGINEERING_SCOPE
