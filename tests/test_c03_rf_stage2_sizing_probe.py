"""RF engineering import and bounded diagnostics on generated arrays only."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier

from experiments.c03.engineering import rf_stage2_sizing_probe as rf, preparation_probe as guard
from experiments.c03.producer.artifacts import array_hash, tensor_hash, identity_hash, write_arrays
from experiments.c03.producer.plan import build_plan, canonical, digest, file_hash
from experiments.c03.producer.validation import historical
from experiments.c03.producer.checkpoints import ENGINEERING_SCOPE
from test_c03_preparation_probe import ample, command


def fake_source(tmp_path, *, native_drift=False):
    source = tmp_path/'old-run'
    directory = source/'diagnostic/prepared'
    directory.mkdir(parents=True)
    current = build_plan()
    old = copy.deepcopy(current)
    old.update(status='ENGINEERING_PREPARATION_PLAN', evidence_scope=ENGINEERING_SCOPE)
    # The explicitly approved import permits producer revisions to differ.
    old['sources']['experiments/c03/producer/validation.py'] = 'a'*64
    if native_drift:
        old['sources']['src/ids_eval_framework/src/two_stage_engine.py'] = 'b'*64
    rng = np.random.RandomState(9)
    X = csr_matrix(rng.normal(size=(100, 3)))
    y = np.tile(np.arange(5, dtype=np.int64), 20)
    _, grid, _ = historical()
    weights = grid.stage2_sample_weights(y, 5, 'balanced')
    ids = pd.DataFrame({'source_part': ['train/fixture.parquet']*100,
        'source_sha256': ['c'*64]*100, 'row_offset': np.arange(100, dtype=np.int64)})
    binding = {'rows': 100, 'tensor': tensor_hash(X), 'labels': array_hash(y),
        'row_ids': identity_hash(ids), 'weights': {rf.COMPONENT: array_hash(weights)}}
    path = source/rf.ARRAY
    write_arrays(path, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape),
        target=y, weight_stage2_rf_shared=weights, source_part=ids.source_part.to_numpy(dtype=str),
        source_sha256=ids.source_sha256.to_numpy(dtype=str), row_offset=ids.row_offset.to_numpy())
    prep = {'scope': ENGINEERING_SCOPE, 'plan_sha256': digest(old), 'bindings': {'stage2': binding},
            'files': {'prepared/stage2.npz': file_hash(path)}}
    (directory/'complete.json').write_bytes(canonical(prep))
    source_record = {'plan_sha256': digest(old), 'probe_sources': {'old.py': 'd'*64}}
    (source/'request.json').write_bytes(canonical({'plan': old, 'source_record': source_record}))
    result = {'reload_verified': True, 'plan_sha256': digest(old), 'identities': {'bindings': {'stage2': binding}},
        'files': {rf.ARRAY: file_hash(path), 'diagnostic/prepared/complete.json': file_hash(directory/'complete.json')}}
    (source/'worker-result.json').write_bytes(canonical(result))
    (source/'supervisor.json').write_bytes(canonical({'status': 'COMPLETE'}))
    receipt = {'scope': ENGINEERING_SCOPE, 'scientific_checkpoint': False,
        **{key: file_hash(source/name) for name,key in [('request.json','request_sha256'),
            ('worker-result.json','worker_result_sha256'),('supervisor.json','supervisor_sha256')]}}
    (source/'complete.json').write_bytes(canonical(receipt))
    success = {'scope': ENGINEERING_SCOPE, 'status': 'VERIFIED_COMBINED_PREPARATION_COMPLETE',
        'run_id': source.name, 'source_record': source_record, 'identities': {'bindings': {'stage2': binding}},
        'evidence_hashes': {name: file_hash(source/name) for name in ['request.json','worker-result.json','supervisor.json','complete.json']}}
    return source, success, current, X, y, weights


def test_explicit_import_checks_old_lineage_and_exact_arrays(tmp_path, monkeypatch):
    source, success, current, X, y, weights = fake_source(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(np, 'load', lambda *_a, **_k: pytest.fail('Decoded array during byte verification'))
        record = rf.verify_source(source, success, current)
    assert record['source_plan_sha256'] != record['current_plan_sha256']
    assert record['scientific_checkpoint'] is False
    gotX, goty, gotw = rf.load_stage2(source/rf.ARRAY, record, 3)
    assert tensor_hash(gotX) == tensor_hash(X)
    np.testing.assert_array_equal(goty, y)
    np.testing.assert_array_equal(gotw, weights)


@pytest.mark.parametrize('target', ['request.json', rf.ARRAY, 'diagnostic/prepared/complete.json'])
def test_tampered_source_is_rejected_before_decoding(tmp_path, monkeypatch, target):
    source, success, current, *_ = fake_source(tmp_path)
    with (source/target).open('ab') as f:
        f.write(b'drift')
    monkeypatch.setattr(np, 'load', lambda *_a, **_k: pytest.fail('Decoded corrupted source'))
    with pytest.raises(ValueError, match='drift'):
        rf.verify_source(source, success, current)


def test_historical_helper_drift_not_allowed_by_import(tmp_path):
    source, success, current, *_ = fake_source(tmp_path, native_drift=True)
    with pytest.raises(ValueError, match='Historical helper'):
        rf.verify_source(source, success, current)


def test_array_identity_and_shape_checks(tmp_path):
    source, success, current, *_ = fake_source(tmp_path)
    record = rf.verify_source(source, success, current)
    with pytest.raises(ValueError, match='shape/dtype'):
        rf.load_stage2(source/rf.ARRAY, record, 46)
    record['expected_binding']['labels'] = '0'*64
    with pytest.raises(ValueError, match='identity drift'):
        rf.load_stage2(source/rf.ARRAY, record, 3)


def test_real_eight_tree_fit_has_no_scoring_or_model_serialization(tmp_path, monkeypatch):
    _, _, plan, X, y, weights = fake_source(tmp_path)
    def forbidden(*_a, **_k):
        pytest.fail('Scoring or model serialization called')
    monkeypatch.setattr(RandomForestClassifier, 'predict', forbidden)
    monkeypatch.setattr(RandomForestClassifier, 'predict_proba', forbidden)
    import joblib
    monkeypatch.setattr(joblib, 'dump', forbidden)
    original = copy.deepcopy(plan['graph']['components'][rf.COMPONENT])
    stages = []
    result = rf.fit_diagnostic(X, y, weights, plan, stages.append)
    assert len(result['tree_node_counts']) == 8 and result['total_tree_storage_bytes'] > 0
    assert result['quality_metrics'] is None and result['model_retained'] is False
    assert result['effective_parameters']['n_jobs'] == 1
    assert result['effective_parameters']['random_state'] == 133
    assert result['effective_parameters']['max_depth'] is None
    assert plan['graph']['components'][rf.COMPONENT] == original
    assert stages == ['fit_eight_trees', 'tree_storage_diagnostics']


def test_low_memory_gate_precedes_imported_array_access(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, 'source_record', lambda: {})
    monkeypatch.setattr(rf, 'load_success', lambda: {})
    monkeypatch.setattr(rf, 'OUTPUT_ROOT', tmp_path)
    monkeypatch.setattr(guard, 'resources', lambda _: {'available': 5*guard.GIB, 'disk_free': 30*guard.GIB})
    monkeypatch.setattr(rf, 'verify_source', lambda *_: pytest.fail('Source import started below gate'))
    monkeypatch.setattr(guard, 'supervise', lambda *_a, **_k: pytest.fail('Worker launched below gate'))
    assert rf.probe()['status'] == 'START_REFUSED'
    assert not list(tmp_path.iterdir())


def test_worker_refuses_changed_source_before_import(tmp_path, monkeypatch):
    (tmp_path/'request.json').write_bytes(canonical({'scope': rf.SCOPE, 'limits': guard.LIMITS, 'source_record': {}}))
    monkeypatch.setattr(rf, 'source_record', lambda: {'current': True})
    monkeypatch.setattr(rf, 'load_success', lambda: pytest.fail('Import resolved after drift'))
    with pytest.raises(ValueError, match='drift'):
        rf.worker(tmp_path)


@pytest.mark.parametrize('status,success', [('RF_SIZING_COMPLETE', True), ('PREPARATION_COMPLETE', False)])
def test_supervisor_requires_rf_specific_result(tmp_path, monkeypatch, status, success):
    monkeypatch.setattr(guard, 'resources', lambda _: ample())
    result = {'scope': rf.SCOPE, 'status': status, 'files': {}}
    code = 'import sys\nfrom pathlib import Path\n(Path(sys.argv[1])/"worker-result.json").write_text('+repr(json.dumps(result))+')\n'
    summary = guard.supervise(tmp_path, command(tmp_path, code), scope=rf.SCOPE)
    assert (summary['status'] == 'COMPLETE') == success
    assert (tmp_path/'complete.json').exists() == success


def test_resume_verifies_result_without_fit_and_rejects_incomplete(tmp_path, monkeypatch):
    request = {'scope': rf.SCOPE, 'source_record': {'revision': 'fixture'}}
    (tmp_path/'request.json').write_bytes(canonical(request))
    result = {'scope': rf.SCOPE, 'status': 'RF_SIZING_COMPLETE', 'files': {}}
    (tmp_path/'worker-result.json').write_bytes(canonical(result))
    (tmp_path/'supervisor.json').write_bytes(canonical({'status': 'COMPLETE'}))
    receipt = {'scope': rf.SCOPE, 'scientific_checkpoint': False,
        **{key: file_hash(tmp_path/name) for name,key in [('request.json','request_sha256'),
            ('worker-result.json','worker_result_sha256'),('supervisor.json','supervisor_sha256')]}}
    monkeypatch.setattr(rf, 'source_record', lambda: request['source_record'])
    monkeypatch.setattr(rf, 'fit_diagnostic', lambda *_: pytest.fail('Fit on completed resume'))
    with pytest.raises(FileNotFoundError):
        rf.verify_completed(tmp_path)
    (tmp_path/'complete.json').write_bytes(canonical(receipt))
    assert rf.verify_completed(tmp_path)['reused_completed_result'] is True
    (tmp_path/'worker-result.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='receipt drift'):
        rf.verify_completed(tmp_path)
