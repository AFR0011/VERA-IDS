"""Two-worker RF comparison uses generated arrays only."""
import copy
import json

import pytest
from sklearn.ensemble import RandomForestClassifier

from experiments.c03.engineering import rf_two_worker_sizing_probe as two
from experiments.c03.producer.plan import canonical, file_hash
from test_c03_rf_stage2_sizing_probe import fake_source


def test_fixed_baseline_remains_verifiable():
    record = two.load_baseline()
    assert record['diagnostics']['effective_parameters']['n_jobs'] == 1
    assert record['source_record'] == two.base.source_record()


def test_import_requires_exact_baseline_inputs(tmp_path):
    source, success, plan, *_ = fake_source(tmp_path)
    old = two.base.verify_source(source, success, plan)
    baseline = {'import_record': old}
    imported = two.verify_import(source, success, plan, baseline)
    assert imported['overrides'] == {'n_estimators': 8, 'n_jobs': 2}
    assert old['overrides']['n_jobs'] == 1
    baseline['import_record'] = {**old, 'array_sha256': '0'*64}
    with pytest.raises(ValueError, match='Comparison input differs'):
        two.verify_import(source, success, plan, baseline)


def test_two_worker_fit_changes_only_concurrency_and_never_scores(tmp_path, monkeypatch):
    _, _, plan, X, y, weights = fake_source(tmp_path)
    def forbidden(*_a, **_k):
        pytest.fail('Unexpected scoring or serialization')
    monkeypatch.setattr(RandomForestClassifier, 'predict', forbidden)
    monkeypatch.setattr(RandomForestClassifier, 'predict_proba', forbidden)
    import joblib
    monkeypatch.setattr(joblib, 'dump', forbidden)
    # Tiny fits can finish within one Windows monotonic-clock tick. This check
    # exercises the comparison formula, not synthetic wall-clock performance.
    ticks = iter(range(10000))
    monkeypatch.setattr(two.time, 'monotonic', lambda: float(next(ticks)))
    original = copy.deepcopy(plan)
    baseline = {'diagnostics': two.base.fit_diagnostic(X, y, weights, plan, lambda _: None)}
    stages = []
    result = two.fit_diagnostic(X, y, weights, plan, baseline, stages.append)
    expected = {**baseline['diagnostics']['effective_parameters'], 'n_jobs': 2}
    assert result['effective_parameters'] == expected
    assert result['comparison']['same_tree_node_counts']
    assert result['comparison']['same_tree_storage_bytes']
    assert result['comparison']['fit_speedup_ratio'] > 0
    assert result['quality_metrics'] is None and result['model_retained'] is False
    assert stages == ['fit_eight_trees_two_workers', 'tree_storage_diagnostics']
    assert plan == original
    baseline['diagnostics']['effective_parameters']['max_depth'] = 7
    monkeypatch.setattr(RandomForestClassifier, 'fit', forbidden)
    with pytest.raises(ValueError, match='beyond worker count'):
        two.fit_diagnostic(X, y, weights, plan, baseline, stages.append)


def test_low_ram_prevents_worker_and_array_import(tmp_path, monkeypatch):
    monkeypatch.setattr(two, 'source_record', lambda: {})
    monkeypatch.setattr(two, 'load_baseline', lambda: {})
    monkeypatch.setattr(two.base, 'load_success', lambda: {})
    monkeypatch.setattr(two, 'OUTPUT_ROOT', tmp_path)
    monkeypatch.setattr(two.guard, 'resources', lambda _: {'available': 5*two.guard.GIB, 'disk_free': 30*two.guard.GIB})
    monkeypatch.setattr(two, 'verify_import', lambda *_: pytest.fail('Import before gate'))
    monkeypatch.setattr(two.guard, 'supervise', lambda *_a, **_k: pytest.fail('Worker before gate'))
    assert two.probe()['status'] == 'START_REFUSED'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('changed', ['definition', 'source_record', 'limits'])
def test_worker_refuses_request_drift_before_import(tmp_path, monkeypatch, changed):
    request = {'scope': two.SCOPE, 'limits': two.guard.LIMITS,
               'definition': two.DEFINITION, 'source_record': {'revision': 'fixture'}}
    monkeypatch.setattr(two, 'source_record', lambda: {'revision': 'fixture'})
    monkeypatch.setattr(two, 'load_baseline', lambda: pytest.fail('Import after drift'))
    request[changed] = {}
    (tmp_path/'request.json').write_bytes(canonical(request))
    with pytest.raises(ValueError, match='request/source drift'):
        two.worker(tmp_path)


def test_completed_resume_rejects_incomplete_corrupt_or_one_worker(tmp_path, monkeypatch):
    request = {'scope': two.SCOPE, 'limits': two.guard.LIMITS,
               'definition': two.DEFINITION, 'source_record': {'revision': 'fixture'}}
    (tmp_path/'request.json').write_bytes(canonical(request))
    result = {'scope': two.SCOPE, 'status': 'RF_SIZING_COMPLETE', 'scientific_checkpoint': False,
              'definition': two.DEFINITION, 'files': {}}
    (tmp_path/'worker-result.json').write_bytes(canonical(result))
    (tmp_path/'supervisor.json').write_bytes(canonical({'status': 'COMPLETE'}))
    receipt = {'scope': two.SCOPE, 'scientific_checkpoint': False,
        **{key: file_hash(tmp_path/name) for name,key in [('request.json','request_sha256'),
            ('worker-result.json','worker_result_sha256'),('supervisor.json','supervisor_sha256')]}}
    monkeypatch.setattr(two, 'source_record', lambda: request['source_record'])
    monkeypatch.setattr(two, 'fit_diagnostic', lambda *_: pytest.fail('Fit on resume'))
    with pytest.raises(FileNotFoundError):
        two.verify_completed(tmp_path)
    (tmp_path/'complete.json').write_bytes(canonical(receipt))
    assert two.verify_completed(tmp_path)['reused_completed_result'] is True
    (tmp_path/'worker-result.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='receipt drift'):
        two.verify_completed(tmp_path)
    request['definition'] = {**two.DEFINITION, 'n_jobs': 1}
    (tmp_path/'request.json').write_bytes(canonical(request))
    with pytest.raises(ValueError, match='source/definition'):
        two.verify_completed(tmp_path)
