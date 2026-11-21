"""Fresh scientific preparation and full-count models on disposable synthetic data."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from experiments.c03.pilot import contract as c, preparation as prep, models as m, runner as r
from experiments.c03.engineering import lifecycle, resource_policy
from experiments.c03.producer import validation as v, models as native
from experiments.c03.producer.checkpoints import Checkpoints, publish
from experiments.c03.producer.sources import TrainingValidationFiles
from experiments.c03.producer.artifacts import read_arrays
from experiments.c03.producer.plan import canonical, digest, file_hash
from test_c03_producer_validation import fixture_plan, fixture_sources


def store_at(path, plan):
    path.mkdir()
    publish(path/'plan.json', canonical(plan))
    publish(path/'run.json', canonical({'schema_version': 1, 'run_id': path.name,
        'plan_sha256': digest(plan), 'scope': c.SCOPE}))
    return c.Store(path, plan)


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('scientific-pilot-generated')
    sources = fixture_sources(root/'sources')
    inventory = sources.inventory()
    for split, entries in inventory.items():
        for entry in entries:
            path = sources.root/entry['path']
            frame = pd.read_parquet(path)
            rng = np.random.RandomState(81)
            for column in range(44):
                frame[f'extra_{column:02d}'] = rng.normal(size=len(frame))
            frame.to_parquet(path, index=False)
            entry['sha256'] = file_hash(path)
    sources = TrainingValidationFiles(sources.root, inventory)
    old = Checkpoints.create(root, 'old', fixture_plan())
    original = v.prepare(old, sources)
    plan = c.frozen_plan()
    fixture = fixture_plan()
    plan['sampling'] = fixture['sampling']
    plan['expected_support'] = fixture['expected_support']
    for node in plan['graph']['components'].values():
        if node['model_family'] == 'xgb':
            node['parameters']['device'] = 'cpu'
    store = store_at(root/'fresh', plan)
    receipt = prep.prepare(store, sources, lambda _: None)
    return store, sources, receipt, original


def test_fresh_preparation_matches_verified_historical_algorithm(prepared):
    store, sources, receipt, original = prepared
    assert receipt['scope'] == c.SCOPE and receipt['fresh_preparation']
    for stage in ('stage1', 'stage2'):
        assert receipt['bindings'][stage] == v.input_bindings(original, store.plan)[stage]
    assert receipt['validation'] == v.validation_identity(original)
    assert receipt['preprocessor_statistics']['selected_rows'] > 0
    assert prep.verify(store) == receipt
    assert (sources.root/'test/DO_NOT_READ.parquet').read_bytes() == b'not a Parquet file'


def test_frozen_plan_preserves_counts_weights_and_has_no_test_execution():
    plan = c.frozen_plan()
    assert [plan['graph']['components'][n]['parameters']['n_estimators'] for n in c.COMPONENTS] == [300,300,800,300,1400]
    assert all(n['parameters']['n_jobs'] == 2 for n in plan['graph']['components'].values())
    assert plan['resource_limits']['start_available'] is None
    assert plan['resource_limits']['min_available'] is None
    assert plan['selection_authorized'] is False and plan['test_reader_authorized'] is False
    assert plan['graph']['candidate_references']['rf_class_weight_balanced']['stage2'] == 'stage2_rf_shared'


def test_cli_store_rejects_changed_scientific_definition(tmp_path):
    plan = c.frozen_plan()
    plan['sampling']['train_cap'] -= 1
    with pytest.raises(ValueError, match='Only the frozen'):
        c.Store.create(tmp_path, 'wrong', plan)
    assert not (tmp_path/'wrong').exists()


def test_real_start_creates_valid_run_id_and_original_deadline(tmp_path, monkeypatch):
    telemetry = {**r.gpu.load_readiness()['selected_gpu'], 'free': 7*2**30, 'used': 0}
    monkeypatch.setattr(r, 'OUTPUT_ROOT', tmp_path)
    monkeypatch.setattr(r.resource_policy, 'query_gpu_start', lambda: telemetry)
    monkeypatch.setattr(r.host, 'resources', lambda _: {'available': 0, 'disk_free': 40*2**30})
    captured = []
    def execute(store, recover=False):
        captured.append(store)
        assert recover is False
        return {'status': 'CREATED_FOR_GENERATED_TEST'}
    monkeypatch.setattr(r, 'execute', execute)
    assert r.start()['status'] == 'CREATED_FOR_GENERATED_TEST'
    store = captured[0]
    assert c.token(store.path.name) == store.path.name
    request = store.read_json(store.path/'request.json')
    assert request['deadline_epoch'] == request['started_epoch']+36000
    assert c.open_store(store.path).plan == store.plan


@pytest.fixture(scope='module')
def completed(prepared):
    store, _, _, _ = prepared
    stages = {}
    for name in c.COMPONENTS:
        fit_dir = store.path/'attempts'/(name+'-fit')/'agenerated'
        fit_dir.mkdir(parents=True)
        imported = m.imported(store, name, 'fit')
        X, y, weight = lifecycle.load_input(store.path/imported['array_path'], imported, name, 'fit')
        result = m.fit(store, fit_dir, name, imported, X, y, weight, lambda _: None, device='cpu')
        m.commit(store, fit_dir, name, 'fit', imported, result)
        assert m.verify(store, fit_dir, name, 'fit')['result'] == result
        score_dir = store.path/'attempts'/(name+'-score')/'agenerated'
        score_dir.mkdir(parents=True)
        imported_val = m.imported(store, name, 'score')
        Xv, _, _ = lifecycle.load_input(store.path/imported_val['array_path'], imported_val, name, 'score')
        scored = m.score(store, score_dir, fit_dir, result, name, Xv, lambda _: None, device='cpu')
        m.commit(store, score_dir, name, 'score', imported_val, scored, file_hash(fit_dir/'complete.json'))
        stages[name] = (fit_dir, score_dir)
    return store, stages


@pytest.mark.parametrize('name', c.COMPONENTS)
def test_full_count_generated_fit_reload_score_and_checkpoints(completed, name):
    store, stages = completed
    fit_dir, score_dir = stages[name]
    fit = m.verify(store, fit_dir, name, 'fit')
    scored = m.verify(store, score_dir, name, 'score', fit_dir)
    assert scored['result']['rows'] == 1750
    model = native.load_verified_local_model(fit_dir/fit['result']['model_file'], fit['result']['model_family'])
    if fit['result']['model_family'] == 'xgb':
        model.set_params(device='cpu', n_jobs=2, tree_method='hist')
    imported = m.imported(store, name, 'score')
    X, _, _ = lifecycle.load_input(store.path/imported['array_path'], imported, name, 'score')
    expected = native.probabilities(model, X, store.plan['graph']['components'][name]['stage'])
    actual = read_arrays(score_dir/'validation.npz')['probabilities']
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    assert r.checkpoint(store, name, 'fit') == fit_dir
    assert r.checkpoint(store, name, 'score', fit_dir) == score_dir


def test_forged_ensemble_count_and_corrupt_model_fail(completed, tmp_path):
    import shutil
    store, stages = completed
    name = c.COMPONENTS[0]
    destination = store.path/('tampered-'+tmp_path.name)
    shutil.copytree(stages[name][0], destination)
    record = store.read_json(destination/'complete.json')
    record['result']['effective_parameters']['n_estimators'] = 8
    (destination/'complete.json').write_bytes(canonical(record))
    with pytest.raises(ValueError, match='parameter drift'):
        m.verify(store, destination, name, 'fit')
    shutil.copyfile(stages[name][0]/'complete.json', destination/'complete.json')
    with (destination/'model.joblib').open('ab') as stream:
        stream.write(b'bad')
    with pytest.raises(ValueError, match='checksum drift'):
        m.verify(store, destination, name, 'fit')


def test_validation_corruption_rejected_before_model_load(completed, monkeypatch):
    store, stages = completed
    name = c.COMPONENTS[0]
    monkeypatch.setattr(native, 'load_verified_local_model', lambda *a: pytest.fail('Unsafe deserialization'))
    fit = store.read_json(stages[name][0]/'complete.json')['result']
    fit['model_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='checksum drift before load'):
        m.score(store, store.path, stages[name][0], fit, name, None, lambda _: None)


def test_recovery_reuses_fit_before_failed_scoring(prepared, tmp_path, monkeypatch):
    store, _, _, _ = prepared
    now = time.time()
    publish(store.path/'request.json', canonical({'plan_sha256': digest(store.plan), 'limits': c.LIMITS,
        'started_epoch': now, 'deadline_epoch': now+c.LIMITS['seconds'], 'gpu': {}}))
    dummy = store.path/'dummy-complete'
    dummy.mkdir()
    publish(dummy/'complete.json', b'fixture')
    def checkpoint(store, name, phase, fit=None):
        return dummy if phase in ('prepare', 'fit') else None
    calls = []
    def stage(store, name, phase, request, budget):
        calls.append((name, phase))
        return {'status': 'STOPPED', 'reason': 'injected score failure'}
    monkeypatch.setattr(r, 'checkpoint', checkpoint)
    monkeypatch.setattr(r, 'run_stage', stage)
    result = r.execute(store, recover=True)
    assert result['status'] == 'STOPPED'
    assert calls == [(c.COMPONENTS[0], 'score')]
    assert result['reused_stages'] == ['preparation:prepare', c.COMPONENTS[0]+':fit']


def test_resources_keep_all_non_available_ram_guards():
    state = {'available': 0, 'disk_free': 31*2**30}
    gpu = {'free': 7*2**30, 'used': 0}
    assert resource_policy.start_reason(state, gpu, limits=c.LIMITS) is None
    assert resource_policy.stop_reason(state, gpu, 1, 1, 35999, limits=c.LIMITS) is None
    assert resource_policy.stop_reason(state, gpu, 1, 1, 36000, limits=c.LIMITS)
    assert resource_policy.stop_reason(state, gpu, 5*2**30, 1, 1, limits=c.LIMITS)
    assert resource_policy.stop_reason(state, gpu, 1, 16*2**30, 1, limits=c.LIMITS)


def test_completed_scoring_does_not_grant_test_or_selection_access(completed):
    store, _ = completed
    from experiments.c03.producer.sealing import select_and_seal
    with pytest.raises(ValueError, match='Full experiment execution'):
        select_and_seal(store)
    assert not (store.path/'selection.json').exists()


def test_real_isolated_full_fit_and_score_workers(prepared, tmp_path, monkeypatch):
    import shutil
    import sys
    store, _, _, _ = prepared
    isolated = store_at(tmp_path/'isolated', store.plan)
    shutil.copytree(store.path/'prepared', isolated.path/'prepared')
    shutil.copyfile(store.path/'inputs.json', isolated.path/'inputs.json')
    now = time.time()
    telemetry = {**r.gpu.load_readiness()['selected_gpu'], 'free': 7*2**30, 'used': 0}
    request = {'plan_sha256': digest(store.plan), 'limits': c.LIMITS,
        'started_epoch': now, 'deadline_epoch': now+c.LIMITS['seconds'], 'gpu': r.gpu.gpu_identity(telemetry)}
    publish(isolated.path/'request.json', canonical(request))
    monkeypatch.setattr(r.gpu, 'query_gpu', lambda: telemetry)
    monkeypatch.setattr(r.host, 'resources', lambda _: {'available': 0, 'disk_free': 40*2**30})
    popen = r.subprocess.Popen
    script = '''
import sys,json
from pathlib import Path
from experiments.c03.pilot import contract as c,runner as r
run=Path(sys.argv[1])
c.frozen_plan=lambda: json.loads((run/'plan.json').read_text())
r.worker(run,sys.argv[2],sys.argv[3],sys.argv[4])
'''
    def launch(command, **kwargs):
        return popen([sys.executable, '-B', '-c', script, command[6], command[8], command[10], command[12]], **kwargs)
    monkeypatch.setattr(r.subprocess, 'Popen', launch)
    name = c.COMPONENTS[0]
    for phase in ('fit', 'score'):
        result = r.run_stage(isolated, name, phase, request, r.lifecycle_probe.Budget(request))
        assert result['status'] == 'COMPLETE', result
        assert result['worker_exit'] == 0 and result['peak_tree_rss'] > 0
    fit_dir = r.checkpoint(isolated, name, 'fit')
    score_dir = r.checkpoint(isolated, name, 'score', fit_dir)
    assert m.verify(isolated, score_dir, name, 'score', fit_dir)['result']['rows'] == 1750


@pytest.mark.parametrize('trigger', ['rss', 'disk', 'artifacts', 'gpu_used', 'gpu_free', 'deadline', 'query'])
def test_supervisor_stops_worker_on_each_limit(tmp_path, monkeypatch, trigger):
    store = store_at(tmp_path/'run', c.frozen_plan())
    telemetry = {**r.gpu.load_readiness()['selected_gpu'], 'free': 7*2**30, 'used': 0}
    request = {'gpu': r.gpu.gpu_identity(telemetry)}
    publish(store.path/'request.json', canonical(request))
    launched, stopped = [], []
    class Process:
        pid = 1234
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout): self.returncode = 15; return 15
    process = Process()
    def launch(*a, **k):
        launched.append(True)
        return process
    class Tree:
        def __init__(self, pid): pass
        def rss(self): return 5*2**30 if trigger == 'rss' else 1000
        def live(self): return []
        def stop(self): stopped.append(True); process.returncode = 15
    def gpu():
        state = telemetry.copy()
        if launched:
            if trigger == 'query': raise TimeoutError('GPU timeout')
            if trigger == 'gpu_used': state['used'] = 5*2**30
            if trigger == 'gpu_free': state['free'] = 1*2**30
        return state
    monkeypatch.setattr(r.subprocess, 'Popen', launch)
    monkeypatch.setattr(r.host, 'ProcessTree', Tree)
    monkeypatch.setattr(r.host, 'resources', lambda _: {'available': 0, 'disk_free': (9 if launched and trigger == 'disk' else 40)*2**30})
    monkeypatch.setattr(r.gpu, 'query_gpu', gpu)
    monkeypatch.setattr(r.host, 'artifact_size', lambda _: 16*2**30 if launched and trigger == 'artifacts' else 0)
    monkeypatch.setitem(c.LIMITS, 'gpu_interval', 0)
    class Budget:
        def elapsed(self): return 36001 if launched and trigger == 'deadline' else 0
    result = r.run_stage(store, c.COMPONENTS[0], 'fit', request, Budget())
    assert result['status'] == 'STOPPED' and result['reason']
    assert launched == [True] and stopped
    assert not list(store.path.glob('attempts/**/complete.json'))
