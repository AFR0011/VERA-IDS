"""Disposable generated inputs only; fit/reload/chunking and receipt boundaries."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from experiments.c03.engineering import lifecycle as c
from experiments.c03.producer import models, validation as v, isolated
from experiments.c03.producer.artifacts import array_hash, tensor_hash, identity_hash, read_arrays
from experiments.c03.producer.checkpoints import Checkpoints
from experiments.c03.producer.plan import build_plan, digest, file_hash


def inputs(root, name, phase):
    rng = np.random.RandomState(73)
    rows = 280
    latent = np.arange(rows, dtype=np.int64) % (7 if phase == 'score' else 6 if name.startswith('stage1') else 5)
    if phase == 'fit' and name.startswith('stage2'):
        latent += 1
    y = latent if phase == 'score' else (latent != 0).astype(np.int64) if name.startswith('stage1') else latent-1
    features = rng.normal(size=(rows, 46))
    features[:, 0] = latent + rng.normal(0, .05, rows)
    features[:, 1] = (latent != 0).astype(float) + rng.normal(0, .01, rows)
    X = csr_matrix(features)
    ids = pd.DataFrame({'source_part': ['generated']*rows, 'source_sha256': ['0'*64]*rows,
                        'row_offset': np.arange(rows, dtype=np.int64)})
    weight = None if phase == 'score' or name == c.COMPONENTS[0] else np.ones(rows, dtype=np.float64)
    path = root/(phase+'.npz')
    if not path.exists():
        v.save_tensor(path, X, y, ids, {name: weight})
    expected = ({'rows': rows, 'tensor': tensor_hash(X), 'labels': array_hash(y), 'row_ids': identity_hash(ids),
                 'weights': {name: digest(None) if weight is None else array_hash(weight)}} if phase == 'fit' else
                {'count': rows, 'tensor': tensor_hash(X), 'truth': array_hash(y), 'rows': identity_hash(ids)})
    imported = {'scope': c.SCOPE, 'scientific_checkpoint': False, 'component': name, 'phase': phase,
                'array_sha256': file_hash(path), 'expected': expected}
    return X, y, weight, path, imported


@pytest.fixture(scope='module')
def five(tmp_path_factory):
    root = tmp_path_factory.mktemp('lifecycle-generated')
    plan = build_plan()
    results = {}
    for name in c.COMPONENTS:
        directory = root/name
        directory.mkdir()
        X, y, weight, path, imported = inputs(directory, name, 'fit')
        fit_dir = directory/'fit'
        fit_dir.mkdir()
        fit = c.fit_save(fit_dir, plan, name, imported, X, y, weight, lambda _: None, device='cpu')
        c.commit_phase(fit_dir, name, 'fit', 'a'*64, imported, fit)
        score_dir = directory/'score'
        score_dir.mkdir()
        Xv, _, _, _, vi = inputs(directory, name, 'score')
        score = c.reload_score(score_dir, fit_dir, fit, name, Xv, lambda _: None, device='cpu')
        c.commit_phase(score_dir, name, 'score', 'a'*64, vi, score, fit_receipt_sha=file_hash(fit_dir/'complete.json'))
        results[name] = (directory, fit, score)
    return results


@pytest.mark.parametrize('name', c.COMPONENTS)
def test_five_models_reload_and_block_equivalence(five, name):
    directory, fit, score = five[name]
    c.verify_phase(directory/'fit', name, 'fit', 'a'*64)
    c.verify_phase(directory/'score', name, 'score', 'a'*64,
                   fit_receipt_sha=file_hash(directory/'fit/complete.json'))
    X, _, _, _, _ = inputs(directory, name, 'score')
    model = models.load_verified_local_model(directory/'fit'/fit['model_file'], fit['model_family'])
    if fit['model_family'] == 'xgb':
        model.set_params(device='cpu', n_jobs=2, tree_method='hist')
    whole = models.probabilities(model, X, 'stage1' if name.startswith('stage1') else 'stage2')
    chunked = c.score_blocks(model, X, name, lambda _: None, block_rows=37)
    # RF summation can differ in its final bit with two threads. No estimator or
    # threshold change; use a strict absolute tolerance and exact XGB equality.
    if fit['model_family'] == 'xgb':
        np.testing.assert_array_equal(chunked, whole)
    else:
        np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-12)
    np.testing.assert_allclose(read_arrays(directory/'score/validation.npz')['probabilities'], whole, rtol=0, atol=1e-12)
    assert score['quality_metrics'] is None and score['rows'] == 280


@pytest.mark.parametrize('phase', ['fit', 'score'])
@pytest.mark.parametrize('name', c.COMPONENTS)
def test_loader_only_opens_requested_array(tmp_path, monkeypatch, name, phase):
    X, y, weight, path, imported = inputs(tmp_path, name, phase)
    native, opened = c.read_arrays, []
    def read(p):
        opened.append(p)
        return native(p)
    monkeypatch.setattr(c, 'read_arrays', read)
    actual, target, sample_weight = c.load_input(path, imported, name, phase)
    assert opened == [path] and tensor_hash(actual) == tensor_hash(X)
    np.testing.assert_array_equal(target, y)
    if weight is not None:
        np.testing.assert_array_equal(sample_weight, weight)


@pytest.mark.parametrize('field', ['tensor', 'labels', 'row_ids', 'weights'])
def test_input_identity_corruption_rejected(tmp_path, field):
    name = c.COMPONENTS[1]
    _, _, _, path, imported = inputs(tmp_path, name, 'fit')
    imported['expected'][field] = {name: 'f'*64} if field == 'weights' else 'f'*64
    with pytest.raises(ValueError, match='identity drift'):
        c.load_input(path, imported, name, 'fit')


def test_model_corruption_rejected_before_deserialization(five, tmp_path, monkeypatch):
    import shutil
    name = c.COMPONENTS[0]
    directory, fit, _ = five[name]
    shutil.copytree(directory/'fit', tmp_path/'fit')
    with (tmp_path/'fit'/fit['model_file']).open('ab') as stream:
        stream.write(b'corrupt')
    monkeypatch.setattr(models, 'load_verified_local_model', lambda *_: pytest.fail('Unsafe model load'))
    with pytest.raises(ValueError, match='checksum drift'):
        c.reload_score(tmp_path, tmp_path/'fit', fit, name, None, lambda _: None)


def test_receipt_drift_and_score_mutation_fail(five, tmp_path):
    import shutil
    name = c.COMPONENTS[0]
    directory, _, _ = five[name]
    shutil.copytree(directory/'score', tmp_path/'score')
    with pytest.raises(ValueError, match='binding drift'):
        c.verify_phase(tmp_path/'score', name, 'score', 'b'*64)
    with (tmp_path/'score/validation.npz').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='Score checksum drift'):
        c.verify_phase(tmp_path/'score', name, 'score', 'a'*64,
                       fit_receipt_sha=file_hash(directory/'fit/complete.json'))


def test_partial_save_is_not_complete(tmp_path, monkeypatch):
    name = c.COMPONENTS[0]
    X, y, w, _, imported = inputs(tmp_path, name, 'fit')
    def fail(*_):
        raise OSError('Interrupted model save')
    monkeypatch.setattr(models, 'save_local_model', fail)
    with pytest.raises(OSError, match='Interrupted'):
        c.fit_save(tmp_path, build_plan(), name, imported, X, y, w, lambda _: None)
    assert not (tmp_path/'complete.json').exists()


def test_new_scope_cannot_enter_scientific_or_synthetic_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace
    plan = build_plan()
    plan.update(status='ENGINEERING_REHEARSAL', evidence_scope=c.SCOPE)
    with pytest.raises(ValueError, match='Only no-fit'):
        Checkpoints.create(tmp_path/'runs', 'rehearsal', plan)
    store = SimpleNamespace(plan=plan)
    with pytest.raises(ValueError, match='Only synthetic'):
        isolated.synthetic_gate(store)
    with pytest.raises(ValueError, match='Full fit not authorized'):
        v._fit_component(store, c.COMPONENTS[0], None)
    from experiments.c03.producer import sealing, evaluator
    with pytest.raises(ValueError, match='Full experiment execution'):
        sealing.select_and_seal(store)
    # Even if seal verification returned metadata carrying this new scope,
    # the evaluator must reject it before reaching a test reader.
    plan['synthetic_evaluation_contract'] = {'scope': c.SCOPE}
    monkeypatch.setattr(evaluator, 'verify_seal', lambda _: {'synthetic_evaluation_contract': {'scope': c.SCOPE}})
    monkeypatch.setattr(evaluator, 'read_test', lambda _: pytest.fail('Test observations accessed'))
    with pytest.raises(ValueError, match='No prospective synthetic'):
        evaluator.evaluate(store)


def test_scoring_failure_preserves_fit_receipt(five, tmp_path, monkeypatch):
    directory, fit, _ = five[c.COMPONENTS[0]]
    before = file_hash(directory/'fit/complete.json')
    def fail(*a, **k):
        raise RuntimeError('interrupted scoring')
    monkeypatch.setattr(c, 'score_blocks', fail)
    X, _, _, _, _ = inputs(tmp_path, c.COMPONENTS[0], 'score')
    with pytest.raises(RuntimeError, match='interrupted scoring'):
        c.reload_score(tmp_path, directory/'fit', fit, c.COMPONENTS[0], X, lambda _: None)
    assert file_hash(directory/'fit/complete.json') == before
    assert not (tmp_path/'complete.json').exists()


def test_bad_import_scope_rejected_before_file_access(tmp_path):
    with pytest.raises(ValueError, match='Invalid preparation scope'):
        c.verify_import(tmp_path, {'scope': 'SCIENTIFIC'}, build_plan(), c.COMPONENTS[0], 'fit')


def test_generated_selection_is_identical_across_scoring_blocks(five):
    from experiments.c03 import c03_paired_runner_core as selectors
    whole, blocked = {}, {}
    plan = build_plan()
    for candidate, refs in plan['graph']['candidate_references'].items():
        arrays = []
        for name in (refs['stage1'], refs['stage2']):
            directory, fit, _ = five[name]
            X, truth, _, _, _ = inputs(directory, name, 'score')
            model = models.load_verified_local_model(directory/'fit'/fit['model_file'], fit['model_family'])
            if fit['model_family'] == 'xgb':
                model.set_params(device='cpu', n_jobs=2, tree_method='hist')
            arrays.append((c.score_blocks(model, X, name, lambda _: None, block_rows=X.shape[0]),
                           c.score_blocks(model, X, name, lambda _: None, block_rows=37)))
        whole[candidate] = selectors.CandidateScores(arrays[0][0], arrays[1][0])
        blocked[candidate] = selectors.CandidateScores(arrays[0][1], arrays[1][1])
    # Use the actual existing selector entrypoint, with a generated support ledger.
    # Repeat generated observations for the selector's >=200-per-family support
    # gate; no benchmark observations or additional fits are involved.
    truth = np.tile(truth, 7)
    for bundle in (whole, blocked):
        for key, value in list(bundle.items()):
            bundle[key] = selectors.CandidateScores(np.tile(value.attack_prob, 7), np.tile(value.family_prob, (7, 1)))
    _, grid, engine = v.historical()
    a = selectors.select_both_arms(truth, whole, grid=grid, engine=engine)
    b = selectors.select_both_arms(truth, blocked, grid=grid, engine=engine)
    # The historical threshold helper can return undefined search metadata when
    # the eight-tree fixture has tied benign scores. Compare all decisions
    # exactly, treating corresponding NaNs as the same undefined value. This
    # does not change the producer's existing rejection of NaNs at sealing.
    def comparable(value):
        if isinstance(value, dict):
            return {k: comparable(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [comparable(v) for v in value]
        return 'UNDEFINED_NAN' if isinstance(value, float) and np.isnan(value) else value
    assert comparable(a['arms']) == comparable(b['arms'])


def test_cuda_fallback_is_rejected_without_gpu_execution(five):
    name = c.COMPONENTS[2]
    directory, fit, _ = five[name]
    model = models.load_verified_local_model(directory/'fit'/fit['model_file'], 'xgb')
    model.set_params(device='cpu', n_jobs=2, tree_method='hist')
    with pytest.raises(ValueError, match='CPU fallback|device'):
        c.check_model(model, name, device='cuda:0')
