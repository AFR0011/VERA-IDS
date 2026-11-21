"""Batch 2 integration: disposable synthetic Parquet only, tiny CPU models."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from experiments.c03.producer.plan import build_plan, file_hash, digest
from experiments.c03.producer.checkpoints import Checkpoints
from experiments.c03.producer.sources import TrainingValidationFiles
from experiments.c03.producer import validation as v, sealing, models
from experiments.c03.producer.artifacts import read_arrays, array_hash, tensor_hash


def fixture_sources(root, *, perturb_unknown=False):
    inventory = {}
    for split, seed in [('train', 1), ('val', 2)]:
        rng = np.random.RandomState(seed)
        labels = np.repeat(['Benign', *v.FAMILIES, 'Botnet'], [400, 225, 225, 225, 225, 225, 225])
        encoded = np.repeat(np.arange(7), [400, 225, 225, 225, 225, 225, 225])
        frame = pd.DataFrame({'feature_one': encoded.astype(float) + rng.normal(0, .15, len(labels)),
                              'feature_two': rng.uniform(0, 1, len(labels)),
                              'y_stage1_attack': (labels != 'Benign').astype(int),
                              'y_stage2_family': labels})
        if split == 'val' and perturb_unknown:
            frame.loc[labels == 'Botnet', 'feature_one'] = -20.
        directory = root/split
        directory.mkdir(parents=True)
        inventory[split] = []
        # Interleaved files preserve all labels even in preprocessing prefix tests.
        for i in range(2):
            path = directory/f'part-{i}.parquet'
            frame.iloc[i::2].reset_index(drop=True).to_parquet(path, index=False)
            inventory[split].append({'path': path.relative_to(root).as_posix(), 'sha256': file_hash(path)})
    # Poison file: any attempt to load test observations will fail immediately.
    (root/'test').mkdir()
    (root/'test/DO_NOT_READ.parquet').write_bytes(b'not a Parquet file')
    return TrainingValidationFiles(root, inventory)


def fixture_plan():
    plan = build_plan()
    plan['status'] = 'SYNTHETIC_VALIDATION_PLAN'
    plan['evidence_scope'] = 'SYNTHETIC_ONLY'
    plan['sampling'].update(train_cap=1750, val_cap=1750)
    plan['expected_support'] = {'stage1_train': 1525, 'stage2_train': 1125,
                                'validation_K': 1525, 'validation_U': 225}
    for node in plan['graph']['components'].values():
        node['parameters'].update(n_estimators=3, max_depth=3, n_jobs=1)
        if node['model_family'] == 'xgb':
            node['parameters']['device'] = 'cpu'
    return plan


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp('c03-synthetic')
    sources = fixture_sources(root/'sources')
    plan = fixture_plan()
    store = Checkpoints.create(root/'runs', 'synthetic', plan)
    result = v.produce_validation(store, sources)
    return store, sources, result


def test_actual_historical_pipeline_five_components_and_both_arm_seal(completed):
    store, sources, result = completed
    assert result['scope'] == 'SYNTHETIC_ONLY'
    assert result['evaluation_authorized'] is False
    assert set(result['decisions']['arms']) == {'blind', 'visible'}
    assert len(list(store.path.glob('stage*/complete.json'))) == 5
    for arm in result['decisions']['arms'].values():
        assert set(arm['selections']) == {'0.01', '0.03', '0.05', '0.10'}
    assert (sources.root/'test/DO_NOT_READ.parquet').read_bytes() == b'not a Parquet file'
    assert len(result['dependencies']) >= 25


def test_fitted_models_reload_only_after_verified_receipts(completed):
    store, _, _ = completed
    data = v.load_prepared(store)
    for name, node in store.plan['graph']['components'].items():
        model = v.verified_model(store, name)
        actual = models.probabilities(model, data.validation, node['stage'])
        expected = read_arrays(store.path/f'{name}/validation.npz')['probabilities']
        np.testing.assert_array_equal(actual, expected)


def test_all_prepared_tensors_match_original_full_transform(completed):
    import joblib
    store, sources, _ = completed
    v.verify_prepared(store)
    prep = joblib.load(store.path/'prepared/preprocessor/preprocessor.joblib')
    data = v.load_prepared(store)
    train = sources.sampler().sample('train', 1750, 123)
    truth = v.truth_labels(train.frame)
    for stage, mask in {'stage1': truth != 6, 'stage2': (truth > 0) & (truth < 6)}.items():
        reference = prep.transform(train.frame.loc[mask].reset_index(drop=True).drop(
            columns=['y_stage1_attack', 'y_stage2_family']))
        assert tensor_hash(reference) == tensor_hash(data.tensors[stage])
    val = sources.sampler().sample('val', 1750, 124)
    reference = prep.transform(val.frame.drop(columns=['y_stage1_attack', 'y_stage2_family']))
    assert tensor_hash(reference) == tensor_hash(data.validation)
    np.testing.assert_array_equal(data.truth, v.truth_labels(val.frame))


def test_shared_rf_probability_array_is_literal_same_object(completed):
    store, _, _ = completed
    _, scores = v.validation_bundle(store)
    assert scores['rf_inv_family_clipped'].family_prob is scores['rf_class_weight_balanced'].family_prob


def test_resume_does_not_refit_or_rewrite(completed, monkeypatch):
    store, sources, first = completed
    monkeypatch.setattr(models, 'construct', lambda *_: pytest.fail('Unexpected refit'))
    before = file_hash(store.path/'selection-seal.json')
    assert v.produce_validation(store, sources, resume=True) == first
    assert file_hash(store.path/'selection-seal.json') == before


def test_no_full_fit_without_synthetic_scope(tmp_path):
    source = fixture_sources(tmp_path/'sources')
    plan = build_plan()
    store = Checkpoints.create(tmp_path/'runs', 'blocked', plan)
    with pytest.raises(ValueError, match='not authorized'):
        v.produce_validation(store, source)
    assert not (store.path/'prepared').exists()


def test_reordered_probability_columns_and_invalid_class_sets():
    class Model:
        classes_ = np.array([4, 1, 0, 3, 2])
        def predict_proba(self, X):
            return np.tile([.5, .2, .05, .15, .1], (len(X), 1))
    model = Model()
    got = models.probabilities(model, np.ones((2, 1)), 'stage2')
    np.testing.assert_allclose(got, [[.05, .2, .1, .15, .5]]*2)
    model.classes_ = np.array([0, 1, 1, 3, 4])
    with pytest.raises(ValueError, match='classes'):
        models.probabilities(model, np.ones((2, 1)), 'stage2')


def clone_store(completed, tmp_path):
    store, _, _ = completed
    target = tmp_path/'runs'/store.path.name
    shutil.copytree(store.path, target)
    return Checkpoints(target, store.plan)


@pytest.mark.parametrize('artifact', ['prepared/stage2.npz', 'stage2_rf_shared/model.joblib',
                                     'stage2_rf_shared/validation.npz', 'selection.json', 'selection-seal.json'])
def test_tampered_dependency_or_seal_blocks_before_deserialization(completed, tmp_path, monkeypatch, artifact):
    store = clone_store(completed, tmp_path)
    with (store.path/artifact).open('ab') as stream:
        stream.write(b'changed')
    monkeypatch.setattr(models, 'load_verified_local_model', lambda *_: pytest.fail('Unsafe model load'))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        sealing.verify_seal(store)
    if 'selection' not in artifact:
        with pytest.raises(ValueError):
            v.verified_model(store, 'stage2_rf_shared')


def test_missing_final_seal_is_not_complete(completed, tmp_path):
    store = clone_store(completed, tmp_path)
    (store.path/'selection-seal.json').unlink()
    with pytest.raises(FileNotFoundError):
        sealing.verify_seal(store)


def test_failure_publishing_final_seal_leaves_run_unsealed(completed, tmp_path, monkeypatch):
    store = clone_store(completed, tmp_path)
    (store.path/'selection-seal.json').unlink()
    (store.path/'selection.json').unlink()
    publish = sealing.publish
    def interrupted(path, data):
        if path.name == 'selection-seal.json':
            raise OSError('simulated interrupted seal publication')
        return publish(path, data)
    monkeypatch.setattr(sealing, 'publish', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        sealing.select_and_seal(store)
    assert (store.path/'selection.json').exists()
    with pytest.raises(FileNotFoundError):
        sealing.verify_seal(store)


def test_incomplete_arm_cannot_pass_rehashed_seal(completed, tmp_path):
    store = clone_store(completed, tmp_path)
    path = store.path/'selection.json'
    content = json.loads(path.read_text())
    del content['decisions']['arms']['visible']
    core, _, _ = v.historical()
    content['decision_sha256'] = core.sha_record(content['decisions'])
    path.write_bytes(core.strict_json(content))
    seal = store.read_json(store.path/'selection-seal.json')
    seal['selection_sha256'] = file_hash(path)
    (store.path/'selection-seal.json').write_text(json.dumps(seal))
    with pytest.raises(RuntimeError, match='both arms'):
        sealing.verify_seal(store)


def test_live_tensor_mutation_blocks_fit(completed, tmp_path, monkeypatch):
    store = clone_store(completed, tmp_path)
    data = v.load_prepared(store)
    data.tensors['stage2'].data[0] += .1
    monkeypatch.setattr(models, 'construct', lambda *_: pytest.fail('Should not construct'))
    with pytest.raises(ValueError, match='Live input'):
        v.fit_component(store, 'stage2_rf_shared', data)


def test_source_bytes_and_test_inventory_rejected(tmp_path):
    source = fixture_sources(tmp_path/'sources')
    with pytest.raises(ValueError, match='Only train/val'):
        TrainingValidationFiles(source.root, {'train': [], 'test': []})
    path = source.root/source.parts['train'][0].name
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='Source byte drift'):
        source.verify()


@pytest.mark.parametrize('receipt', ['prepared/complete.json', 'stage2_rf_shared/complete.json',
                                    'selection-seal.json'])
def test_interrupted_stage_never_auto_retries(completed, tmp_path, monkeypatch, receipt):
    store = clone_store(completed, tmp_path)
    (store.path/receipt).unlink()
    monkeypatch.setattr(models, 'construct', lambda *_: pytest.fail('Unexpected retry'))
    with pytest.raises(FileNotFoundError):
        v.produce_validation(store, completed[1], resume=True)


def test_revision_drift_blocks_before_source_read(completed, tmp_path, monkeypatch):
    store = clone_store(completed, tmp_path)
    name = next(iter(store.plan['sources']))
    store.plan['sources'][name] = '0'*64
    monkeypatch.setattr(v, 'fixture_gate', lambda *_: pytest.fail('Gate reached after source drift'))
    with pytest.raises(ValueError, match='revision drift'):
        v.produce_validation(store, completed[1], resume=True)


def test_insufficient_support_stops_before_model_construction(tmp_path, monkeypatch):
    source = fixture_sources(tmp_path/'sources')
    entry = source.parts['val'][0]
    path = source.root/entry.name
    frame = pd.read_parquet(path)
    frame.loc[frame.y_stage2_family == 'BruteForce', 'y_stage2_family'] = 'DDoS'
    frame.to_parquet(path, index=False)
    inventory = source.inventory()
    inventory['val'][0]['sha256'] = file_hash(path)
    source = TrainingValidationFiles(source.root, inventory)
    store = Checkpoints.create(tmp_path/'runs', 'undersupported', fixture_plan())
    monkeypatch.setattr(models, 'construct', lambda *_: pytest.fail('Unexpected model construction'))
    with pytest.raises(ValueError, match='class support'):
        v.produce_validation(store, source)


def test_unknown_only_feature_change_preserves_models_and_K_scores(completed, tmp_path):
    before, _, _ = completed
    source = fixture_sources(tmp_path/'changed', perturb_unknown=True)
    store = Checkpoints.create(tmp_path/'runs', 'changed', fixture_plan())
    v.produce_validation(store, source)
    truth, first = v.validation_bundle(before)
    truth2, second = v.validation_bundle(store)
    np.testing.assert_array_equal(truth, truth2)
    for name in first:
        np.testing.assert_array_equal(first[name].attack_prob[truth != 6], second[name].attack_prob[truth != 6])
        np.testing.assert_array_equal(first[name].family_prob[truth != 6], second[name].family_prob[truth != 6])
    for name, node in store.plan['graph']['components'].items():
        extension = 'ubj' if node['model_family'] == 'xgb' else 'joblib'
        assert file_hash(before.path/f'{name}/model.{extension}') == file_hash(store.path/f'{name}/model.{extension}')
    assert sealing.verify_seal(before)['decisions']['arms']['blind'] == sealing.verify_seal(store)['decisions']['arms']['blind']
