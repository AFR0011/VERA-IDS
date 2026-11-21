"""C03 producer foundation: synthetic observations and disposable checkpoints only."""
import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from experiments.c03.producer import plan as p
from experiments.c03.producer.checkpoints import Checkpoints, publish
from experiments.c03.producer.sampling import Part, TrainingValidationReader


def profiles():
    result = {}
    for name in ('xgb_inv_family_clipped', 'rf_inv_family_clipped', 'rf_class_weight_balanced'):
        family = name.split('_')[0]
        result[name] = {'family': family, 'apply_loao_stage1': True,
                        'stage1_weight': name.split('_', 1)[1], 'stage2_weight': 'balanced',
                        'stage1_params': {'n_estimators': 8}, 'stage2_params': {'n_estimators': 9}}
    return result


@pytest.fixture
def plan():
    return {'status': 'NO_FIT_PLAN', 'execution_authorized': False,
            'graph': p.component_graph(profiles()), 'sources': {'fixture': 'a'*64},
            'environment': {'fixture': '1'},
            'expected_support': {'stage1_train': 4, 'stage2_train': 3}}


def bindings(plan):
    return {'preprocessor': 'a'*64, **{
        stage: {'rows': plan['expected_support'][f'{stage}_train'], 'row_ids': 'b'*64,
                'tensor': 'c'*64, 'labels': 'd'*64,
                'weights': {key: p.digest(None) for key, value in plan['graph']['components'].items()
                            if value['stage'] == stage}}
        for stage in ('stage1', 'stage2')}}


@pytest.fixture
def store(tmp_path, plan):
    store = Checkpoints.create(tmp_path/'outside', 'synthetic', plan)
    store.bind_inputs(bindings(plan))
    return store


def test_five_unique_components_and_literal_shared_rf_reference(plan):
    graph = plan['graph']
    assert len(graph['components']) == 5
    refs = graph['candidate_references']
    assert refs['rf_inv_family_clipped']['stage2'] == refs['rf_class_weight_balanced']['stage2']
    assert refs['rf_inv_family_clipped']['stage1'] != refs['rf_class_weight_balanced']['stage1']
    for node in graph['components'].values():
        assert node['seed'] == (123 if node['stage'] == 'stage1' else 133)


def test_shared_model_requires_identical_definition():
    values = profiles()
    values['rf_class_weight_balanced']['stage2_params'] = {'n_estimators': 10}
    with pytest.raises(ValueError, match='sharing'):
        p.component_graph(values)


@pytest.mark.parametrize('sizes,cap', [([3], 10), ([15], 4), ([2, 3, 12, 7], 6),
                                     ([0, 3, 0, 12], 6), ([2, 3], 5)])
@pytest.mark.parametrize('seed', [123, 124, 271])
def test_identity_sidecar_matches_actual_historical_sampler(monkeypatch, sizes, cap, seed):
    from ids_eval_framework._native import protocol_b_grid as legacy
    frames, parts = {}, []
    offset = 0
    for i, n in enumerate(sizes):
        name = f'train/part-{i}.parquet'
        parts.append(Part(name, p.digest(name)))
        frames[name] = pd.DataFrame({'Feature-A': np.arange(offset, offset+n, dtype=np.float64),
                                     'Family Name': [f'f{x%3}' for x in range(offset, offset+n)]})
        offset += n
    monkeypatch.setattr(legacy, 'list_parts', lambda *_: list(frames))
    monkeypatch.setattr(legacy, 'read_part', lambda path, usecols=None: frames[path].copy())
    expected = legacy.collect_split_frame('unused', 'train', list(next(iter(frames.values())).columns), cap, seed)
    reader = TrainingValidationReader({'train': tuple(parts), 'val': (Part('val/one', 'e'*64),)},
                                      lambda part: frames[part.name])
    sample = reader.sample('train', cap, seed)
    pd.testing.assert_frame_equal(sample.frame, expected)
    assert 'source_part' not in sample.frame
    for i, identity in sample.identities.iterrows():
        original = frames[identity.source_part].iloc[identity.row_offset]
        assert sample.frame.iloc[i]['feature-a'] == original['Feature-A']
        assert identity.source_sha256 == p.digest(identity.source_part)


def test_test_reader_refused_before_callback():
    calls = []
    reader = TrainingValidationReader({'train': (Part('train/one', 'a'*64),),
                                       'val': (Part('val/one', 'b'*64),)}, calls.append)
    with pytest.raises(ValueError, match='Test access'):
        reader.sample('test', 4, 123)
    assert not calls
    with pytest.raises(ValueError, match='Only train and val'):
        TrainingValidationReader({'test': ()}, calls.append)


def test_schema_drift_and_duplicate_sources_rejected():
    with pytest.raises(ValueError, match='repeats'):
        TrainingValidationReader({'train': (Part('same', 'a'*64),),
                                   'val': (Part('same', 'a'*64),)}, lambda _: None)
    frames = iter([pd.DataFrame({'a': [1]}), pd.DataFrame({'b': [2]})])
    reader = TrainingValidationReader({'train': (Part('a', 'a'*64), Part('b', 'b'*64)),
                                       'val': (Part('v', 'c'*64),)}, lambda _: next(frames))
    with pytest.raises(ValueError, match='schema'):
        reader.sample('train', 4, 123)


def test_complete_checkpoint_resumes_and_shares_one_rf_payload(store, plan):
    key = 'stage2_rf_shared'
    store.begin(key)
    store.complete(key, b'opaque synthetic payload')
    resumed = Checkpoints.resume(store.path.parent, store.path.name, plan)
    assert resumed.verify_component(key).read_bytes() == b'opaque synthetic payload'
    refs = plan['graph']['candidate_references']
    assert resumed.verify_component(refs['rf_inv_family_clipped']['stage2']) == resumed.verify_component(refs['rf_class_weight_balanced']['stage2'])


def test_existing_run_and_component_cannot_be_overwritten(store, plan):
    with pytest.raises(FileExistsError):
        Checkpoints.create(store.path.parent, store.path.name, plan)
    store.begin('stage2_rf_shared')
    with pytest.raises(FileExistsError):
        store.begin('stage2_rf_shared')
    store.complete('stage2_rf_shared', b'first')
    with pytest.raises(FileExistsError):
        store.complete('stage2_rf_shared', b'second')
    assert store.verify_component('stage2_rf_shared').read_bytes() == b'first'


def test_interrupted_component_and_missing_prerequisites_block(tmp_path, plan, store):
    empty = Checkpoints.create(tmp_path/'other', 'empty', plan)
    with pytest.raises(FileNotFoundError):
        empty.begin('stage2_rf_shared')
    store.begin('stage2_rf_shared')
    with pytest.raises(FileNotFoundError):
        Checkpoints.resume(store.path.parent, store.path.name, plan)


@pytest.mark.parametrize('target', ['payload', 'binding', 'completion'])
def test_checkpoint_corruption_blocks_resume(store, plan, target):
    key = 'stage2_rf_shared'
    store.begin(key)
    store.complete(key, b'first')
    directory = store.path/key
    if target == 'payload':
        (directory/'payload.bin').write_bytes(b'other')
    else:
        path = directory/('started.json' if target == 'binding' else 'complete.json')
        obj = json.loads(path.read_text())
        obj['plan_sha256' if target == 'binding' else 'payload_sha256'] = '0'*64
        path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        Checkpoints.resume(store.path.parent, store.path.name, plan)


@pytest.mark.parametrize('field', ['sources', 'environment'])
def test_changed_plan_blocks_resume(store, plan, field):
    changed = copy.deepcopy(plan)
    changed[field]['fixture'] = 'different'
    with pytest.raises(ValueError, match='drift'):
        Checkpoints.resume(store.path.parent, store.path.name, changed)


def test_run_identity_cannot_be_renamed_for_resume(store, plan):
    renamed = store.path.with_name('another-run')
    store.path.rename(renamed)
    with pytest.raises(ValueError, match='drift'):
        Checkpoints.resume(renamed.parent, renamed.name, plan)


def test_input_drift_and_wrong_support_block(store, plan):
    store.begin('stage2_rf_shared')
    store.complete('stage2_rf_shared', b'fixture')
    path = store.path/'inputs.json'
    obj = json.loads(path.read_text())
    obj['bindings']['stage2']['tensor'] = 'e'*64
    obj['bindings_sha256'] = p.digest(obj['bindings'])
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match='binding drift'):
        store.verify_component('stage2_rf_shared')
    wrong = bindings(plan)
    wrong['stage2']['rows'] = 7
    with pytest.raises(ValueError, match='support'):
        store.validate_inputs(wrong)


def test_writer_lock_blocks_concurrent_mutation(store):
    with store.lock():
        with pytest.raises(FileExistsError):
            store.begin('stage2_rf_shared')
    assert not (store.path/'.writer-lock').exists()


def test_checkpoint_path_escape_and_inside_repo_block(tmp_path, plan, store):
    with pytest.raises(ValueError, match='outside repository'):
        Checkpoints.create(p.ROOT/'not-allowed', 'run', plan)
    for name in ('../escape', '/absolute', 'C:drive', 'a/b'):
        with pytest.raises(ValueError):
            Checkpoints.create(tmp_path, name, plan)
        with pytest.raises(ValueError):
            store.begin(name)


def test_interruption_before_completion_never_publishes_complete(store, monkeypatch):
    import experiments.c03.producer.checkpoints as cp
    store.begin('stage2_rf_shared')
    real = cp.publish
    def fail_receipt(path, payload):
        if path.name == 'complete.json':
            raise OSError('simulated interruption')
        real(path, payload)
    monkeypatch.setattr(cp, 'publish', fail_receipt)
    with pytest.raises(OSError):
        store.complete('stage2_rf_shared', b'fixture')
    with pytest.raises(FileNotFoundError):
        store.verify_component('stage2_rf_shared')


def test_atomic_publish_does_not_clobber_existing(tmp_path):
    path = tmp_path/'receipt'
    publish(path, b'old')
    with pytest.raises(FileExistsError):
        publish(path, b'new')
    assert path.read_bytes() == b'old'
    assert not list(tmp_path.glob('.pending-*'))


def test_real_plan_is_metadata_only_and_detects_source_drift(monkeypatch):
    real_open = io.open
    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            name = str(file).lower()
            assert 'processed_v5' not in name
            assert not name.endswith(('.parquet', '.joblib', '.pkl', '.npy', '.npz'))
        return real_open(file, *args, **kwargs)
    monkeypatch.setattr(io, 'open', guarded_open)
    real = p.build_plan()
    assert real['execution_authorized'] is False
    assert len(real['graph']['components']) == 5
    original_hash = p.file_hash
    monkeypatch.setattr(p, 'file_hash', lambda path: '0'*64 if Path(path).name == 'protocol_b_grid.py' else original_hash(path))
    with pytest.raises(ValueError, match='Historical source'):
        p.build_plan()


def test_plan_environment_drift_rejected(monkeypatch):
    monkeypatch.setattr(p, 'version', lambda _: '0.0.0')
    with pytest.raises(ValueError, match='Environment version drift'):
        p.build_plan()


def test_default_cli_writes_no_artifacts(monkeypatch, capsys):
    from experiments.c03.producer import __main__ as cli
    monkeypatch.setattr(sys, 'argv', ['producer'])
    monkeypatch.setattr(cli.Checkpoints, 'create', lambda *_: pytest.fail('Unexpected write'))
    cli.main()
    assert json.loads(capsys.readouterr().out)['status'] == 'NO_FIT_PLAN'
