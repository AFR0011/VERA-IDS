"""Post-seal evaluator tests; all sources/models are generated synthetic fixtures."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from test_c03_producer_validation import fixture_plan, fixture_sources
from experiments.c03.producer import evaluator as e, validation as v
from experiments.c03.producer.checkpoints import Checkpoints
from experiments.c03.producer.plan import file_hash, ROOT


@pytest.fixture(scope='module')
def sealed(tmp_path_factory):
    root = tmp_path_factory.mktemp('c03-evaluator')
    sources = fixture_sources(root/'sources')
    testroot = root/'test-fixture'
    testroot.mkdir()
    labels = ['Benign', *v.FAMILIES, 'Botnet']*3
    frame = pd.DataFrame({'feature_one': np.tile(np.arange(7), 3).astype(float),
                          'feature_two': np.full(21, .5),
                          'y_stage1_attack': [int(x != 'Benign') for x in labels],
                          'y_stage2_family': labels})
    path = testroot/'test.parquet'
    frame.to_parquet(path, index=False)
    plan = fixture_plan()
    plan['synthetic_evaluation_contract'] = {'scope': 'SYNTHETIC_ONLY', 'root': str(testroot),
        'evaluator_revision': file_hash(e.__file__),
        'parts': [{'path': path.name, 'sha256': file_hash(path), 'rows': len(frame)}]}
    store = Checkpoints.create(root/'runs', 'fixture', plan)
    v.produce_validation(store, sources)
    return store


def clone(sealed, tmp_path):
    import shutil
    path = tmp_path/sealed.path.name
    shutil.copytree(sealed.path, path)
    return Checkpoints(path, sealed.plan)


def test_separate_process_evaluator(sealed, tmp_path):
    store = clone(sealed, tmp_path)
    proc = subprocess.run([sys.executable, '-B', '-m', 'experiments.c03.producer.evaluator',
                           '--run', str(store.path)], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((store.path/'evaluation.json').read_text())
    assert result['scope'] == 'SYNTHETIC_ONLY' and result['rows'] == 21
    assert len(result['records']) == 16
    receipt = json.loads((store.path/'evaluation-complete.json').read_text())
    assert receipt['evaluation_sha256'] == file_hash(store.path/'evaluation.json')
    for pair in result['records'].values():
        for arm in pair['arms'].values():
            if arm['status'] == 'feasible':
                assert sum(map(sum, arm['metrics']['confusion_matrix'])) == 21


@pytest.mark.parametrize('mutation', ['missing', 'tamper', 'model', 'revision'])
def test_failed_seal_precedes_any_test_reader(sealed, tmp_path, monkeypatch, mutation):
    store = clone(sealed, tmp_path)
    if mutation == 'missing':
        (store.path/'selection-seal.json').unlink()
    elif mutation == 'revision':
        store.plan['sources']['experiments/c03/producer/evaluator.py'] = '0'*64
    else:
        path = store.path/('selection.json' if mutation == 'tamper' else 'stage2_rf_shared/model.joblib')
        with path.open('ab') as stream:
            stream.write(b'changed')
    monkeypatch.setattr(e, 'read_test', lambda *_: pytest.fail('Test opened before seal verification'))
    with pytest.raises((ValueError, FileNotFoundError)):
        e.evaluate(store)


def test_no_implicit_upgrade_of_old_seal(sealed, monkeypatch):
    monkeypatch.setattr(e, 'verify_seal', lambda _: {'synthetic_evaluation_contract': None})
    monkeypatch.setattr(e, 'read_test', lambda *_: pytest.fail('Old seal read test'))
    with pytest.raises(ValueError, match='prospective'):
        e.evaluate(sealed)


def test_exact_denominators_and_fixed_seven_f1():
    truth = np.arange(7, dtype=np.int8)
    pred = np.array([1, 6, 2, 3, 4, 5, 6], dtype=np.int8)
    result = e.metrics(truth, pred, np.linspace(.1, .9, 7))
    assert result['udr']['value'] == 1
    assert result['fur_known']['value'] == 1/6
    assert result['fur_known_attacks']['value'] == 1/5
    assert result['benign_family_fpr']['value'] == 1
    assert result['realized_rejection']['value'] == 2/7
    assert result['macro_f1'] == pytest.approx(f1_score(truth, pred, labels=np.arange(7), average='macro', zero_division=0))
    absent = e.metrics(np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.int8), np.zeros(3))
    assert absent['udr']['value'] is None and absent['udr']['denominator'] == 0
    assert absent['stage1_attack_auroc'] is None
    assert absent['macro_f1'] == pytest.approx(1/7)


def test_rejection_equalities_and_no_rejection_sentinel():
    p1, p2 = np.array([.5]), np.array([[.6, .1, .1, .1, .1]])
    q = p2/p2.sum(axis=1, keepdims=True)
    arm = {'stage1_threshold': .5}
    selection = {'status': 'feasible', 'method': 'max_confidence', 'threshold': 1-q[0, 0]}
    assert e.apply_policy(p1, p2, arm, selection)[0] == 6
    selection['threshold'] = '+inf'
    assert e.apply_policy(p1, p2, arm, selection)[0] == 1
    selection.update(method='family_conditional', family_thresholds=[q[0, 0]]*5)
    assert e.apply_policy(p1, p2, arm, selection)[0] == 1
    selection['family_thresholds'] = [np.nextafter(q[0, 0], 1)]*5
    assert e.apply_policy(p1, p2, arm, selection)[0] == 6


def artificial_arms():
    arms = {}
    for name in ('blind', 'visible'):
        arm = {'candidate_profile': name, 'stage1_threshold': .5, 'selections': {}}
        for budget in ('0.01', '0.03', '0.05', '0.10'):
            arm['selections'][budget] = {method: {'status': 'infeasible'}
                for method in ('max_confidence', 'margin', 'entropy', 'family_conditional')}
        arm['selections']['0.03']['max_confidence'] = {'status': 'feasible', 'method': 'max_confidence',
            'budget': .03, 'threshold': '+inf' if name == 'blind' else 0.}
        arms[name] = arm
    return arms


def test_different_profiles_alignment_sign_and_infeasibility():
    truth = np.arange(7, dtype=np.int8)
    scores = {name: (np.ones(7), np.tile([.6, .1, .1, .1, .1], (7, 1)), 'same') for name in ('blind', 'visible')}
    result = e.paired_results(truth, scores, artificial_arms(), 'same')
    assert result['0.03/max_confidence']['visible_minus_blind']['udr'] == 1
    assert result['0.03/max_confidence']['visible_minus_blind']['fur_known'] == 1
    assert result['0.01/entropy']['visible_minus_blind'] is None
    scores['visible'] = (*scores['visible'][:2], 'reordered')
    with pytest.raises(ValueError, match='row identity'):
        e.paired_results(truth, scores, artificial_arms(), 'same')


def test_changed_test_bytes_duplicate_parts_and_row_count(sealed, tmp_path):
    contract = copy.deepcopy(sealed.plan['synthetic_evaluation_contract'])
    contract['parts'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        e.read_test(contract)
    contract = copy.deepcopy(sealed.plan['synthetic_evaluation_contract'])
    contract['parts'][0]['rows'] += 1
    with pytest.raises(ValueError, match='row count'):
        e.read_test(contract)
    contract['parts'][0]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='byte drift'):
        e.read_test(contract)


def test_repeated_evaluation_refuses_overwrite(sealed, tmp_path):
    store = clone(sealed, tmp_path)
    e.evaluate(store)
    before = file_hash(store.path/'evaluation.json')
    with pytest.raises(ValueError, match='already exists'):
        e.evaluate(store)
    assert file_hash(store.path/'evaluation.json') == before
