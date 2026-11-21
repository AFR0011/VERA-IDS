"""Validation-only selection, with atomic both-arm publication and seal recovery."""
import copy
import numpy as np

from . import contract
from experiments.c03.producer import validation as v
from experiments.c03.producer.artifacts import safe_artifact, array_hash
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.plan import canonical, digest, file_hash, require


def bundle(store):
    core, _, _ = v.historical()
    prep = store.source.read_json(store.source.path/'prepared/complete.json')
    with np.load(store.source.path/'prepared/validation.npz', allow_pickle=False) as arrays:
        truth = arrays['target']
    require(array_hash(truth) == prep['validation']['truth'], 'Validation truth drift')
    cached = {}
    for name, paths in store.plan['components'].items():
        folder = safe_artifact(store.source.path, paths['score'])
        result = store.source.read_json(folder/'complete.json')['result']
        with np.load(folder/'validation.npz', allow_pickle=False) as arrays:
            cached[name] = arrays['probabilities']
        require(array_hash(cached[name]) == result['probabilities_sha256'], 'Validation score content drift')
    scores = {name: core.CandidateScores(cached[refs['stage1']], cached[refs['stage2']])
              for name, refs in store.source.plan['graph']['candidate_references'].items()}
    return truth, scores


def verify_publication(store):
    store.verify()
    record = store.read('selection.json')
    require(record['scope'] == contract.SCOPE and record['contract_sha256'] == digest(store.plan)
            and record['test_accessed'] is False, 'Selection binding drift')
    prep = store.source.read_json(store.source.path/'prepared/complete.json')
    require(record['validation'] == prep['validation'], 'Validation population drift')
    core, _, _ = v.historical()
    original = copy.deepcopy(record['decisions'])
    require(original['status'] == 'SEALED_VALIDATION_SELECTION', 'Wrong decision state')
    original['status'] = 'VALIDATION_ONLY_DECISIONS_NOT_YET_SEALED'
    require(core.FrozenSelection().seal(original) == record['decision_sha256'], 'Decision ledger drift')
    return record


def verify_seal(store):
    require(store.read('selection-seal.json') == {'scope': contract.SCOPE,
        'contract_sha256': digest(store.plan), 'selection_sha256': file_hash(store.path/'selection.json'),
        'state': 'BOTH_ARMS_SEALED_BEFORE_TEST'}, 'Selection seal drift')
    return verify_publication(store)


def select(store, emit):
    store.verify()
    if (store.path/'selection-seal.json').exists():
        return verify_seal(store)
    if not (store.path/'selection.json').exists():
        emit('loading_verified_validation_scores')
        truth, scores = bundle(store)
        core, grid, engine = v.historical()
        counts = store.source.plan['expected_support']
        emit('selecting_both_arms_four_methods_four_budgets')
        decision = core.select_both_arms(truth, scores, grid=grid, engine=engine,
            expected_counts=(counts['validation_K'], counts['validation_U']))
        frozen = core.FrozenSelection()
        frozen.seal(decision)
        ledger, sha = frozen.export()
        store.verify()
        record = {'scope': contract.SCOPE, 'contract_sha256': digest(store.plan),
            'decisions': ledger, 'decision_sha256': sha, 'test_accessed': False,
            'validation': store.source.read_json(store.source.path/'prepared/complete.json')['validation']}
        publish(store.path/'selection.json', core.strict_json(record))
    verify_publication(store)  # Recovery only completes a verified existing publication.
    publish(store.path/'selection-seal.json', canonical({'scope': contract.SCOPE,
        'contract_sha256': digest(store.plan), 'selection_sha256': file_hash(store.path/'selection.json'),
        'state': 'BOTH_ARMS_SEALED_BEFORE_TEST'}))
    emit('both_arms_sealed')
    return verify_seal(store)


def selected_components(store, record):
    names = set()
    for arm in record['decisions']['arms'].values():
        if any(s['status'] == 'feasible' for b in arm['selections'].values() for s in b.values()):
            names.update(store.source.plan['graph']['candidate_references'][arm['candidate_profile']].values())
    return sorted(names)
