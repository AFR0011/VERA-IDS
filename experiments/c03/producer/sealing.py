"""Persistent validation-only selection seal. Never resolves/opens test sources."""
import copy
import json

from .artifacts import safe_artifact
from .checkpoints import publish
from .plan import ROOT, canonical, digest, file_hash, require
from .validation import historical, verify_prepared, component_record, validation_bundle


def current_code(plan):
    for name, sha in plan['sources'].items():
        require(file_hash(safe_artifact(ROOT, name)) == sha, 'Producer/source revision drift')


def dependencies(store):
    receipt = verify_prepared(store)
    paths = ['plan.json', 'run.json', 'inputs.json', 'prepared/complete.json', *receipt['files']]
    for name in store.plan['graph']['components']:
        component = component_record(store, name)
        paths += [f'{name}/started.json', f'{name}/complete.json', f'{name}/payload.bin', *component['files']]
    return {name: file_hash(safe_artifact(store.path, name)) for name in sorted(set(paths))}


def select_and_seal(store):
    require(store.plan.get('evidence_scope') == 'SYNTHETIC_ONLY', 'Full experiment execution is not authorized')
    with store.lock():
        current_code(store.plan)
        before = dependencies(store)
        truth, scores = validation_bundle(store)
        core, grid, engine = historical()
        counts = store.plan['expected_support']
        decisions = core.select_both_arms(truth, scores, grid=grid, engine=engine,
                                        expected_counts=(counts['validation_K'], counts['validation_U']))
        frozen = core.FrozenSelection()
        frozen.seal(decisions)
        ledger, ledger_digest = frozen.export()
        require(before == dependencies(store), 'Inputs changed during selection')
        publication = {'schema_version': 1, 'scope': 'SYNTHETIC_ONLY',
                       'plan_sha256': digest(store.plan), 'decision_sha256': ledger_digest,
                       'validation_provenance': verify_prepared(store)['validation'],
                       'source_inventory_sha256': digest(verify_prepared(store)['source_inventory']),
                       'original_snapshot_role': 'reference-policy identity; not synthetic-data identity',
                       'dependencies': before, 'decisions': ledger,
                       'evaluation_authorized': False,
                       'synthetic_evaluation_contract': store.plan.get('synthetic_evaluation_contract'),
                       'evaluator_revision': (store.plan.get('synthetic_evaluation_contract') or {}).get('evaluator_revision')}
        # Both arms in one atomic object; publication of the seal is last.
        publish(store.path/'selection.json', core.strict_json(publication))
        publish(store.path/'selection-seal.json', canonical({
            'schema_version': 1, 'plan_sha256': digest(store.plan),
            'selection_sha256': file_hash(store.path/'selection.json'),
            'state': 'BOTH_ARMS_SEALED_SYNTHETIC_ONLY'}))
    return verify_seal(store)


def verify_seal(store):
    """Receipt verification only; does not grant test access or deserialize models."""
    with store.lock():
        current_code(store.plan)
        store.verify_plan()
        seal = store.read_json(store.path/'selection-seal.json')
        selection_path = safe_artifact(store.path, 'selection.json')
        require(seal == {'schema_version': 1, 'plan_sha256': digest(store.plan),
                         'selection_sha256': file_hash(selection_path),
                         'state': 'BOTH_ARMS_SEALED_SYNTHETIC_ONLY'}, 'Selection seal drift')
        return _verified_selection(store)


def _verified_selection(store):
    selection_path = safe_artifact(store.path, "selection.json")
    selection = store.read_json(selection_path)
    require(selection['scope'] == 'SYNTHETIC_ONLY' and selection['evaluation_authorized'] is False and
            selection['evaluator_revision'] == (store.plan.get('synthetic_evaluation_contract') or {}).get('evaluator_revision') and
            selection.get('synthetic_evaluation_contract') == store.plan.get('synthetic_evaluation_contract') and
            selection['plan_sha256'] == digest(store.plan),
            'Unexpected selection scope or plan')
    require(selection['dependencies'] == dependencies(store), 'Sealed dependency drift')
    receipt = verify_prepared(store)
    require(selection['validation_provenance'] == receipt['validation'] and
            selection['source_inventory_sha256'] == digest(receipt['source_inventory']),
            'Selection provenance drift')
    core, _, _ = historical()
    ledger = selection['decisions']
    require(core.sha_record(ledger) == selection['decision_sha256'], 'Decision digest drift')
    require(ledger['status'] == 'SEALED_VALIDATION_SELECTION', 'Wrong decision state')
    original = copy.deepcopy(ledger)
    original['status'] = 'VALIDATION_ONLY_DECISIONS_NOT_YET_SEALED'
    check = core.FrozenSelection()
    require(check.seal(original) == selection['decision_sha256'], 'Incomplete arm selection')
    return selection


def finish_seal(store):
    """Explicit recovery of only the final receipt; never rerun selection."""
    require(store.plan.get('evidence_scope') == 'SYNTHETIC_ONLY', 'Full experiment execution is not authorized')
    with store.lock():
        current_code(store.plan)
        store.verify_plan()
        _verified_selection(store)
        path = store.path/"selection-seal.json"
        if not path.exists():
            publish(path, canonical({"schema_version": 1, "plan_sha256": digest(store.plan),
                "selection_sha256": file_hash(store.path/"selection.json"),
                "state": "BOTH_ARMS_SEALED_SYNTHETIC_ONLY"}))
    return verify_seal(store)
