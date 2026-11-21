"""Additive contract; never edits or upgrades the completed producer plan."""
import json
from pathlib import Path

from experiments.c03.pilot import contract as producer, runner
from experiments.c03.producer.plan import ROOT, OUTPUT_ROOT, canonical, digest, file_hash, require, output_location
from experiments.c03.producer.artifacts import safe_artifact
from experiments.c03.producer.checkpoints import publish
from experiments.c03.engineering import resource_policy, preparation_probe as host

SCOPE = 'C03_FIXED_BOTNET_SEED123_POSTSEAL'
LIMITS = {**resource_policy.LIMITS, 'seconds': 3600}
FILES = ('__init__.py', 'contract.py', 'selection.py', 'evaluation.py', 'supervisor.py')


def sources():
    paths = [Path(__file__).with_name(n) for n in FILES] + [ROOT/'tests/test_c03_postseal.py']
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def test_inventory():
    # Read the already frozen inventory, not the test files or their metadata.
    manifest = json.loads((ROOT/'.release-audit/c03/C03_input_content_manifest.json').read_text())
    parts = [{'path': (ROOT/e['relative_path']).relative_to(host.DATA_ROOT).as_posix(),
              'sha256': e['sha256']} for e in manifest['data_files'] if e['role'] == 'test']
    require(len(parts) == 65, 'Test inventory differs')
    return {'parts': sorted(parts, key=lambda x: x['path']), 'rows': 1190899,
            'minimum_per_class': 200, 'block_rows': 8192,
            'order': 'lexicographic part path, then original zero-based row offset'}


def dependencies(store):
    runner.verify_completed(store)
    paths = {'plan.json', 'run.json', 'inputs.json', 'request.json', 'prepared/complete.json', 'producer-complete.json'}
    receipt = store.read_json(store.path/'prepared/complete.json')
    paths.update(receipt['files'])
    paths.add(store.read_json(store.path/'producer-complete.json')['summary_path'])
    components = {}
    for name in producer.COMPONENTS:
        fit = runner.checkpoint(store, name, 'fit')
        score = runner.checkpoint(store, name, 'score', fit)
        components[name] = {'fit': fit.relative_to(store.path).as_posix(), 'score': score.relative_to(store.path).as_posix()}
        for folder, key in ((fit, 'model_file'), (score, 'score_file')):
            result = store.read_json(folder/'complete.json')['result']
            paths.update(p.relative_to(store.path).as_posix() for p in (folder/'complete.json', folder/result[key]))
    return {n: file_hash(safe_artifact(store.path, n)) for n in sorted(paths)}, components


def definition(store):
    hashes, components = dependencies(store)
    return {'scope': SCOPE, 'producer_run_id': store.path.name, 'producer_plan_sha256': digest(store.plan),
            'producer_dependencies': hashes, 'components': components, 'sources': sources(),
            'test': test_inventory(), 'limits': LIMITS,
            'authorization': 'User autonomous fixed-pilot continuation; 2026-11-18 proceed',
            'policy_sha256': store.plan['policy_sha256'], 'comparison': 'visible minus blind',
            'uncertainty': 'Not estimated; fixed-case conditional descriptive pairing only',
            'selection': 'Unchanged approved selector; both arms sealed before test access',
            'inference': 'One verified model per worker; unchanged float64 CSR transform and probability semantics',
            'metrics': 'Existing producer.evaluator.paired_results; fixed seven-class labels and explicit denominators'}


class Store:
    def __init__(self, path, plan, source):
        self.path, self.plan, self.source = path, plan, source

    def read(self, name):
        return json.loads(safe_artifact(self.path, name).read_text())

    def verify(self, *, artifacts=True):
        require(self.read('contract.json') == self.plan and self.plan['scope'] == SCOPE, 'Post-seal contract drift')
        require(self.plan['sources'] == sources() and self.plan['limits'] == LIMITS
                and self.plan['test'] == test_inventory(), 'Post-seal source/definition drift')
        require(producer.open_store(self.source.path).plan == self.source.plan
                and digest(self.source.plan) == self.plan['producer_plan_sha256'], 'Producer definition drift')
        if artifacts:
            for name, sha in self.plan['producer_dependencies'].items():
                require(file_hash(safe_artifact(self.source.path, name)) == sha, 'Producer dependency drift: '+name)


def open_store(path):
    path = output_location(path)
    plan = json.loads(safe_artifact(path, 'contract.json').read_text())
    source = producer.open_store(safe_artifact(OUTPUT_ROOT, plan['producer_run_id']))
    store = Store(path, plan, source)
    store.verify(artifacts=False)
    return store


def create(path, source):
    plan = definition(source)
    path = output_location(path)
    path.mkdir(exist_ok=False)
    publish(path/'contract.json', canonical(plan))
    return Store(path, plan, source)
