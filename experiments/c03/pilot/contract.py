"""Fixed Botnet/seed-123 scientific producer, authorized 2026-11-17.

This is a new scope and fresh preparation, not promotion of engineering models.
It produces fitted components and validation scores only. A later selection seal
must bind its own evaluator revision before any test observations are decoded.
"""
import copy
from pathlib import Path

from experiments.c03.producer.plan import ROOT, build_plan, canonical, digest, file_hash, require, output_location
from experiments.c03.producer.checkpoints import Checkpoints, publish, token
from experiments.c03.engineering import lifecycle, resource_policy

SCOPE = 'C03_FIXED_BOTNET_SEED123_PRODUCER'
COMPONENTS = lifecycle.COMPONENTS
LIMITS = {**resource_policy.LIMITS, 'seconds': 36000, 'max_artifacts': 15*1024**3,
          'start_disk': 30*1024**3}


def frozen_plan():
    base = build_plan()
    files = [Path(__file__), Path(__file__).with_name('preparation.py'),
             Path(__file__).with_name('models.py'), Path(__file__).with_name('runner.py'),
             ROOT/'tests/test_c03_pilot.py']
    plan = copy.deepcopy(base)
    plan.update(status='FROZEN_FIXED_CASE_PRODUCER', evidence_scope=SCOPE, execution_authorized=True,
        authorization={'date': '2026-11-17', 'basis': 'User autonomous continuation of fixed C03 plan',
                       'available_ram_gates_removed': True, 'bounded_jobs_may_outlive_agent_session': True},
        reference_plan_sha256=digest(base), producer_sources={p.relative_to(ROOT).as_posix(): file_hash(p) for p in files},
        lifecycle_sources=lifecycle.source_record(), resource_limits=LIMITS,
        preparation='Fresh historical TRAIN-stream fit; exact existing sampler and float64 CSR transforms',
        selection_authorized=False, test_reader_authorized=False,
        claim={'id': 'C03.fixed.Botnet.seed123', 'type': 'magnitude',
               'changed_factor': 'heldout-family validation availability to supervised selection',
               'arms': ['blind_K', 'visible_same_K_plus_U'], 'contrast': 'visible minus blind',
               'interpretation': 'Single prepared-corpus fixed-case descriptive pairing; no population causal claim',
               'uncertainty': 'Deferred; producer emits no performance claims or inferential results'})
    for node in plan['graph']['components'].values():
        node['parameters']['n_jobs'] = 2
        if node['model_family'] == 'xgb':
            node['parameters']['device'] = 'cuda:0'
    require([plan['graph']['components'][name]['parameters']['n_estimators'] for name in COMPONENTS]
            == [300, 300, 800, 300, 1400], 'Historical ensemble counts differ')
    return plan


class Store(Checkpoints):
    @staticmethod
    def scope(plan):
        return SCOPE

    @classmethod
    def create(cls, root, run_id, plan):
        require(plan == frozen_plan(), 'Only the frozen fixed-case plan can create a scientific run')
        path = output_location(root)/token(run_id)
        path.mkdir(parents=True, exist_ok=False)
        publish(path/'plan.json', canonical(plan))
        publish(path/'run.json', canonical({'schema_version': 1, 'run_id': path.name,
            'plan_sha256': digest(plan), 'scope': SCOPE}))
        return cls(path, plan)


def open_store(path):
    from experiments.c03.engineering.lifecycle import read
    path = output_location(path)
    plan = read(path/'plan.json')
    require(plan == frozen_plan(), 'Scientific producer source/environment/definition drift')
    store = Store(path, plan)
    store.verify_plan()
    return store
