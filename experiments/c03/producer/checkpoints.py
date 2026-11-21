"""Single-writer, immutable local checkpoint contracts. No model deserialization.

Accepts no-fit, synthetic validation, and engineering preparation namespaces.
The preparation layer derives tensor/input identities. This storage layer checks
their consistency and never grants classifier-fitting or evaluation authority.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile

from .plan import canonical, digest, file_hash, require, valid_hash, output_location
from .locking import file_lock

ENGINEERING_SCOPE = 'ENGINEERING_TRAIN_VALIDATION_PREPARATION_ONLY'

def publish(path: Path, payload: bytes):
    """Flush a same-volume temporary file; atomically link without overwriting.

    Requires hard-link support (tested on local NTFS). Failure leaves the run
    incomplete. No power-loss durability guarantee for directory metadata.
    """
    fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Fails if destination already exists.
    finally:
        Path(temporary).unlink(missing_ok=True)


def token(value):
    require(isinstance(value, str) and re.fullmatch('[a-z][a-z0-9_-]{0,95}', value),
            'Invalid run/component identifier')
    return value


class Checkpoints:
    def __init__(self, path: Path, plan):
        self.path = path
        self.plan = json.loads(canonical(plan))  # Caller mutation cannot change identity.

    @classmethod
    def create(cls, output_root, run_id, plan):
        require(plan.get('execution_authorized') is False and (
            plan.get('status') == 'NO_FIT_PLAN' or
            (plan.get('status') == 'SYNTHETIC_VALIDATION_PLAN' and plan.get('evidence_scope') == 'SYNTHETIC_ONLY') or
            (plan.get('status') == 'ENGINEERING_PREPARATION_PLAN' and plan.get('evidence_scope') == ENGINEERING_SCOPE)),
            'Only no-fit, explicit synthetic or engineering preparation plans are accepted')
        root = output_location(output_root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / token(run_id)
        path.mkdir(exist_ok=False)
        publish(path/'plan.json', canonical(plan))
        publish(path/'run.json', canonical({'schema_version': 1, 'run_id': run_id, 'plan_sha256': digest(plan),
                                          'scope': cls.scope(plan)}))
        return cls(path, plan)

    @staticmethod
    def scope(plan):
        if plan.get('status') == 'ENGINEERING_PREPARATION_PLAN':
            return ENGINEERING_SCOPE
        return ('SYNTHETIC_VALIDATION_ONLY' if plan.get('status') == 'SYNTHETIC_VALIDATION_PLAN'
                else 'NO_FIT_FOUNDATION_ONLY')

    @classmethod
    def resume(cls, output_root, run_id, plan):
        path = output_location(output_root) / token(run_id)
        require(path.resolve().parent == output_location(output_root), 'Run path redirected')
        store = cls(path, plan)
        with store.lock():
            store.verify_plan()
            if (path/'inputs.json').exists():
                store.inputs()
            for component in sorted(plan['graph']['components']):
                if (path/component).exists():
                    store.verify_component(component)
        return store

    @contextmanager
    def lock(self):
        require(self.path.is_dir() and not self.path.is_symlink(), 'Invalid run directory')
        if (self.path/'.writer-lock').exists():
            raise FileExistsError('Legacy writer lock requires review; it is not stolen')
        with file_lock(self.path/'.writer.guard'):
            yield

    @contextmanager
    def pipeline_lock(self):
        if (self.path/'.pipeline-lock').exists():
            raise FileExistsError('Legacy pipeline lock requires review; it is not stolen')
        with file_lock(self.path/'.pipeline.guard'):
            yield

    def component_lock(self, name):
        return file_lock(self.path/('.component-'+token(name)+'.guard'))

    def read_json(self, path):
        require(not path.is_symlink() and path.resolve().is_relative_to(self.path.resolve()),
                'Artifact redirected outside run')
        return json.loads(path.read_text(encoding='utf-8'))

    def verify_plan(self):
        stored = self.read_json(self.path/'plan.json')
        record = self.read_json(self.path/'run.json')
        require(stored == self.plan and record == {
            'schema_version': 1, 'run_id': self.path.name, 'plan_sha256': digest(self.plan),
            'scope': self.scope(self.plan)}, 'Plan/source/environment drift')

    def bind_inputs(self, bindings):
        self.validate_inputs(bindings)
        with self.lock():
            self.verify_plan()
            publish(self.path/'inputs.json', canonical({
                'plan_sha256': digest(self.plan), 'bindings': bindings,
                'bindings_sha256': digest(bindings)}))

    def validate_inputs(self, bindings):
        require(set(bindings) == {'preprocessor', 'stage1', 'stage2'}, 'Incomplete input bindings')
        require(valid_hash(bindings['preprocessor']), 'Invalid preprocessor hash')
        for stage in ('stage1', 'stage2'):
            item = bindings[stage]
            require(set(item) == {'rows', 'row_ids', 'tensor', 'labels', 'weights'},
                    'Incomplete tensor/row/label/weight binding')
            require(type(item['rows']) is int and item['rows'] > 0, 'Invalid row count')
            require(item['rows'] == self.plan['expected_support'][f'{stage}_train'],
                    'Sampled training support differs from plan')
            for name in ('row_ids', 'tensor', 'labels'):
                require(valid_hash(item[name]), 'Invalid input digest')
            components = {k for k, v in self.plan['graph']['components'].items() if v['stage'] == stage}
            require(set(item['weights']) == components, 'Weight bindings must cover each component')
            require(all(valid_hash(v) for v in item['weights'].values()), 'Invalid weight digest')
            # For a model-native weighting mode, hash a canonical explicit no-array
            # marker; never conflate absent metadata with intentionally absent weights.

    def inputs(self):
        record = self.read_json(self.path/'inputs.json')
        self.validate_inputs(record['bindings'])
        require(record['plan_sha256'] == digest(self.plan) and
                record['bindings_sha256'] == digest(record['bindings']), 'Input binding drift')
        return record['bindings']

    def expected(self, component):
        token(component)
        require(component in self.plan['graph']['components'], 'Unknown component')
        config = self.plan['graph']['components'][component]
        inputs = self.inputs()
        stage = inputs[config['stage']]
        return {'component': component, 'plan_sha256': digest(self.plan), 'config': config,
                'preprocessor': inputs['preprocessor'],
                'inputs': {k: v for k, v in stage.items() if k != 'weights'},
                'weights': stage['weights'][component]}

    def component_path(self, component):
        path = self.path/token(component)
        require(not path.is_symlink() and path.resolve().parent == self.path.resolve(),
                'Component path redirected')
        return path

    def begin(self, component):
        with self.lock():
            self.verify_plan()
            expected = self.expected(component)
            directory = self.component_path(component)
            directory.mkdir(exist_ok=False)
            publish(directory/'started.json', canonical(expected))

    def complete(self, component, payload: bytes):
        require(isinstance(payload, bytes) and payload, 'Nonempty opaque bytes required')
        with self.lock():
            self.verify_plan()
            expected = self.expected(component)
            directory = self.component_path(component)
            require(self.read_json(directory/'started.json') == expected, 'Component binding drift')
            publish(directory/'payload.bin', payload)
            publish(directory/'complete.json', canonical({
                'binding_sha256': digest(expected), 'payload_sha256': file_hash(directory/'payload.bin'),
                'payload_bytes': len(payload), 'state': 'COMPLETE'}))

    def verify_component(self, component):
        self.verify_plan()
        expected = self.expected(component)
        directory = self.component_path(component)
        require(self.read_json(directory/'started.json') == expected, 'Component binding drift')
        record = self.read_json(directory/'complete.json')
        payload = directory/'payload.bin'
        require(not payload.is_symlink(), 'Payload redirected')
        require(record == {'binding_sha256': digest(expected), 'payload_sha256': file_hash(payload),
                           'payload_bytes': payload.stat().st_size, 'state': 'COMPLETE'},
                'Incomplete/corrupt checkpoint')
        return payload  # Caller may inspect bytes; this module never deserializes.
