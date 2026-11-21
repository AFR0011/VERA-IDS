"""Checkpoint-boundary recovery and actual isolated workers on tiny fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from experiments.c03.producer import validation as v, isolated, sealing
from experiments.c03.producer.checkpoints import Checkpoints, ENGINEERING_SCOPE
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer.recovery import fork_verified, open_store
from experiments.c03.producer.plan import ROOT, file_hash
from experiments.c03.engineering import preparation_probe as guard, combined_preparation_probe as combined
from test_c03_producer_validation import fixture_plan, fixture_sources


def prepared(tmp_path, engineering=False):
    plan = fixture_plan()
    if engineering:
        plan.update(status='ENGINEERING_PREPARATION_PLAN', evidence_scope=ENGINEERING_SCOPE)
    sources = fixture_sources(tmp_path/'sources')
    store = Checkpoints.create(tmp_path/'runs', 'original', plan)
    data = (v.prepare_engineering(store, sources, lambda _: None) if engineering else v.prepare(store, sources))
    return store, sources, data


def test_component_loader_reads_only_its_training_stage_and_validation(tmp_path, monkeypatch):
    store, _, full = prepared(tmp_path)
    name = store.plan['graph']['fit_order'][0]
    stage = store.plan['graph']['components'][name]['stage']
    native = v.read_arrays
    reads = []
    def read(path):
        reads.append(path.name)
        return native(path)
    monkeypatch.setattr(v, 'read_arrays', read)
    subset = v.load_component_prepared(store, name)
    assert reads == [stage+'.npz', 'validation.npz']
    assert set(subset.tensors) == {stage} and set(subset.weights) == {name}
    assert v.tensor_hash(subset.tensors[stage]) == v.tensor_hash(full.tensors[stage])
    np.testing.assert_array_equal(subset.targets[stage], full.targets[stage])
    subset.tensors[stage].data[0] += 1
    with pytest.raises(ValueError, match='identity drift'):
        v.fit_component(store, name, subset)


def test_fork_reuses_complete_components_and_preserves_failed_attempt(tmp_path):
    source, _, data = prepared(tmp_path)
    first, second = source.plan['graph']['fit_order'][:2]
    v.fit_component(source, first, data)
    source.begin(second)
    partial = source.component_path(second)/'partial.bin'
    partial.write_bytes(b'failed attempt evidence')
    before = file_hash(source.component_path(first)/'model.joblib')
    target = fork_verified(source, tmp_path/'runs', 'retry')
    assert partial.read_bytes() == b'failed attempt evidence'
    assert not target.component_path(second).exists()
    assert file_hash(target.component_path(first)/'model.joblib') == before
    assert v.component_record(target, first) == v.component_record(source, first)
    assert open_store(target.path).plan == source.plan
    (target.path/'fork-complete.json').unlink()
    with pytest.raises(ValueError, match='Interrupted recovery'):
        open_store(target.path)


def test_corrupt_complete_component_is_not_treated_as_interrupted(tmp_path):
    source, _, data = prepared(tmp_path)
    name = source.plan['graph']['fit_order'][0]
    v.fit_component(source, name, data)
    with (source.component_path(name)/'model.joblib').open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum drift'):
        fork_verified(source, tmp_path/'runs', 'retry')
    assert not (tmp_path/'runs/retry').exists()


def test_live_component_owner_blocks_recovery(tmp_path):
    source, _, _ = prepared(tmp_path)
    name = source.plan['graph']['fit_order'][0]
    with source.component_lock(name):
        with pytest.raises(FileExistsError, match='owned'):
            fork_verified(source, tmp_path/'runs', 'retry')
    assert not (tmp_path/'runs/retry').exists()


def test_os_lock_released_when_owner_process_is_killed(tmp_path):
    lock, ready = tmp_path/'owner.guard', tmp_path/'ready'
    code = ('import sys,time\nfrom pathlib import Path\n'
        'from experiments.c03.producer.locking import file_lock\n'
        'with file_lock(Path(sys.argv[1])):\n Path(sys.argv[2]).write_text("ready")\n time.sleep(45)\n')
    process = subprocess.Popen([sys.executable, '-B', '-c', code, str(lock), str(ready)], cwd=ROOT)
    tree = guard.ProcessTree(process.pid)
    try:
        deadline = time.monotonic()+10
        while not ready.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert ready.exists()
        with pytest.raises(FileExistsError):
            with file_lock(lock):
                pytest.fail('Live owner lock acquired')
        tree.stop()
        process.wait(timeout=5)
        with file_lock(lock):
            pass  # Persistent filename is not stale ownership.
    finally:
        tree.stop()
        process.wait(timeout=5)


def test_killed_component_worker_preserves_prefix_and_can_restart_in_fork(tmp_path):
    store, _, data = prepared(tmp_path)
    first, second = store.plan['graph']['fit_order'][:2]
    v.fit_component(store, first, data)
    attempt = store.path/'attempts/fault-injection'
    attempt.mkdir(parents=True)
    ready = store.path/'fit-paused'
    code = ('import sys,time\nfrom pathlib import Path\n'
        'from experiments.c03.producer import models,isolated\n'
        'class Paused:\n'
        ' def get_params(self): return {}\n'
        ' def fit(self,*a,**k):\n  Path(sys.argv[4]).write_text("ready")\n  time.sleep(45)\n'
        'models.construct=lambda *_: Paused()\n'
        'isolated.component_worker(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]))\n')
    process = subprocess.Popen([sys.executable, '-B', '-c', code, str(store.path), second, str(attempt), str(ready)], cwd=ROOT)
    tree = guard.ProcessTree(process.pid)
    try:
        deadline = time.monotonic()+15
        while not ready.exists() and time.monotonic()<deadline:
            time.sleep(.05)
        assert ready.exists()
        with pytest.raises(FileExistsError):
            fork_verified(store, tmp_path/'runs', 'busy')
        tree.stop()
        process.wait(timeout=5)
        assert (store.component_path(second)/'started.json').exists()
        assert not (store.component_path(second)/'complete.json').exists()
        retry = fork_verified(store, tmp_path/'runs', 'retry')
        assert v.component_record(retry, first) == v.component_record(store, first)
        assert v.fit_component(retry, second, v.load_component_prepared(retry, second))
        v.component_record(retry, second)
        assert not (store.component_path(second)/'complete.json').exists()
    finally:
        tree.stop()
        process.wait(timeout=5)


def test_actual_isolated_workers_and_resume_without_refitting(tmp_path, monkeypatch):
    store, _, data = prepared(tmp_path)
    del data
    monkeypatch.setattr(guard, 'resources', lambda _: {'available': 8*guard.GIB, 'disk_free': 30*guard.GIB})
    result = isolated.resume(store)
    assert result['scope'] == 'SYNTHETIC_ONLY'
    assert len(list((store.path/'attempts').glob('*/result.json'))) == 5
    for name in store.plan['graph']['components']:
        v.component_record(store, name)
    monkeypatch.setattr(isolated, 'execute_component', lambda *_: pytest.fail('Refitted completed component'))
    assert isolated.resume(store) == result
    # Receipt-only recovery must not rerun selection or modify frozen decisions.
    digest = file_hash(store.path/'selection.json')
    (store.path/'selection-seal.json').unlink()
    monkeypatch.setattr(sealing, 'select_and_seal', lambda *_: pytest.fail('Repeated selection'))
    assert isolated.resume(store) == result
    assert file_hash(store.path/'selection.json') == digest


def test_incomplete_component_requires_explicit_fork(tmp_path, monkeypatch):
    store, _, _ = prepared(tmp_path)
    store.begin(store.plan['graph']['fit_order'][0])
    monkeypatch.setattr(isolated, 'execute_component', lambda *_: pytest.fail('Implicit retry'))
    with pytest.raises(ValueError, match='--fork'):
        isolated.resume(store)


def test_engineering_checkpoint_cannot_fit_or_be_promoted_on_fork(tmp_path):
    store, _, _ = prepared(tmp_path, engineering=True)
    target = fork_verified(store, tmp_path/'runs', 'retry')
    assert target.plan['evidence_scope'] == ENGINEERING_SCOPE
    with pytest.raises(ValueError, match='Only synthetic'):
        isolated.resume(target)


def test_combined_resume_skips_preparation(tmp_path, monkeypatch):
    store, sources, data = prepared(tmp_path, engineering=True)
    del data
    target = fork_verified(store, tmp_path/'runs', 'reload')
    monkeypatch.setattr(v, 'prepare_engineering', lambda *_: pytest.fail('Repeated preparation'))
    stages = []
    result = combined.prepare_and_reload(target, sources, stages.append, resume=True)
    assert result['reload_verified']
    assert stages == ['prepared_checkpoint_reused', 'reload', 'reload_verified']


def test_preparation_receipt_is_last_commit_marker(tmp_path, monkeypatch):
    sources = fixture_sources(tmp_path/'sources')
    store = Checkpoints.create(tmp_path/'runs', 'fixture', fixture_plan())
    native = v.publish
    def interrupt(path, payload):
        if path == store.path/'prepared/complete.json':
            assert (store.path/'inputs.json').exists()
            raise OSError('receipt interrupted')
        return native(path, payload)
    monkeypatch.setattr(v, 'publish', interrupt)
    with pytest.raises(OSError, match='interrupted'):
        v.prepare(store, sources)
    with pytest.raises(FileNotFoundError):
        v.verify_prepared(store)
    assert not (store.path/'prepared/complete.json').exists()
