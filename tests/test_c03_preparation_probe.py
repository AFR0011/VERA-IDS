"""Disposable fixtures and worker processes only; no benchmark reads or fits."""
import json
import io
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import psutil
import pytest

from experiments.c03.engineering import preparation_probe as p
from experiments.c03.producer.plan import file_hash
from experiments.c03.producer.validation import historical, truth_labels
from experiments.c03.producer.artifacts import tensor_hash, identity_hash, array_hash
from test_c03_producer_validation import fixture_sources


def ample():
    return {'available': 8*p.GIB, 'disk_free': 30*p.GIB}


def test_progress_reports_stage_changes_and_thirty_second_heartbeat(tmp_path):
    stream = io.StringIO()
    progress = p.Progress(tmp_path, stream)
    progress.update(0, p.GIB, ample(), 0)
    progress.update(20, p.GIB, ample(), 0)
    assert len(stream.getvalue().splitlines()) == 1
    progress.update(30, 2*p.GIB, ample(), p.GIB)
    (tmp_path/'stages.jsonl').write_text('{"stage":"sample_train"}\n')
    progress.update(31, 2*p.GIB, ample(), p.GIB)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert '[00:30 / 15:00 budget]' in lines[1]
    assert 'worker 2.00/4.00 GiB' in lines[1] and 'available 8.00 GiB' in lines[1]
    assert 'sample train' in lines[2] and '%' not in stream.getvalue()


def test_progress_handles_incomplete_stage_write_and_keeps_stdout_clean(tmp_path, capsys):
    (tmp_path/'stages.jsonl').write_text('{"stage":"sample_train"}\n{"stage":')
    progress = p.Progress(tmp_path)
    progress.update(31, p.GIB, ample(), 0)
    progress.finish({'wall_seconds': 32, 'status': 'STOPPED', 'peak_tree_rss': p.GIB,
                     'reason': 'fixture stop'})
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'sample train' in captured.err and 'STOPPED' in captured.err and 'fixture stop' in captured.err


@pytest.mark.parametrize('cap', [700, 1750, 5000])
def test_train_sampler_equals_verified_reader(tmp_path, cap):
    sources = fixture_sources(tmp_path/'sources')
    train = p.TrainSources(sources.root, sources.inventory()['train'])
    expected = sources.sampler().sample('train', cap, 123)
    frame, ids = train.sample(cap, 123)
    pd.testing.assert_frame_equal(frame, expected.frame)
    pd.testing.assert_frame_equal(ids, expected.identities)


@pytest.mark.parametrize('numeric,categorical', [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize('block_rows', [1, 7, 8192])
def test_bounded_transform_byte_identical_to_full_historical(numeric, categorical, block_rows):
    _, _, engine = historical()
    frame = pd.DataFrame({'Feature One': np.resize([0., 1., np.nan, np.inf, -np.inf, -3.5], 37),
        'feature_two': np.zeros(37), 'category': np.resize(['a', 'b', 'unknown', None], 37),
        'y_stage1_attack': np.ones(37, dtype=int), 'y_stage2_family': ['DoS']*37})
    original = frame.copy(deep=True)
    prep = engine.Preprocessor(num_cols=['feature_one', 'feature_two'] if numeric else [],
        cat_cols=['category'] if categorical else [],
        num_mean=np.array([.2, 0.]) if numeric else np.array([]),
        num_std=np.array([2., 1.]) if numeric else np.array([]),
        cat_maps={'category': {'a': 0, 'b': 1, engine.CFG['cat_na_token']: 2}} if categorical else {},
        cat_offsets={'category': 0} if categorical else {}, n_num=2 if numeric else 0,
        n_cat=3 if categorical else 0)
    mask = np.arange(37) % 3 != 1
    full = prep.transform(frame.loc[mask].reset_index(drop=True).drop(columns=['y_stage1_attack', 'y_stage2_family']))
    calls = []
    native = prep.transform
    def watched(part):
        calls.append(len(part))
        return native(part)
    prep.transform = watched
    actual = p.transform_selected(prep, frame, mask, block_rows=block_rows)
    assert actual.shape == full.shape
    for name in ('data', 'indices', 'indptr'):
        left, right = getattr(actual, name), getattr(full, name)
        assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
    assert tensor_hash(actual) == tensor_hash(full)
    assert max(calls) <= block_rows
    assert len(calls) == 2*((int(mask.sum())+block_rows-1)//block_rows)
    pd.testing.assert_frame_equal(frame, original)


def test_real_preparation_train_only_and_no_classifiers(tmp_path, monkeypatch):
    sources = fixture_sources(tmp_path/'sources')
    train = p.TrainSources(sources.root, sources.inventory()['train'])
    core, grid, engine = historical()
    def forbidden(*args, **kwargs):
        pytest.fail('Classifier constructor called')
    monkeypatch.setattr(grid, 'build_stage1_model', forbidden)
    monkeypatch.setattr(grid, 'build_stage2_model', forbidden)
    original = pd.read_parquet
    opened = []
    def checked(path, **kwargs):
        assert Path(path).parent == sources.root/'train'
        opened.append(Path(path))
        return original(path, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', checked)
    run = tmp_path/'run'
    run.mkdir()
    stages = []
    result = p.prepare_training(train, run, cap=1750, expected_source_rows=1750,
        expected_counts={'stage1': 1525, 'stage2': 1125}, emit=stages.append)
    assert result['status'] == 'PREPARATION_COMPLETE'
    assert result['counts'] == {'stage1': 1525, 'stage2': 1125}
    assert result['tensors']['stage2']['shape'] == [1125, 2]
    assert result['tensors']['stage2']['dtype'] == 'float64'
    assert stages[-1] == 'preparation_complete' and opened
    assert not (run/'complete.json').exists()  # Only supervisor may mark completion.
    assert not list(run.glob('*.npz'))  # No reusable scientific tensor checkpoint.
    # Differential reference: original full-selected-frame transform, including
    # the exact byte hashes subsequently bound to scientific artifacts.
    import joblib
    prep = joblib.load(run/'preprocessor/preprocessor.joblib')
    frame, identities = train.sample(1750, 123)
    truth = truth_labels(frame)
    assert result['sample_identity_sha256'] == identity_hash(identities)
    for stage, mask in {'stage1': truth != 6, 'stage2': (truth > 0) & (truth < 6)}.items():
        reference = prep.transform(frame.loc[mask].reset_index(drop=True).drop(
            columns=['y_stage1_attack', 'y_stage2_family']))
        target = (truth[mask] != 0).astype(int) if stage == 'stage1' else (truth[mask]-1).astype(int)
        assert result['tensors'][stage]['tensor_sha256'] == tensor_hash(reference)
        assert result['tensors'][stage]['labels_sha256'] == array_hash(target)
        assert result['tensors'][stage]['row_ids_sha256'] == identity_hash(identities.loc[mask].reset_index(drop=True))


def test_source_rejection_before_preprocessor(tmp_path, monkeypatch):
    sources = fixture_sources(tmp_path/'sources')
    entries = sources.inventory()['train']
    with pytest.raises(ValueError, match='Only TRAIN'):
        p.TrainSources(sources.root, sources.inventory()['val'])
    with pytest.raises(ValueError, match='Duplicate'):
        p.TrainSources(sources.root, entries+entries)
    train = p.TrainSources(sources.root, entries)
    (sources.root/entries[0]['path']).write_bytes(b'drift')
    _, _, engine = historical()
    monkeypatch.setattr(engine, 'fit_preprocessor', lambda *_: pytest.fail('Fit before source verification'))
    with pytest.raises(ValueError, match='byte drift'):
        p.prepare_training(train, tmp_path, cap=1750, expected_source_rows=1750,
                           expected_counts={}, emit=lambda _: None)


def test_wrong_counts_and_extra_source_fail_closed(tmp_path):
    sources = fixture_sources(tmp_path/'sources')
    train = p.TrainSources(sources.root, sources.inventory()['train'])
    with pytest.raises(ValueError, match='source row count'):
        p.prepare_training(train, tmp_path, cap=1750, expected_source_rows=1751,
                           expected_counts={}, emit=lambda _: None)
    assert not (tmp_path/'preprocessor').exists()
    with pytest.raises(ValueError, match='supervised TRAIN counts'):
        p.prepare_training(train, tmp_path, cap=1750, expected_source_rows=1750,
                           expected_counts={'stage1': 1750, 'stage2': 1750}, emit=lambda _: None)
    assert not (tmp_path/'worker-result.json').exists()
    (sources.root/'train/unexpected.csv').write_text('feature\n1\n')
    with pytest.raises(ValueError, match='inventory differs'):
        train.verify(historical()[2])


@pytest.mark.parametrize('state', [
    {'available': 5*p.GIB, 'disk_free': 30*p.GIB},
    {'available': 8*p.GIB, 'disk_free': 19*p.GIB}])
def test_start_refusal_never_launches(tmp_path, monkeypatch, state):
    monkeypatch.setattr(p, 'resources', lambda _: state)
    monkeypatch.setattr(p.subprocess, 'Popen', lambda *_a, **_k: pytest.fail('Worker launched'))
    result = p.supervise(tmp_path, [])
    assert result['status'] == 'START_REFUSED'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('overrides,fragment', [
    ({'state': {'available': p.GIB, 'disk_free': 30*p.GIB}}, 'RAM below'),
    ({'rss': 4*p.GIB+1}, 'RSS above'),
    ({'state': {'available': 8*p.GIB, 'disk_free': 9*p.GIB}}, 'space below'),
    ({'size': 5*p.GIB+1}, 'artifacts above'),
    ({'elapsed': 900}, 'budget reached')])
def test_every_stop_threshold(overrides, fragment):
    args = {'state': ample(), 'rss': 0, 'size': 0, 'elapsed': 0}
    args.update(overrides)
    assert fragment in p.stop_reason(**args)
    assert p.stop_reason(ample(), 4*p.GIB, 5*p.GIB, 899) is None


def command(run, code):
    script = run/'fixture_worker.py'
    script.write_text(code, encoding='utf-8')
    return [sys.executable, '-B', str(script), str(run)]


def test_supervisor_success_and_missing_result(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'resources', lambda _: ample())
    run = tmp_path/'success'
    run.mkdir()
    cmd = command(run, "import json,sys\nfrom pathlib import Path\nr=Path(sys.argv[1])\n"
        "(r/'worker-result.json').write_text(json.dumps({'scope':'ENGINEERING_TRAIN_PREPARATION_ONLY',"
        "'status':'PREPARATION_COMPLETE','files':{}}))\n")
    result = p.supervise(run, cmd)
    assert result['status'] == 'COMPLETE'
    receipt = json.loads((run/'complete.json').read_text())
    assert receipt['worker_result_sha256'] == file_hash(run/'worker-result.json')
    bad = tmp_path/'incomplete'
    bad.mkdir()
    result = p.supervise(bad, command(bad, 'pass\n'))
    assert result['status'] == 'STOPPED'
    assert not (bad/'complete.json').exists()


def test_memory_stop_reaps_actual_descendants(tmp_path, monkeypatch):
    child_code = "import os,sys,time;from pathlib import Path;Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(45)"
    script = ("import subprocess,sys,time,os\nfrom pathlib import Path\nr=Path(sys.argv[1])\n"
        "(r/'parent.pid').write_text(str(os.getpid()))\n"
        f"subprocess.Popen([sys.executable,'-B','-c',{child_code!r},str(r/'child.pid')])\n"
        "time.sleep(45)\n")
    def state(_):
        return {'available': p.GIB if (tmp_path/'child.pid').exists() else 8*p.GIB,
                'disk_free': 30*p.GIB}
    monkeypatch.setattr(p, 'resources', state)
    # Prevent a broken fixture from waiting for a 15-minute production limit.
    monkeypatch.setitem(p.LIMITS, 'seconds', 10)
    result = p.supervise(tmp_path, command(tmp_path, script))
    assert result['status'] == 'STOPPED' and 'RAM below' in result['reason']
    for name in ('parent.pid', 'child.pid'):
        assert not psutil.pid_exists(int((tmp_path/name).read_text()))
    assert not (tmp_path/'complete.json').exists()


def test_monitor_exception_fails_closed_and_reaps_worker(tmp_path, monkeypatch):
    calls = 0
    def state(_):
        nonlocal calls
        calls += 1
        if (tmp_path/'worker.pid').exists():
            raise psutil.AccessDenied()
        return ample()
    monkeypatch.setattr(p, 'resources', state)
    monkeypatch.setitem(p.LIMITS, 'seconds', 10)
    script = "import os,sys,time\nfrom pathlib import Path\n(Path(sys.argv[1])/'worker.pid').write_text(str(os.getpid()))\ntime.sleep(45)\n"
    result = p.supervise(tmp_path, command(tmp_path, script))
    assert result['status'] == 'STOPPED' and 'AccessDenied' in result['reason']
    assert not psutil.pid_exists(int((tmp_path/'worker.pid').read_text()))
    assert not (tmp_path/'complete.json').exists()


def test_interruption_cleans_up_without_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'resources', lambda _: ample())
    monkeypatch.setattr(p.time, 'sleep', lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    result = p.supervise(tmp_path, command(tmp_path, 'import time\ntime.sleep(45)\n'))
    assert result['status'] == 'STOPPED' and 'KeyboardInterrupt' in result['reason']
    assert not (tmp_path/'complete.json').exists()
