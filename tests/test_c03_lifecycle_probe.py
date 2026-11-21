"""Bounded supervisor, no automatic retry, and independent fit/score recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from experiments.c03.engineering import lifecycle as c, lifecycle_probe as p
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer.plan import canonical, file_hash, ROOT


def request():
    start = time.time()
    return {**c.request_base(), 'started_epoch': start, 'deadline_epoch': start+900,
            'selected_gpu': p.gpu_guard.load_readiness()['selected_gpu']}


def test_request_drift_and_deadline(tmp_path):
    r = request()
    publish(tmp_path/'request.json', canonical(r))
    assert p.verify_request(tmp_path) == r
    r = json.loads(canonical(r))
    r['definition']['n_estimators'] = 9
    (tmp_path/'request.json').write_bytes(canonical(r))
    with pytest.raises(ValueError, match='definition drift'):
        p.verify_request(tmp_path)


def test_recovery_never_resets_original_deadline(tmp_path, monkeypatch):
    r = request()
    r['started_epoch'] -= 901
    r['deadline_epoch'] -= 901
    publish(tmp_path/'request.json', canonical(r))
    monkeypatch.setattr(p, 'run_phase', lambda *a: pytest.fail('Expired work started'))
    with pytest.raises(ValueError, match='deadline exhausted'):
        p.execute(tmp_path, recover=True)


def test_budget_uses_monotonic_if_clock_moves_back(monkeypatch):
    monkeypatch.setattr(p.time, 'time', lambda: 100)
    ticks = iter([10, 15])
    monkeypatch.setattr(p.time, 'monotonic', lambda: next(ticks))
    budget = p.Budget({'started_epoch': 90})
    monkeypatch.setattr(p.time, 'time', lambda: 95)
    assert budget.elapsed() == 15


def test_live_worker_blocks_recovery(tmp_path):
    publish(tmp_path/'request.json', canonical(request()))
    with file_lock(tmp_path/('.worker-'+c.COMPONENTS[0]+'.guard')):
        with pytest.raises(FileExistsError, match='owned'):
            p.execute(tmp_path, recover=True)


def test_incomplete_attempt_requires_explicit_recovery(tmp_path):
    publish(tmp_path/'request.json', canonical(request()))
    (tmp_path/'components').mkdir()
    with pytest.raises(ValueError, match='explicit --recover'):
        p.execute(tmp_path)


@pytest.mark.parametrize('kind', ['gpu', 'telemetry'])
def test_start_refused_launches_no_worker(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(p, 'OUTPUT_ROOT', tmp_path)
    identity = p.gpu_guard.load_readiness()['selected_gpu']
    gpu = {**identity, 'free': 7*2**30, 'used': 0}
    state = {'available': 8*2**30, 'disk_free': 30*2**30}
    if kind == 'gpu':
        gpu['free'] = 5*2**30
    def query():
        if kind == 'telemetry':
            raise TimeoutError('GPU query exceeded two seconds')
        return gpu
    monkeypatch.setattr(p.host, 'resources', lambda _: state)
    monkeypatch.setattr(p.gpu_guard, 'query_gpu', query)
    monkeypatch.setattr(p.resource_policy, 'query_gpu_start', query)
    monkeypatch.setattr(p, 'execute', lambda *a, **k: pytest.fail('Worker launched'))
    result = p.probe()
    assert result['status'] == 'START_REFUSED'
    assert not list(tmp_path.iterdir())


def test_completed_result_verification_never_queries_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(p.gpu_guard, 'query_gpu', lambda: pytest.fail('Queried GPU'))
    publish(tmp_path/'request.json', canonical(request()))
    # Missing completion is a hard failure, not an implicit execution request.
    with pytest.raises(FileNotFoundError):
        p.verify_completed(tmp_path)


def test_orphan_worker_watchdog_exits_real_process(tmp_path):
    import psutil
    ready = tmp_path/'ready'
    code = ('import threading,time,sys\nfrom pathlib import Path\n'
            'from experiments.c03.engineering.lifecycle_probe import watch_owner,Budget\n'
            'Path(sys.argv[1]).write_text("ready")\n'
            'watch_owner(99999999,0,Budget({"started_epoch":time.time()}),threading.Event())\n')
    process = subprocess.Popen([sys.executable, '-B', '-c', code, str(ready)], cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = process.communicate(timeout=15)
        assert process.returncode == 15 and ready.exists()
        assert b'supervisor exited' in err
    finally:
        if process.poll() is None:
            p.host.ProcessTree(process.pid).stop()


def test_pid_reuse_does_not_count_as_supervisor():
    import psutil
    assert p.supervisor_alive(os.getpid(), psutil.Process().create_time())
    assert not p.supervisor_alive(os.getpid(), psutil.Process().create_time()+1)


def healthy_monitor(monkeypatch, r):
    gpu = {**r['selected_gpu'], 'free': 7*2**30, 'used': 256*2**20}
    monkeypatch.setattr(p.gpu_guard, 'query_gpu', lambda: gpu)
    monkeypatch.setattr(p.resource_policy, 'query_gpu_start', lambda: gpu)
    monkeypatch.setattr(p.host, 'resources', lambda _: {'available': 8*2**30, 'disk_free': 30*2**30})


def generated_worker_command(run, name, phase, attempt, fixture, mode='normal'):
    # Actual worker entrypoint, process isolation, locks, receipts and model
    # operations. Only original benchmark imports are replaced with generated
    # inputs; this helper is never reachable through the runtime CLI.
    script = '''
import sys,time
from pathlib import Path
from experiments.c03.engineering import lifecycle as c,lifecycle_probe as p
sys.path.insert(0,str(p.ROOT/'tests'))
from test_c03_lifecycle import inputs
root=Path(sys.argv[5]); root.mkdir(parents=True,exist_ok=True)
p.OUTPUT_ROOT=root.parent
c.prep.load_success=lambda: {'run_id':root.name}
def imported(source,success,plan,name,phase):
    folder=source/name;folder.mkdir(exist_ok=True)
    X,y,w,path,record=inputs(folder,name,phase)
    record['array_path']=path.relative_to(source).as_posix()
    return record
c.verify_import=imported
mode=sys.argv[6]
native=c.commit_phase
def commit(*args,**kwargs):
    native(*args,**kwargs)
    if mode=='pause_after_commit':
        (Path(sys.argv[1])/'checkpoint-ready').write_text('ready')
        time.sleep(45)
c.commit_phase=commit
if mode=='interrupt_score':
    def fail(*args,**kwargs):
        (Path(sys.argv[1])/'scoring-ready').write_text('ready')
        time.sleep(45)
    c.score_blocks=fail
p.worker(Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4])
'''
    return [sys.executable, '-B', '-c', script, str(run), name, phase, attempt, str(fixture), mode]


def test_real_isolated_fit_and_score_workers(tmp_path, monkeypatch):
    run = tmp_path/'run'
    run.mkdir()
    r = request()
    publish(run/'request.json', canonical(r))
    healthy_monitor(monkeypatch, r)
    monkeypatch.setattr(p, 'command', lambda run, name, phase, attempt:
                        generated_worker_command(run, name, phase, attempt, tmp_path/'generated'))
    name = c.COMPONENTS[0]
    budget = p.Budget(r)
    for phase in ('fit', 'score'):
        result = p.run_phase(run, name, phase, r, budget)
        assert result['status'] == 'COMPLETE', result
        assert result['worker_exit'] == 0 and result['peak_tree_rss'] > 0
    fit = p.completed_phase(run, name, 'fit')
    score = p.completed_phase(run, name, 'score', fit)
    assert fit is not None and score is not None
    assert c.read(score/'complete.json')['result']['rows'] == 280


@pytest.mark.parametrize('phase,marker', [('fit', 'checkpoint-ready'), ('score', 'scoring-ready')])
def test_killed_worker_preserves_committed_fit(tmp_path, monkeypatch, phase, marker):
    import psutil
    run = tmp_path/'run'
    run.mkdir()
    r = request()
    publish(run/'request.json', canonical(r))
    healthy_monitor(monkeypatch, r)
    name = c.COMPONENTS[0]
    fixture = tmp_path/'generated'
    monkeypatch.setattr(p, 'command', lambda run, name, phase, attempt:
                        generated_worker_command(run, name, phase, attempt, fixture))
    if phase == 'score':
        assert p.run_phase(run, name, 'fit', r, p.Budget(r))['status'] == 'COMPLETE'
    attempt = run/'components'/name/phase/'akilled'
    attempt.mkdir(parents=True)
    publish(attempt/'job.json', canonical({'request_sha256': file_hash(run/'request.json'),
        'component': name, 'phase': phase, 'owner_pid': os.getpid(), 'owner_created': psutil.Process().create_time()}))
    mode = 'pause_after_commit' if phase == 'fit' else 'interrupt_score'
    process = subprocess.Popen(generated_worker_command(run, name, phase, attempt.name, fixture, mode), cwd=ROOT,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    tree = p.host.ProcessTree(process.pid)
    try:
        deadline = time.monotonic()+15
        while not (run/marker).exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.05)
        assert (run/marker).exists(), process.communicate(timeout=1) if process.poll() is not None else 'worker not ready'
        with pytest.raises(FileExistsError):
            p.execute(run, recover=True)
        tree.stop()
        process.communicate(timeout=5)
        fit = p.completed_phase(run, name, 'fit')
        assert fit is not None
        sha = file_hash(fit/'complete.json')
        assert p.completed_phase(run, name, 'score', fit) is None
        # New score attempt reuses the saved fit and preserves the killed attempt.
        assert p.run_phase(run, name, 'score', r, p.Budget(r))['status'] == 'COMPLETE'
        assert file_hash(fit/'complete.json') == sha and attempt.exists()
        assert p.completed_phase(run, name, 'score', fit) is not None
    finally:
        tree.stop()
        if process.poll() is None:
            process.communicate(timeout=5)


def test_recovery_scheduler_skips_completed_fit_and_never_auto_retries(tmp_path, monkeypatch):
    run = tmp_path/'run'
    run.mkdir()
    r = request()
    publish(run/'request.json', canonical(r))
    name = c.COMPONENTS[0]
    fit = run/'saved-fit'
    fit.mkdir()
    publish(fit/'complete.json', b'fixture-fit-receipt')
    calls = []
    monkeypatch.setattr(p, 'completed_phase', lambda run, n, phase, fit=None: fit_path if n == name and phase == 'fit' else None)
    fit_path = fit
    monkeypatch.setattr(p, 'validate_current_import', lambda *a: None)
    def fail(run, n, phase, request, budget):
        calls.append((n, phase))
        return {'status': 'STOPPED', 'reason': 'injected scoring stop'}
    monkeypatch.setattr(p, 'run_phase', fail)
    result = p.execute(run, recover=True)
    assert result['status'] == 'STOPPED'
    assert result['reused_phases'] == [name+':fit']
    assert calls == [(name, 'score')] and not (run/'complete.json').exists()


@pytest.mark.parametrize('trigger', ['rss', 'disk', 'artifacts', 'gpu_used', 'gpu_free', 'deadline', 'query'])
def test_supervisor_stops_worker_on_each_limit(tmp_path, monkeypatch, trigger):
    run = tmp_path/'run'
    run.mkdir()
    r = request()
    publish(run/'request.json', canonical(r))
    healthy_monitor(monkeypatch, r)
    native_resources = p.host.resources
    native_gpu = p.gpu_guard.query_gpu
    class Process:
        pid = 1234
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout): self.returncode = 15; return 15
    process = Process()
    launched, stopped = [], []
    def launch(*a, **k):
        launched.append(True)
        return process
    class Tree:
        def __init__(self, pid): pass
        def rss(self): return 5*2**30 if trigger == 'rss' else 1000
        def live(self): return []
        def stop(self): stopped.append(True); process.returncode = 15
    def resources(path):
        state = native_resources(path).copy()
        if launched:
            if trigger == 'disk': state['disk_free'] = 9*2**30
        return state
    def gpu():
        state = native_gpu().copy()
        if launched:
            if trigger == 'query': raise TimeoutError('GPU timeout')
            if trigger == 'gpu_used': state['used'] = 5*2**30
            if trigger == 'gpu_free': state['free'] = 1*2**30
        return state
    monkeypatch.setattr(p.subprocess, 'Popen', launch)
    monkeypatch.setattr(p.host, 'ProcessTree', Tree)
    monkeypatch.setattr(p.host, 'resources', resources)
    monkeypatch.setattr(p.gpu_guard, 'query_gpu', gpu)
    monkeypatch.setattr(p.host, 'artifact_size', lambda _: 6*2**30 if launched and trigger == 'artifacts' else 0)
    monkeypatch.setitem(p.LIMITS, 'gpu_interval', 0)
    class Budget:
        def elapsed(self): return 901 if launched and trigger == 'deadline' else 0
    result = p.run_phase(run, c.COMPONENTS[0], 'fit', r, Budget())
    assert result['status'] == 'STOPPED' and result['reason']
    assert launched == [True] and stopped
    assert not list((run/'components').glob('**/complete.json'))


def test_artifact_accounting_keeps_failed_and_pending_attempts(tmp_path):
    sizes = [4096, 2048, 1024]
    for folder, filename, size in zip(['afailed', 'acomplete', 'anew'], ['partial.bin', 'model.joblib', '.pending-model'], sizes):
        directory = tmp_path/'components'/c.COMPONENTS[0]/'fit'/folder
        directory.mkdir(parents=True)
        (directory/filename).write_bytes(b'x'*size)
    assert p.host.artifact_size(tmp_path) == sum(sizes)


def test_available_ram_is_observed_but_never_gates_execution():
    from experiments.c03.engineering import resource_policy as policy
    state = {'available': 0, 'disk_free': 30*2**30}
    gpu = {'free': 7*2**30, 'used': 0}
    assert policy.LIMITS['start_available'] is None
    assert policy.LIMITS['min_available'] is None
    assert policy.start_reason(state, gpu) is None
    assert policy.stop_reason(state, gpu, 1*2**30, 0, 1) is None
    assert policy.stop_reason(state, gpu, 5*2**30, 0, 1) == 'Worker-tree RSS above ceiling'


def test_driver_startup_allowance_does_not_relax_live_query_timeout(monkeypatch):
    from types import SimpleNamespace
    policy = p.resource_policy
    observed = []
    def run(command, **kwargs):
        observed.append(kwargs['timeout'])
        return SimpleNamespace(stdout='0, GPU, 8192, 0, 7900, driver')
    monkeypatch.setattr(policy.subprocess, 'run', run)
    assert policy.query_gpu_start()['index'] == 0
    assert p.gpu_guard.query_gpu()['index'] == 0
    assert observed == [10., 2.]


def test_low_available_ram_reaches_probe_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'OUTPUT_ROOT', tmp_path)
    r = request()
    healthy_monitor(monkeypatch, r)
    monkeypatch.setattr(p.host, 'resources', lambda _: {'available': 1*2**20, 'disk_free': 30*2**30})
    reached = []
    def execute(run, **kwargs):
        reached.append(run)
        assert p.verify_request(run)['limits']['min_available'] is None
        return {'status': 'GENERATED_TEST_ENTRYPOINT_ONLY'}
    monkeypatch.setattr(p, 'execute', execute)
    assert p.probe()['status'] == 'GENERATED_TEST_ENTRYPOINT_ONLY'
    assert len(reached) == 1


def test_completed_resume_is_read_only_and_missing_phase_rejected(tmp_path, monkeypatch):
    r = request()
    publish(tmp_path/'request.json', canonical(r))
    locations, checkpoints = {}, {}
    for name in c.COMPONENTS:
        checkpoints[name] = {}
        for phase in ('fit', 'score'):
            directory = tmp_path/name/phase
            directory.mkdir(parents=True)
            publish(directory/'complete.json', canonical({'fixture': name+phase}))
            locations[name, phase] = directory
            checkpoints[name][phase] = file_hash(directory/'complete.json')
    summary = {'status': 'COMPLETE', 'scope': c.SCOPE, 'wall_seconds': 100}
    publish(tmp_path/'summary.json', canonical(summary))
    publish(tmp_path/'complete.json', canonical({'scope': c.SCOPE, 'scientific_checkpoint': False,
        'request_sha256': file_hash(tmp_path/'request.json'), 'checkpoints': checkpoints,
        'summary_path': 'summary.json', 'summary_sha256': file_hash(tmp_path/'summary.json')}))
    monkeypatch.setattr(p, 'completed_phase', lambda run, name, phase, fit=None: locations.get((name, phase)))
    monkeypatch.setattr(p.gpu_guard, 'query_gpu', lambda: pytest.fail('GPU query during completed resume'))
    monkeypatch.setattr(p, 'run_phase', lambda *a: pytest.fail('New work during completed resume'))
    assert p.verify_completed(tmp_path)['reused_completed_result']
    assert p.execute(tmp_path, recover=True)['status'] == 'COMPLETE'
    locations.pop((c.COMPONENTS[-1], 'score'))
    with pytest.raises(ValueError, match='Checkpoint receipt drift'):
        p.verify_completed(tmp_path)
