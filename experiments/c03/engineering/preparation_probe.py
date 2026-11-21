"""Approved bounded TRAIN-only preparation probe; never fits classifiers.

Run from the repository root: python -B -m experiments.c03.engineering.preparation_probe --probe
No budget overrides, retry, scientific checkpoint reuse or validation/test reader.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

import psutil

from experiments.c03.producer.plan import (
    ROOT, OUTPUT_ROOT, build_plan, canonical, digest, file_hash, require, safe_relative)
from experiments.c03.producer.checkpoints import ENGINEERING_SCOPE, publish
from experiments.c03.producer.transforms import BLOCK_ROWS as TRANSFORM_ROWS, transform_selected

GIB = 1024**3
LIMITS = {'start_available': 6*GIB, 'start_disk': 20*GIB,
          'min_available': 2*GIB, 'max_rss': 4*GIB,
          'min_disk': 10*GIB, 'max_artifacts': 5*GIB,
          'seconds': 900, 'interval': .3}
SCOPE = 'ENGINEERING_TRAIN_PREPARATION_ONLY'
RF_SCOPE = 'ENGINEERING_RF_STAGE2_SIZING_ONLY'
DATA_ROOT = ROOT/'processed_V5/B_day_file/CICIoT2023'


def resources(path):
    return {'available': psutil.virtual_memory().available,
            'disk_free': shutil.disk_usage(path).free}


def start_reason(state):
    if state['available'] < LIMITS['start_available']:
        return 'Less than 6 GiB available RAM'
    if state['disk_free'] < LIMITS['start_disk']:
        return 'Less than 20 GiB free output-volume space'


def stop_reason(state, rss, size, elapsed):
    if state['available'] < LIMITS['min_available']:
        return 'Available RAM below 2 GiB'
    if rss > LIMITS['max_rss']:
        return 'Worker-tree RSS above 4 GiB'
    if state['disk_free'] < LIMITS['min_disk']:
        return 'Free output-volume space below 10 GiB'
    if size > LIMITS['max_artifacts']:
        return 'Probe artifacts above 5 GiB'
    if elapsed >= LIMITS['seconds']:
        return '900-second probe budget reached'


class ProcessTree:
    """Retain process identities so children are reaped even if the launcher exits."""
    def __init__(self, pid):
        self.members = {}
        self.add(psutil.Process(pid))

    def add(self, process):
        self.members[(process.pid, process.create_time())] = process

    def live(self):
        for process in list(self.members.values()):
            if process.is_running():
                try:
                    for child in process.children(recursive=True):
                        self.add(child)
                except psutil.NoSuchProcess:
                    pass
        return [p for p in self.members.values() if p.is_running()]

    def rss(self):
        total = 0
        for process in self.live():
            try:
                total += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return total

    def stop(self):
        live = self.live()
        for process in reversed(live):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(live, timeout=5)
        require(not alive, 'Worker descendants did not stop')


def artifact_size(run):
    # Publication briefly exposes a temporary name and a hard link to the same
    # file; count its bytes once and tolerate removal of the temporary name.
    seen, total = set(), 0
    for path in run.rglob('*'):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        identity = (info.st_dev, info.st_ino)
        if stat.S_ISREG(info.st_mode) and identity not in seen:
            seen.add(identity)
            total += info.st_size
    return total


class Progress:
    """Observed stages/resources on stderr; stdout stays available for final JSON."""
    def __init__(self, run, stream=None):
        self.run = run
        self.stream = sys.stderr if stream is None else stream
        self.stage = 'starting_worker'
        self.last_stage = None
        self.last_time = float('-inf')

    @staticmethod
    def clock(seconds):
        minutes, seconds = divmod(int(seconds), 60)
        return f'{minutes:02d}:{seconds:02d}'

    def update(self, elapsed, rss, state, size):
        try:
            raw = (self.run/'stages.jsonl').read_text(encoding='utf-8')
            # A worker may be in the middle of writing its next JSON line.
            complete = raw[:raw.rfind('\n')+1].splitlines()
            if complete:
                candidate = json.loads(complete[-1]).get('stage')
                if isinstance(candidate, str):
                    self.stage = candidate
        except (OSError, ValueError):
            pass  # Display failure never substitutes for the resource monitor.
        if self.stage != self.last_stage or elapsed-self.last_time >= 30:
            print(f'[{self.clock(elapsed)} / {self.clock(LIMITS["seconds"])} budget] '
                  f'{self.stage.replace("_", " ")} | '
                  f'worker {rss/GIB:.2f}/{LIMITS["max_rss"]/GIB:.2f} GiB | '
                  f'available {state["available"]/GIB:.2f} GiB | '
                  f'artifacts {size/GIB:.2f}/{LIMITS["max_artifacts"]/GIB:.2f} GiB',
                  file=self.stream, flush=True)
            self.last_stage, self.last_time = self.stage, elapsed

    def finish(self, summary):
        print(f'[{self.clock(summary["wall_seconds"])}] {summary["status"]} | '
              f'peak worker {summary["peak_tree_rss"]/GIB:.2f} GiB'
              + (f' | {summary["reason"]}' if summary['reason'] else ''),
              file=self.stream, flush=True)


def supervise(run, command, *, scope=SCOPE):
    """Run one fresh worker. Only a successful, verified result gets a final receipt."""
    require(scope in (SCOPE, ENGINEERING_SCOPE, RF_SCOPE), 'Unapproved supervisor scope')
    initial = resources(run)
    reason = start_reason(initial)
    if reason:
        return {'scope': scope, 'status': 'START_REFUSED', 'reason': reason, **initial}
    started = time.monotonic()
    progress = Progress(run)
    print(f'Probe run: {run}', file=sys.stderr, flush=True)
    peak, minimum, max_bytes = 0, initial['available'], 0
    process = tree = None
    error = None
    try:
        with (run/'worker.log').open('x', encoding='utf-8') as log, \
             (run/'monitor.jsonl').open('x', encoding='utf-8') as trace:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            tree = ProcessTree(process.pid)
            while True:
                state = resources(run)
                rss = tree.rss()
                size = artifact_size(run)
                elapsed = time.monotonic()-started
                peak, minimum, max_bytes = max(peak, rss), min(minimum, state['available']), max(max_bytes, size)
                trace.write(json.dumps({'seconds': elapsed, 'rss': rss, 'artifact_bytes': size, **state})+'\n')
                trace.flush()
                progress.update(elapsed, rss, state, size)
                reason = stop_reason(state, rss, size, elapsed)
                if reason:
                    break
                if process.poll() is not None:
                    require(not tree.live(), 'Launcher exited with live descendants')
                    require(process.returncode == 0, f'Worker exit {process.returncode}')
                    result = json.loads((run/'worker-result.json').read_text(encoding='utf-8'))
                    expected_status = 'RF_SIZING_COMPLETE' if scope == RF_SCOPE else 'PREPARATION_COMPLETE'
                    require(result['scope'] == scope and result['status'] == expected_status,
                            'Missing or invalid worker result')
                    for name, sha in result['files'].items():
                        from experiments.c03.producer.artifacts import safe_artifact
                        require(file_hash(safe_artifact(run, name)) == sha, 'Probe artifact drift')
                        reason = stop_reason(resources(run), 0, artifact_size(run), time.monotonic()-started)
                        if reason:
                            break
                    break
                time.sleep(LIMITS['interval'])
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        if tree is not None:
            try:
                tree.stop()
            except Exception as exc:
                error = f'Cleanup failed: {type(exc).__name__}: {exc}; earlier={error}'
        elif process is not None and process.poll() is None:
            # Monitoring initialization failed: still attempt full descendant cleanup.
            try:
                ProcessTree(process.pid).stop()
            except Exception as exc:
                error = f'Cleanup failed: {type(exc).__name__}: {exc}; earlier={error}'
                process.kill()
        if process is not None:
            process.wait(timeout=10)
    summary = {'scope': scope, 'status': 'STOPPED' if reason or error else 'COMPLETE',
               'reason': reason or error, 'peak_tree_rss': peak, 'minimum_available': minimum,
               'max_observed_artifact_bytes': max_bytes, 'wall_seconds': time.monotonic()-started,
               'worker_exit': None if process is None else process.returncode, 'limits': LIMITS}
    publish(run/'supervisor.json', canonical(summary))
    if summary['status'] == 'COMPLETE':
        publish(run/'complete.json', canonical({
            'scope': scope, 'scientific_checkpoint': False,
            'request_sha256': file_hash(run/'request.json') if (run/'request.json').exists() else None,
            'worker_result_sha256': file_hash(run/'worker-result.json'),
            'supervisor_sha256': file_hash(run/'supervisor.json')}))
    progress.finish(summary)
    return summary


def source_record():
    plan = build_plan()  # Metadata/code only; no benchmark observations.
    paths = [Path(__file__), ROOT/'tests/test_c03_preparation_probe.py']
    return {'plan_sha256': digest(plan),
            'probe_sources': {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}}


def training_inventory():
    manifest = json.loads((ROOT/'.release-audit/c03/C03_input_content_manifest.json').read_text())
    entries = []
    for entry in manifest['data_files']:
        if entry['role'] == 'train':
            path = ROOT/safe_relative(entry['relative_path'])
            entries.append({'path': path.relative_to(DATA_ROOT).as_posix(), 'sha256': entry['sha256']})
    require(len(entries) == 239, 'Expected 239 TRAIN parts')
    return sorted(entries, key=lambda x: x['path'])


class TrainSources:
    def __init__(self, root, entries):
        self.root = Path(root).resolve()
        self.entries = entries
        names = [x['path'] for x in entries]
        require(names and names == sorted(set(names)), 'Duplicate/unordered TRAIN inventory')
        for name in names:
            relative = safe_relative(name)
            require(len(relative.parts) == 2 and relative.parts[0] == 'train' and
                    relative.suffix == '.parquet', 'Only TRAIN Parquet sources allowed')

    def path(self, entry):
        from experiments.c03.producer.artifacts import safe_artifact
        require(entry in self.entries, 'Unregistered TRAIN source')
        return safe_artifact(self.root, entry['path'])

    def verify(self, engine):
        paths = [self.path(x) for x in self.entries]
        require([Path(p).resolve() for p in engine.list_parts(str(self.root), 'train')] == paths,
                'Historical TRAIN inventory differs')
        for entry, path in zip(self.entries, paths):
            require(file_hash(path) == entry['sha256'], 'TRAIN byte drift')

    def sample(self, cap, seed):
        # Exact legacy sequential buffer draws, with IDs outside feature columns.
        import numpy as np
        import pandas as pd
        rng = np.random.RandomState(seed)
        current = ids = columns = None
        for entry in self.entries:
            path = self.path(entry)
            require(file_hash(path) == entry['sha256'], 'TRAIN byte drift')
            frame = pd.read_parquet(path)
            require(file_hash(path) == entry['sha256'], 'TRAIN changed during read')
            frame.columns = [str(c).strip().lower().replace(' ', '_') for c in frame.columns]
            require(frame.columns.is_unique, 'Canonical columns collide')
            if columns is None:
                columns = list(frame.columns)
            require(list(frame.columns) == columns, 'TRAIN schema/order drift')
            part_ids = pd.DataFrame({'source_part': [entry['path']]*len(frame),
                'source_sha256': [entry['sha256']]*len(frame), 'row_offset': np.arange(len(frame), dtype=np.int64)})
            mixed = frame if current is None else pd.concat([current, frame], ignore_index=True)
            mixed_ids = part_ids if ids is None else pd.concat([ids, part_ids], ignore_index=True)
            if len(mixed) > cap:
                positions = pd.Series(np.arange(len(mixed))).sample(
                    n=cap, random_state=rng.randint(0, 2**31-1)).to_numpy()
                current = mixed.iloc[positions].reset_index(drop=True)
                ids = mixed_ids.iloc[positions].reset_index(drop=True)
            else:
                current = mixed.copy() if current is None else mixed.reset_index(drop=True)
                ids = mixed_ids.reset_index(drop=True)
        require(current is not None and len(current) > 0, 'Empty TRAIN sample')
        return current, ids




def prepare_training(sources, run, *, cap, expected_source_rows, expected_counts, emit):
    """Actual historical preparation; injected tiny sources are used only by tests."""
    import numpy as np
    import pyarrow.parquet as pq
    from experiments.c03.producer.validation import historical, truth_labels
    from experiments.c03.producer.artifacts import tensor_hash, identity_hash, array_hash
    from experiments.c03.producer.models import json_parameters
    _, grid, engine = historical()
    timings = {}
    start = time.monotonic()
    emit('verify_train')
    sources.verify(engine)
    count = sum(pq.ParquetFile(sources.path(x)).metadata.num_rows for x in sources.entries)
    require(count == expected_source_rows, 'TRAIN source row count differs')
    timings['verify_train'] = time.monotonic()-start
    emit('preprocessor')
    start = time.monotonic()
    prep_dir = run/'preprocessor'
    prep_dir.mkdir()
    prep = engine.fit_preprocessor(str(sources.root), str(prep_dir))
    timings['preprocessor'] = time.monotonic()-start
    emit('sample_train')
    start = time.monotonic()
    frame, identities = sources.sample(cap, 123)
    require(len(frame) == cap, 'TRAIN sample cap not reached')
    truth = truth_labels(frame)
    masks = {'stage1': truth != 6, 'stage2': (truth > 0) & (truth < 6)}
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    require(counts == expected_counts, 'Strict supervised TRAIN counts differ')
    require(all(int((truth == label).sum()) >= 200 for label in range(6)), 'Insufficient TRAIN support')
    timings['sample_train'] = time.monotonic()-start
    tensors, targets, ids, weights, records = {}, {}, {}, {}, {}
    for stage, mask in masks.items():
        emit(stage+'_tensor')
        start = time.monotonic()
        tensors[stage] = transform_selected(prep, frame, mask)
        emit(stage+'_metadata')
        targets[stage] = (truth[mask] != 0).astype(int) if stage == 'stage1' else (truth[mask]-1).astype(int)
        ids[stage] = identities.loc[mask].reset_index(drop=True)
        if stage == 'stage1':
            for mode in ('class_weight_balanced', 'inv_family_clipped'):
                weights[mode] = grid.make_stage1_sample_weights(targets[stage],
                    frame.loc[mask, 'y_stage2_family'].astype(str).to_numpy(dtype=object), mode,
                    float(grid.CFG['inv_family_clip_max']))
        else:
            weights[stage] = grid.stage2_sample_weights(targets[stage], 5, 'balanced')
        tensor = tensors[stage]
        records[stage] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'csr_bytes': tensor.data.nbytes+tensor.indices.nbytes+tensor.indptr.nbytes,
            'tensor_sha256': tensor_hash(tensor), 'row_ids_sha256': identity_hash(ids[stage]),
            'labels_sha256': array_hash(targets[stage])}
        timings[stage+'_tensor'] = time.monotonic()-start
    emit('final_verification')
    sources.verify(engine)
    result = {'scope': SCOPE, 'status': 'PREPARATION_COMPLETE', 'scientific_checkpoint': False,
        'source_rows': count, 'sample_rows': len(frame), 'sample_seed': 123,
        'transform_execution': {'rows_per_block': TRANSFORM_ROWS, 'passes': 2,
                                'historical_transform_unchanged': True},
        'sample_identity_sha256': identity_hash(identities), 'support': np.bincount(truth, minlength=7).tolist(),
        'counts': counts, 'tensors': records, 'timings_seconds': timings,
        'source_inventory_sha256': digest(sources.entries), 'helper_config': json_parameters(engine.CFG),
        'weight_sha256': {k: None if v is None else array_hash(v) for k, v in weights.items()},
        'files': {p.relative_to(run).as_posix(): file_hash(p) for p in prep_dir.iterdir() if p.is_file()}}
    # Both tensors/labels/weights/IDs remain live through this result; no model fits.
    publish(run/'worker-result.json', canonical(result))
    emit('preparation_complete')
    return result


def worker(run):
    request = json.loads((run/'request.json').read_text())
    require(request['scope'] == SCOPE and request['limits'] == LIMITS, 'Unapproved probe scope')
    require(request['source_record'] == source_record(), 'Probe source/environment drift')
    require(request['inventory'] == training_inventory(), 'TRAIN inventory drift')
    with (run/'stages.jsonl').open('x', encoding='utf-8') as stream:
        started = time.monotonic()
        def emit(stage):
            stream.write(json.dumps({'stage': stage, 'seconds': time.monotonic()-started})+'\n')
            stream.flush()
        prepare_training(TrainSources(DATA_ROOT, request['inventory']), run,
                         cap=1500000, expected_source_rows=3676338,
                         expected_counts={'stage1': 1270025, 'stage2': 1201422}, emit=emit)
        require(request['source_record'] == source_record(), 'Probe source changed during preparation')


def probe():
    record, inventory = source_record(), training_inventory()
    # Inspect existing parent volume before creating a run or launching a worker.
    parent = OUTPUT_ROOT
    while not parent.exists():
        parent = parent.parent
    state = resources(parent)
    reason = start_reason(state)
    if reason:
        return {'scope': SCOPE, 'status': 'START_REFUSED', 'reason': reason, **state}
    run = OUTPUT_ROOT/('prep-readiness-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    require(not run.resolve().is_relative_to(ROOT), 'Probe output redirected into repository')
    run.mkdir(parents=True, exist_ok=False)
    publish(run/'request.json', canonical({'scope': SCOPE, 'limits': LIMITS,
                                         'source_record': record, 'inventory': inventory}))
    result = supervise(run, [sys.executable, '-B', '-u', '-m',
                            'experiments.c03.engineering.preparation_probe', '--_worker', str(run)])
    return {**result, 'run_directory': str(run)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe', action='store_true')
    group.add_argument('--inspect', action='store_true')
    group.add_argument('--_worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        worker(args._worker)
        return 0
    result = resources(ROOT) if args.inspect else probe()
    print(json.dumps(result, indent=2))
    return 0 if args.inspect or result['status'] == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
