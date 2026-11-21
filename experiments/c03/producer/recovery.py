"""Preserve failed runs; fork only verified preparation and completed components."""
from contextlib import ExitStack
import json
from pathlib import Path
import shutil

from .checkpoints import Checkpoints, publish
from .plan import ROOT, build_plan, canonical, digest, file_hash, require, output_location


def open_store(path):
    from .sealing import current_code
    path = output_location(path)
    plan = json.loads((path/'plan.json').read_text(encoding='utf-8'))
    store = Checkpoints(path, plan)
    store.verify_plan()
    current_code(plan)
    current = build_plan()
    require(plan['environment'] == current['environment'] and
            plan['policy_sha256'] == current['policy_sha256'], 'Checkpoint environment/policy drift')
    require(not (path/'fork-intent.json').exists() or (path/'fork-complete.json').exists(),
            'Interrupted recovery fork; choose a fresh run ID')
    if (path/'fork-intent.json').exists():
        require(store.read_json(path/'fork-complete.json') ==
                {'intent_sha256': file_hash(path/'fork-intent.json')}, 'Recovery receipt drift')
    return store


def fork_verified(source, output_root, run_id):
    """Explicit recovery: no deletion, moving, overwriting or promotion of scope."""
    from .sealing import current_code
    from .validation import verify_prepared, component_record
    current_code(source.plan)
    with source.pipeline_lock(), source.lock(), ExitStack() as locks:
        for name in source.plan['graph']['fit_order']:
            locks.enter_context(source.component_lock(name))
        require(not (source.path/'selection.json').exists(), 'Frozen selection exists; do not fork its decisions')
        receipt = verify_prepared(source)
        files = {'inputs.json', 'prepared/complete.json', *receipt['files']}
        reused, incomplete = [], []
        for name in source.plan['graph']['fit_order']:
            directory = source.component_path(name)
            if not directory.exists():
                continue
            if not (directory/'complete.json').exists():
                incomplete.append(name)
                continue
            record = component_record(source, name)  # Corrupt COMPLETE is not an interrupted fit.
            files.update({f'{name}/started.json', f'{name}/complete.json', f'{name}/payload.bin', *record['files']})
            reused.append(name)
        hashes = {name: file_hash(source.path/name) for name in sorted(files)}
        target = Checkpoints.create(output_root, run_id, source.plan)
        with target.pipeline_lock():
            intent = {'parent_run': str(source.path.resolve()), 'plan_sha256': digest(source.plan),
                      'reused_components': reused, 'incomplete_components_not_copied': incomplete, 'files': hashes}
            publish(target.path/'fork-intent.json', canonical(intent))
            for name, sha in hashes.items():
                path = target.path/name
                require(path.resolve().is_relative_to(target.path.resolve()), 'Recovery destination escapes run')
                path.parent.mkdir(parents=True, exist_ok=True)
                with (source.path/name).open('rb') as src, path.open('xb') as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
                    dst.flush()
                    import os
                    os.fsync(dst.fileno())
                require(file_hash(path) == sha == file_hash(source.path/name), 'Recovery byte drift')
            verify_prepared(target)
            for name in reused:
                component_record(target, name)
            publish(target.path/'fork-complete.json', canonical({'intent_sha256': file_hash(target.path/'fork-intent.json')}))
        return target
