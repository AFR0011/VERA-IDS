"""Shared DOC readiness bindings and owner/deadline supervision, no data loading."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import psutil

from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.plan import canonical, file_hash, require, ROOT

SCOPE = 'ENGINEERING_DOC_FULL_ROW_READINESS_ONLY'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def request(run):
    path = Path(run) / 'request.json'
    require(not path.is_symlink(), 'Redirected request')
    value = read(path)
    require(value['scope'] == SCOPE and value['deadline'] == value['started'] + 900, 'Scope/deadline drift')
    for name, expected in value['sources'].items():
        require(file_hash(ROOT / name) == expected, 'Readiness source drift: ' + name)
    return value


def guard(plan):
    require(plan['started'] <= time.time() < plan['deadline'], 'Original shared deadline exhausted or clock moved')


def supervise_owner(run, attempt):
    plan = request(run)
    owner = read(Path(run) / 'attempts' / attempt / 'owner.json')
    done = threading.Event()
    monotonic_deadline = time.monotonic() + max(0, plan['deadline'] - time.time())
    def watch():
        while not done.wait(.3):
            try:
                alive = psutil.Process(owner['pid']).create_time() == owner['created']
            except psutil.Error:
                alive = False
            if not alive or time.time() >= plan['deadline'] or time.monotonic() >= monotonic_deadline:
                os._exit(15)
    threading.Thread(target=watch, daemon=True).start()
    return plan, done


def emit(run, attempt, stage, **details):
    with (Path(run) / 'attempts' / attempt / 'progress.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'stage': stage, **details}) + '\n')


def bound_json(run, name, value):
    path = Path(run) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = {'request_sha256': file_hash(Path(run) / 'request.json'), **value}
    if path.exists():
        require(not path.is_symlink() and read(path) == expected, 'Checkpoint identity drift: ' + name)
    else:
        publish(path, canonical(expected))
    return expected


def verify_files(root, record):
    for name, expected in record['files'].items():
        path = Path(root) / name
        require(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve())
                and file_hash(path) == expected, 'Prepared file drift: ' + name)


def verify_journal(run, expected_commit):
    folder=Path(run)/'fit'; plan_sha=file_hash(folder/'plan.json'); previous=None
    for path in sorted(folder.glob('commit-*.json')):
        require(not path.is_symlink(), 'Redirected journal commit')
        entry=read(path)
        require(entry['plan_sha256']==plan_sha and entry['previous']==previous, 'Journal chain drift')
        payload=path.with_name(path.name.replace('commit-','payload-').replace('.json','.pt'))
        require(not payload.is_symlink() and file_hash(payload)==entry['payload_sha256'], 'Journal payload drift')
        previous=file_hash(path)
    require(previous is not None and previous==expected_commit, 'Final journal binding drift')
