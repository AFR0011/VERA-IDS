"""Bindings and deadlines for the approved fixed-case DOC scientific pilot."""
import json
import os
from pathlib import Path
import threading
import time

import psutil

from .doc_runtime import read, bound_json, verify_files, emit
from experiments.c03.producer.plan import ROOT, canonical, file_hash, require, safe_relative
from experiments.c03.producer.checkpoints import publish

SCOPE = 'DOC_PAPER_RULE_TABULAR_BOTNET_SEED123_PILOT'
PREPARED = ROOT.parent / 'VERA-IDS-local-runs/literature/doc-readiness-20261119t200421626706z'
OUTPUT = PREPARED.parent
PYTHON = ROOT / '.release-audit/doc-env/Scripts/python.exe'


def safe_artifact(root, name):
    relative = safe_relative(name)
    root = Path(root)
    path = root.joinpath(*relative.parts)
    require(not root.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Redirected artifact')
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), 'Redirected artifact component')
    return path


def request(run):
    plan = read(safe_artifact(run, 'request.json'))
    require(plan['scope'] == SCOPE and plan['fit_deadline'] == plan['started'] + 7200
            and plan['deadline'] == plan['started'] + 8100, 'Pilot scope/deadline drift')
    for name, sha in plan['sources'].items():
        require(file_hash(safe_artifact(ROOT, name)) == sha, 'Pilot source drift: ' + name)
    return plan


def preparation(plan):
    require(plan['prepared_run_id'] == PREPARED.name, 'Unexpected preparation run')
    folder = PREPARED / 'prepare'
    require(file_hash(folder / 'complete.json') == plan['prepared_sha256'], 'Preparation binding drift')
    value = read(folder / 'complete.json')
    verify_files(folder, value)
    require(value['rows'] == {'train': 1270025, 'validation': 599278}
            and value['columns'] == plan['columns'], 'Wrong pilot population')
    return folder, value


def deadline(run, plan):
    path = Path(run) / 'fit-complete.json'
    if path.exists():
        fit = read(safe_artifact(run, 'fit-complete.json'))
        require(fit['request_sha256'] == file_hash(Path(run) / 'request.json')
                and plan['started'] <= fit['completed_epoch'] < plan['fit_deadline'], 'Fit deadline/binding drift')
        return min(plan['deadline'], fit['completed_epoch'] + 900)
    return plan['fit_deadline']


def guard(run, plan):
    require(plan['started'] <= time.time() < deadline(run, plan), 'Original pilot phase deadline exhausted or clock moved')


def supervise_owner(run, attempt):
    plan = request(run)
    owner = read(safe_artifact(run, 'attempts/' + attempt + '/owner.json'))
    done = threading.Event()
    started_wall, started_mono = time.time(), time.monotonic()

    def watch():
        while not done.wait(.3):
            try:
                alive = psutil.Process(owner['pid']).create_time() == owner['created']
                end = deadline(run, plan)
                valid = plan['started'] <= time.time() < end and started_wall + time.monotonic() - started_mono < end
            except (psutil.Error, OSError, ValueError, KeyError):
                alive = valid = False
            if not alive or not valid:
                os._exit(15)

    threading.Thread(target=watch, daemon=True).start()
    return plan, done


def verify_seal(run, plan):
    seal = read(safe_artifact(run, 'seal.json'))
    require(seal['request_sha256'] == file_hash(Path(run) / 'request.json')
            and seal['fit_sha256'] == file_hash(Path(run) / 'fit-complete.json')
            and seal['prepared_sha256'] == plan['prepared_sha256']
            and seal['prediction_rule'] == 'alpha3; reject iff all p<t; global argmax; equality accepted; first-index ties',
            'DOC seal binding drift')
    for name, sha in seal['files'].items():
        require(file_hash(safe_artifact(run, name)) == sha, 'Sealed artifact drift: ' + name)
    import numpy as np
    from .doc_core import CutoffMoments
    cutoffs = read(safe_artifact(run, 'cutoff-result.json'))
    require(cutoffs['request_sha256'] == seal['request_sha256'] and cutoffs['fit_sha256'] == seal['fit_sha256'],
            'Cutoff binding drift')
    calculated = CutoffMoments(np.asarray(cutoffs['counts']), np.asarray(cutoffs['squares'])).thresholds().tolist()
    require(calculated == cutoffs['thresholds'] == seal['thresholds'], 'Sealed cutoff drift')
    require(seal['files']['fit/' + seal['model_ref']['path']] == seal['model_ref']['sha256']
            and seal['inference'] == {'device': 'cuda:0', 'block_rows': 8192,
                'float': 'float32 sigmoid promoted to float64 comparison'}, 'Sealed inference drift')
    return seal
