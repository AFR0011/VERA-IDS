"""DOC tabular training primitives and synthetic-only resumable fit.

No benchmark command is exposed. A real-data caller needs the separate readiness
and population/seal contract. PyTorch is optional and isolated from the release env.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.locking import file_lock
from experiments.c03.producer import checkpoints, locking
from . import doc_core as core

CONFIG = {'width': 46, 'hidden': 250, 'classes': 6, 'dropout': .2, 'batch_size': 128,
          'seed': 123, 'max_epochs': 100, 'patience': 5, 'learning_rate': .001,
          'betas': [.9, .999], 'epsilon': 1e-8, 'checkpoint_updates': 250, 'block_rows': 8192}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    h = hashlib.sha256(canonical({'shape': list(array.shape), 'dtype': str(array.dtype)}))
    for start in range(0, len(array), 8192):
        h.update(np.ascontiguousarray(array[start:start + 8192]).tobytes())
    return h.hexdigest()


def configure(device):
    core.require(device in ('cpu', 'cuda:0'), 'Explicit CPU or GPU0 required')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device == 'cuda:0':
        core.require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    if device == 'cuda:0':
        torch.cuda.manual_seed_all(123)


def network(device):
    model = nn.Sequential(nn.Linear(46, 250), nn.ReLU(), nn.Dropout(.2), nn.Linear(250, 6))
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model.to(device)


def loss_rows(logits, labels):
    target = torch.nn.functional.one_hot(labels, num_classes=6).to(dtype=logits.dtype)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none').sum(dim=1)


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def rng_state(device):
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]],
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if device == 'cuda:0' else []}


def restore_rng(state, device):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['cpu'].cpu())
    if device == 'cuda:0':
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


class Journal:
    """Immutable payload/commit pairs. Caller holds lock across the fit.

    Payload orphans are ignored; a committed payload with a bad hash fails closed.
    Checkpoints are locally generated and loaded with weights_only=True.
    """
    def __init__(self, root, definition, seconds=900):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        core.require(not self.root.is_symlink(), 'Redirected checkpoint directory')
        self.definition = definition
        self.seconds = seconds
        self.previous = None
        self.ordinal = 0

    def lock(self):
        return file_lock(self.root / '.writer.guard')

    def open(self):
        path = self.root / 'plan.json'
        if not path.exists():
            now = time.time()
            publish(path, canonical({'definition': self.definition, 'started': now,
                                     'deadline': now + self.seconds, 'seconds': self.seconds}))
        core.require(not path.is_symlink(), 'Redirected plan')
        self.plan = json.loads(path.read_text())
        core.require(self.plan['definition'] == self.definition and self.plan['seconds'] == self.seconds
                     and self.plan['deadline'] == self.plan['started'] + self.seconds, 'Plan/input/source/environment drift')
        self.plan_hash = sha(path)
        self.previous = None
        latest = None
        for record_path in sorted(self.root.glob('commit-*.json')):
            core.require(not record_path.is_symlink(), 'Redirected commit')
            record = json.loads(record_path.read_text())
            core.require(record['plan_sha256'] == self.plan_hash and record['previous'] == self.previous, 'Broken checkpoint chain')
            payload = self.root / record_path.name.replace('commit-', 'payload-').replace('.json', '.pt')
            core.require(payload.is_file() and not payload.is_symlink() and sha(payload) == record['payload_sha256'], 'Checkpoint payload drift')
            self.previous = sha(record_path)
            latest = payload
        entries = list(self.root.glob('payload-*.pt')) + list(self.root.glob('commit-*.json'))
        self.ordinal = max([int(p.stem.split('-')[1]) for p in entries], default=0)
        return torch.load(latest, map_location='cpu', weights_only=True) if latest else None

    def guard(self):
        core.require(self.plan['started'] <= time.time() < self.plan['deadline'], 'Original deadline exhausted or clock moved')

    def save(self, state, hook=lambda event: None):
        self.guard()
        self.ordinal += 1
        stem = f'{self.ordinal:06d}'
        payload = self.root / f'payload-{stem}.pt'
        data = io.BytesIO()
        torch.save(state, data)
        publish(payload, data.getvalue())
        hook('payload_published')
        record = {'plan_sha256': self.plan_hash, 'previous': self.previous, 'payload_sha256': sha(payload)}
        record_path = self.root / f'commit-{stem}.json'
        publish(record_path, canonical(record))
        self.previous = sha(record_path)
        hook('commit_published')


def probabilities(model, x, device, block=8192):
    core.matrix(x, 46)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), block):
            p = torch.sigmoid(model(torch.as_tensor(x[start:start + block], dtype=torch.float32, device=device)))
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def synthetic_fit(root, train, labels, validation, validation_labels, *, device='cpu',
                  epochs=3, checkpoint_updates=2, block_rows=8192, hook=lambda event: None, guard=lambda: None):
    """Generated-data harness only (<=4096 rows); epoch override is diagnostic.

    Resume includes validation and cutoff block cursors; no benchmark adapters,
    feature transforms or held-out-family arrays are accepted through a CLI.
    """
    core.matrix(train, 46)
    core.matrix(validation, 46)
    labels = core.labels_checked(labels, len(train), 6)
    validation_labels = core.labels_checked(validation_labels, len(validation), 6)
    core.require(len(train) <= 4096 and len(validation) <= 4096, 'Synthetic harness row cap')
    core.require(set(labels.tolist()) == set(range(6)), 'Missing TRAIN classes')
    core.require(type(epochs) is int and 1 <= epochs <= 100 and type(checkpoint_updates) is int
                 and checkpoint_updates > 0 and type(block_rows) is int and block_rows > 0, 'Invalid diagnostic schedule')
    configure(device)
    definition = {'scope': 'DOC_SYNTHETIC_ONLY', 'config': CONFIG, 'epochs': epochs,
                  'checkpoint_updates': checkpoint_updates, 'block_rows': block_rows, 'device': device,
                  'inputs': [array_hash(v) for v in (train, labels, validation, validation_labels)],
                  'sources': {p.name: sha(p) for p in (Path(__file__), Path(core.__file__),
                              Path(checkpoints.__file__), Path(locking.__file__))},
                  'environment': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__),
                                  'cuda': torch.version.cuda,
                                  'gpu': torch.cuda.get_device_name(0) if device == 'cuda:0' else None}}
    journal = Journal(root, definition)
    with journal.lock():
        state = journal.open()
        model = network(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
        if state is None:
            state = {'phase': 'fit', 'epoch': 0, 'cursor': 0, 'updates': 0, 'best_loss': None,
                     'best_epoch': None, 'best_model': None, 'bad_epochs': 0, 'history': [],
                     'order': torch.from_numpy(np.random.default_rng(np.random.SeedSequence([123, 0])).permutation(len(train))),
                     'val_cursor': 0, 'val_sum': 0.0, 'cutoff_cursor': 0,
                     'counts': torch.zeros(6, dtype=torch.int64), 'squares': torch.zeros(6, dtype=torch.float64)}
        else:
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['rng'], device)

        def checkpoint():
            state['model'] = snapshot(model)
            state['optimizer'] = optimizer.state_dict()
            state['rng'] = rng_state(device)
            journal.save(state, hook)

        if state['phase'] == 'complete':
            return model, state
        if not journal.previous:
            checkpoint()
        while state['phase'] != 'complete':
            journal.guard()
            guard()
            if state['phase'] == 'fit':
                model.train()
                start = state['cursor']
                indices = state['order'][start:start + 128].numpy()
                x = torch.as_tensor(train[indices], dtype=torch.float32, device=device)
                y = torch.as_tensor(labels[indices], dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_rows(model(x), y).mean()
                core.require(torch.isfinite(loss).item(), 'Nonfinite training loss')
                loss.backward()
                optimizer.step()
                hook('after_update')
                state['cursor'] += len(indices)
                state['updates'] += 1
                if state['cursor'] == len(train):
                    state['phase'] = 'validation'
                    state['val_cursor'], state['val_sum'] = 0, 0.0
                if state['updates'] % checkpoint_updates == 0 or state['phase'] == 'validation':
                    checkpoint()
            elif state['phase'] == 'validation':
                model.eval()
                start = state['val_cursor']
                end = min(start + block_rows, len(validation))
                with torch.no_grad():
                    rows = loss_rows(model(torch.as_tensor(validation[start:end], dtype=torch.float32, device=device)),
                                     torch.as_tensor(validation_labels[start:end], dtype=torch.long, device=device))
                state['val_sum'] += rows.double().sum().item()
                state['val_cursor'] = end
                hook('validation_block')
                if end == len(validation):
                    value = state['val_sum'] / len(validation)
                    core.require(np.isfinite(value), 'Nonfinite validation loss')
                    state['history'].append(value)
                    if state['best_loss'] is None or value < state['best_loss']:
                        state['best_loss'], state['best_epoch'] = value, state['epoch']
                        state['best_model'], state['bad_epochs'] = snapshot(model), 0
                    else:
                        state['bad_epochs'] += 1
                    print(f'DOC synthetic epoch {state["epoch"] + 1} | validation BCE {value:.6f} | updates {state["updates"]}', flush=True)
                    state['epoch'] += 1
                    if state['epoch'] == epochs or state['bad_epochs'] >= 5:
                        model.load_state_dict(state['best_model'])
                        state['phase'] = 'cutoffs'
                    else:
                        state['order'] = torch.from_numpy(np.random.default_rng(np.random.SeedSequence([123, state['epoch']])).permutation(len(train)))
                        state['cursor'], state['phase'] = 0, 'fit'
                checkpoint()
            elif state['phase'] == 'cutoffs':
                start = state['cutoff_cursor']
                end = min(start + block_rows, len(train))
                moments = core.CutoffMoments(state['counts'].numpy().copy(), state['squares'].numpy().copy())
                moments.update(probabilities(model, train[start:end], device), labels[start:end])
                state['counts'] = torch.from_numpy(moments.counts)
                state['squares'] = torch.from_numpy(moments.squares)
                state['cutoff_cursor'] = end
                hook('cutoff_block')
                if end == len(train):
                    state['thresholds'] = torch.from_numpy(moments.thresholds())
                    state['phase'] = 'complete'
                checkpoint()
            else:
                raise ValueError('Unknown checkpoint phase')
        return model, state
