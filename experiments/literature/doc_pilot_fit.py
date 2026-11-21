"""Fresh scientific DOC fit with compact recovery, TRAIN cutoffs and immutable seal."""
import argparse
import io
from pathlib import Path
import time

import numpy as np
import torch

from . import doc_core as core, doc_training as training, doc_pilot_io as rt
from .doc_readiness_fit import environment
from experiments.c03.producer.locking import file_lock


def save_tensor(root, value):
    stream = io.BytesIO()
    torch.save(value, stream)
    data = stream.getvalue()
    import hashlib
    sha = hashlib.sha256(data).hexdigest()
    path = Path(root) / 'weights' / (sha + '.pt')
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        rt.publish(path, data)
    rt.require(rt.file_hash(path) == sha, 'Model payload drift')
    return {'path': path.relative_to(root).as_posix(), 'sha256': sha}


def load_tensor(root, ref):
    path = rt.safe_artifact(root, ref['path'])
    rt.require(rt.file_hash(path) == ref['sha256'], 'Referenced model payload drift')
    return torch.load(path, map_location='cpu', weights_only=True)


def order_for(root, epoch, rows):
    """One deterministic row-order artifact per epoch; never copied into checkpoints."""
    path = Path(root) / f'order-{epoch:03d}.npy'
    expected = np.random.default_rng(np.random.SeedSequence([123, epoch])).permutation(rows)
    if not path.exists():
        stream = io.BytesIO(); np.save(stream, expected, allow_pickle=False)
        rt.publish(path, stream.getvalue())
    actual = np.load(path, allow_pickle=False, mmap_mode='r')
    rt.require(np.array_equal(actual, expected), 'Epoch row order drift')
    return actual


def verify_references(root):
    """Check every committed reference, including checkpoints preceding the selected epoch."""
    checked = set()
    for commit in sorted(Path(root).glob('commit-*.json')):
        payload = commit.with_name(commit.name.replace('commit-', 'payload-').replace('.json', '.pt'))
        rt.require(rt.file_hash(payload) == rt.read(commit)['payload_sha256'], 'Journal payload drift')
        state = torch.load(payload, map_location='cpu', weights_only=True)
        for key in ('current_ref', 'best_ref'):
            ref = state.get(key)
            if ref and ref['path'] not in checked:
                rt.require(rt.file_hash(rt.safe_artifact(root, ref['path'])) == ref['sha256'], 'Historical model reference drift')
                checked.add(ref['path'])


def fit_arrays(root, train, labels, validation, val_labels, *, binding, device='cuda:0',
               epochs=100, checkpoint_updates=250, block_rows=8192, fixture=False,
               guard=lambda: None, emit=lambda *a, **k: None, hook=lambda event: None):
    """Production schedule is fixed; overrides require bounded generated fixtures."""
    guard()
    if fixture:
        rt.require(0 < len(train) <= 4096 and 0 < len(validation) <= 4096, 'Synthetic row cap')
    else:
        rt.require(device == 'cuda:0' and len(train) == 1270025 and len(validation) == 599278
                   and (epochs, checkpoint_updates, block_rows) == (100, 250, 8192), 'Frozen scientific surface required')
        environment()
    rt.require(train.shape == (len(train), 46) and validation.shape == (len(validation), 46)
               and train.dtype == validation.dtype == np.dtype('float32'), 'Tensor layout differs')
    core.labels_checked(labels, len(train), 6); core.labels_checked(val_labels, len(validation), 6)
    rt.require(np.all(np.bincount(labels, minlength=6) > 0), 'Missing TRAIN class')
    training.configure(device)
    definition = {'binding': binding, 'config': training.CONFIG, 'epochs': epochs,
                  'checkpoint_updates': checkpoint_updates, 'block_rows': block_rows,
                  'device': device, 'fixture': fixture, 'torch': str(torch.__version__)}
    if fixture:
        definition['fixture_inputs'] = [training.array_hash(a) for a in (train, labels, validation, val_labels)]
    journal = training.Journal(root, definition, seconds=7200)
    with journal.lock():
        state = journal.open()
        model = training.network(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
        if state is None:
            state = {'phase': 'fit', 'epoch': 0, 'cursor': 0, 'updates': 0, 'best_loss': None,
                     'best_epoch': None, 'best_ref': None, 'bad_epochs': 0, 'history': [],
                     'val_cursor': 0, 'val_sum': 0.0}
        else:
            payload = load_tensor(root, state['current_ref'])
            model.load_state_dict(payload['model']); optimizer.load_state_dict(payload['optimizer'])
            training.restore_rng(payload['rng'], device)

        def checkpoint(weights=False):
            guard()
            if weights:
                state['current_ref'] = save_tensor(root, {'model': training.snapshot(model),
                    'optimizer': optimizer.state_dict(), 'rng': training.rng_state(device)})
                hook('weights_published')
            journal.save(state, hook)

        if not journal.previous:
            checkpoint(weights=True)
        order = None; order_epoch = None
        while state['phase'] != 'complete':
            guard(); journal.guard()
            if state['phase'] == 'fit':
                if order_epoch != state['epoch']:
                    order = order_for(root, state['epoch'], len(train)); order_epoch = state['epoch']
                positions = order[state['cursor']:state['cursor'] + 128]
                x = np.array(train[positions], dtype=np.float32, copy=True)
                rt.require(np.isfinite(x).all(), 'Nonfinite TRAIN input')
                y = np.array(labels[positions], dtype=np.int64, copy=True)
                model.train(); optimizer.zero_grad(set_to_none=True)
                loss = training.loss_rows(model(torch.as_tensor(x, device=device)), torch.as_tensor(y, device=device)).mean()
                rt.require(torch.isfinite(loss).item(), 'Nonfinite scientific loss')
                loss.backward(); optimizer.step(); hook('after_update')
                state['cursor'] += len(positions); state['updates'] += 1
                if state['cursor'] == len(train):
                    state['phase'] = 'validation'; state['val_cursor'] = 0; state['val_sum'] = 0.0
                if state['updates'] % checkpoint_updates == 0 or state['phase'] == 'validation':
                    checkpoint(weights=True)
                    emit('fit', epoch=state['epoch'] + 1, rows=state['cursor'], total=len(train), updates=state['updates'])
            elif state['phase'] == 'validation':
                model.eval(); start = state['val_cursor']; end = min(start + block_rows, len(validation))
                x = np.array(validation[start:end], dtype=np.float32, copy=True)
                rt.require(np.isfinite(x).all(), 'Nonfinite validation input')
                y = np.array(val_labels[start:end], dtype=np.int64, copy=True)
                with torch.no_grad():
                    loss = training.loss_rows(model(torch.as_tensor(x, device=device)), torch.as_tensor(y, device=device))
                rt.require(torch.isfinite(loss).all().item(), 'Nonfinite validation loss')
                state['val_sum'] += loss.double().sum().item(); state['val_cursor'] = end
                hook('validation_block')
                if end == len(validation):
                    value = state['val_sum'] / len(validation); state['history'].append(value)
                    if state['best_loss'] is None or value < state['best_loss']:
                        state['best_loss'], state['best_epoch'] = value, state['epoch']
                        state['best_ref'], state['bad_epochs'] = dict(state['current_ref']), 0
                    else:
                        state['bad_epochs'] += 1
                    state['epoch'] += 1
                    state['phase'] = 'complete' if state['epoch'] == epochs or state['bad_epochs'] >= 5 else 'fit'
                    state['cursor'] = 0
                    emit('epoch_complete', epoch=state['epoch'], validation_bce=value,
                         best_epoch=state['best_epoch'] + 1, bad_epochs=state['bad_epochs'])
                checkpoint()
            else:
                raise ValueError('Unknown scientific checkpoint phase')
        model.load_state_dict(load_tensor(root, state['best_ref'])['model'])
        guard()
        return model, state, journal.previous


def cutoffs(run, model, train, labels, *, device, guard, emit=lambda *a, **k: None, block=8192):
    moments = core.CutoffMoments.empty(6)
    for i, start in enumerate(range(0, len(train), block)):
        guard(); end = min(start + block, len(train)); name = f'cutoffs/{i:04d}.json'
        path = Path(run) / name
        # Cheap replay verifies committed block moments rather than trusting edited JSON.
        partial = core.CutoffMoments.empty(6)
        partial.update(training.probabilities(model, np.array(train[start:end]), device), labels[start:end])
        rt.bound_json(run, name, {'fit_sha256': rt.file_hash(Path(run) / 'fit-complete.json'),
            'start': start, 'end': end, 'counts': partial.counts.tolist(), 'squares': partial.squares.tolist()})
        moments.counts += partial.counts; moments.squares += partial.squares
        if i % 10 == 0:
            emit('TRAIN_cutoffs', rows=end, total=len(train))
    rt.require(np.array_equal(moments.counts, np.bincount(labels, minlength=6)), 'Cutoff support differs')
    rt.bound_json(run, 'cutoff-result.json', {'counts': moments.counts.tolist(), 'squares': moments.squares.tolist(),
        'thresholds': moments.thresholds().tolist(), 'fit_sha256': rt.file_hash(Path(run) / 'fit-complete.json')})
    return moments.thresholds()


def run_fit(run, plan, emit):
    folder, prepared = rt.preparation(plan)
    train, labels, validation, vy = [np.load(folder / name, mmap_mode='r', allow_pickle=False) for name in
        ('train-X.npy', 'train-y.npy', 'validation-X.npy', 'validation-y.npy')]
    binding = {'request_sha256': rt.file_hash(run / 'request.json'), 'prepared_sha256': plan['prepared_sha256']}
    model, state, commit = fit_arrays(run / 'fit', train, labels, validation, vy, binding=binding,
        guard=lambda: rt.guard(run, plan), emit=emit)
    if not (run / 'fit-complete.json').exists():
        rt.require(time.time() < plan['fit_deadline'], 'Fit budget exhausted')
        rt.bound_json(run, 'fit-complete.json', {'completed_epoch': time.time(), 'journal_commit': commit,
            'epochs': state['epoch'], 'updates': state['updates'], 'best_epoch': state['best_epoch'] + 1,
            'best_validation_bce': state['best_loss'], 'history': state['history'], 'best_ref': state['best_ref'],
            'stop_reason': 'patience' if state['bad_epochs'] >= 5 else 'epoch_cap', 'prepared_sha256': plan['prepared_sha256']})
    fit = rt.read(run / 'fit-complete.json')
    rt.require(fit['journal_commit'] == commit and fit['best_ref'] == state['best_ref'], 'Completed fit drift')
    thresholds = cutoffs(run, model, train, labels, device='cuda:0', guard=lambda: rt.guard(run, plan), emit=emit)
    ref = save_tensor(run / 'fit', {'model': training.snapshot(model)})
    before = training.probabilities(model, np.array(validation[:8192]), 'cuda:0')
    other = training.network('cuda:0'); other.load_state_dict(load_tensor(run / 'fit', ref)['model'])
    rt.require(np.array_equal(before, training.probabilities(other, np.array(validation[:8192]), 'cuda:0')), 'Sealed reload differs')
    rt.guard(run, plan)
    verify_references(run / 'fit')
    rt.bound_json(run, 'fit-inventory.json', {'files': {
        p.relative_to(run).as_posix(): rt.file_hash(p) for p in sorted((run / 'fit').rglob('*'))
        if p.is_file() and p.suffix in ('.pt', '.json', '.npy')}})
    rt.bound_json(run, 'seal.json', {'fit_sha256': rt.file_hash(run / 'fit-complete.json'),
        'prepared_sha256': plan['prepared_sha256'], 'model_ref': ref, 'thresholds': thresholds.tolist(),
        'prediction_rule': 'alpha3; reject iff all p<t; global argmax; equality accepted; first-index ties',
        'inference': {'device': 'cuda:0', 'block_rows': 8192, 'float': 'float32 sigmoid promoted to float64 comparison'},
        'reload_exact_rows': 8192, 'files': {'fit/' + ref['path']: ref['sha256'],
            'cutoff-result.json': rt.file_hash(run / 'cutoff-result.json'),
            'fit-inventory.json': rt.file_hash(run / 'fit-inventory.json')}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--attempt', required=True)
    args = parser.parse_args(); plan, done = rt.supervise_owner(args.run, args.attempt)
    try:
        with file_lock(args.run / '.worker.guard'):
            run_fit(args.run, plan, lambda stage, **info: rt.emit(args.run, args.attempt, stage, **info))
    finally:
        done.set()


if __name__ == '__main__':
    main()
