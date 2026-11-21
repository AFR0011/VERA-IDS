"""Successor checkpoint import with the unchanged DOC epoch/update algorithm.

The static fit_arrays body is copied from the frozen first pilot; its sole engine
change is an explicitly injected initial state in a fresh bound journal.
"""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch
from . import doc_core as core, doc_training as training, doc_pilot_io as rt
from . import doc_recovery_contract as contract
from .doc_pilot_fit import save_tensor, load_tensor, order_for, verify_references, cutoffs
from .doc_readiness_fit import environment
from experiments.c03.producer.locking import file_lock


class RecoveryJournal(training.Journal):
    def __init__(self, *args, initial_state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state = initial_state

    def open(self):
        state = super().open()
        return copy.deepcopy(self.initial_state) if state is None else state


def read_checkpoint(donor, expected_commit):
    from .doc_runtime import verify_journal
    verify_journal(donor, expected_commit)
    verify_references(Path(donor) / 'fit')
    commit = sorted((Path(donor) / 'fit').glob('commit-*.json'))[-1]
    payload = commit.with_name(commit.name.replace('commit-', 'payload-').replace('.json', '.pt'))
    return torch.load(payload, map_location='cpu', weights_only=True)


def copy_initial_state(donor_fit, target_fit, state):
    target_fit = Path(target_fit); target_fit.mkdir(parents=True, exist_ok=True)
    for key in ('current_ref', 'best_ref'):
        ref = state.get(key)
        if ref is None: continue
        source = rt.safe_artifact(donor_fit, ref['path'])
        rt.require(rt.file_hash(source) == ref['sha256'], 'Imported model reference drift')
        target = rt.safe_artifact(target_fit, ref['path']); target.parent.mkdir(exist_ok=True)
        if not target.exists(): rt.publish(target, source.read_bytes())
        rt.require(rt.file_hash(target) == ref['sha256'], 'Imported model copy drift')
    return copy.deepcopy(state)


def import_checkpoint(run, plan):
    rt.guard(run, plan)
    binding = contract.binding()
    rt.require(plan['recovery'] == binding, 'Recovery contract drift')
    donor_plan = rt.request(contract.DONOR)
    rt.require(donor_plan['prepared_sha256'] == plan['prepared_sha256']
               and donor_plan['test'] == plan['test'] and donor_plan['columns'] == plan['columns'],
               'Recovery population or preparation drift')
    state = read_checkpoint(contract.DONOR, binding['donor_commit_sha256'])
    rt.require(state['phase'] == 'fit' and state['epoch'] == 38 and state['cursor'] == binding['cursor']
               and state['updates'] == binding['updates'] and state['best_epoch'] + 1 == binding['best_epoch']
               and state['best_loss'] == binding['best_validation_bce'] and state['bad_epochs'] == 0
               and len(state['history']) == 38, 'Donor progress differs')
    copied = copy_initial_state(contract.DONOR / 'fit', run / 'fit', state)
    rt.bound_json(run, 'recovery-import.json', {'recovery': binding, 'current_ref': state['current_ref'],
        'best_ref': state['best_ref'], 'history': state['history'], 'state_preserved': True})
    rt.guard(run, plan)
    return copied


def fit_arrays(root, train, labels, validation, val_labels, *, binding, device='cuda:0',
               epochs=100, checkpoint_updates=250, block_rows=8192, fixture=False, initial_state=None,
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
    journal = RecoveryJournal(root, definition, seconds=7200, initial_state=initial_state)
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


def run_fit(run, plan, emit):
    emit("verify_and_import_donor_checkpoint")
    initial_state = import_checkpoint(run, plan)
    folder, prepared = rt.preparation(plan)
    train, labels, validation, vy = [np.load(folder / name, mmap_mode='r', allow_pickle=False) for name in
        ('train-X.npy', 'train-y.npy', 'validation-X.npy', 'validation-y.npy')]
    binding = {'request_sha256': rt.file_hash(run / 'request.json'), 'prepared_sha256': plan['prepared_sha256'], 'recovery': plan['recovery']}
    model, state, commit = fit_arrays(run / 'fit', train, labels, validation, vy, binding=binding,
        guard=lambda: rt.guard(run, plan), emit=emit, initial_state=initial_state)
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


if __name__ == '__main__': main()
