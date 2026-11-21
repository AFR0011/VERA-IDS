"""Exact known TRAIN/validation import and checkpointed TRAIN-only DOC scaling."""
import gc
import os
from pathlib import Path

import numpy as np

from . import doc_core as core, doc_runtime as rt, efc_input_probe as inputs
from experiments.c03.producer.artifacts import array_hash, generated_file, safe_artifact
from experiments.c03.producer.plan import digest, require, file_hash, canonical
from experiments.c03.producer.checkpoints import publish


def mapped(path, shape, dtype):
    """Atomically publish a header/allocation; mutable values are checkpoint-bound."""
    path = Path(path)
    if not path.exists():
        with generated_file(path) as temporary:
            a = np.lib.format.open_memmap(temporary, mode='w+', dtype=dtype, shape=shape)
            a.flush()
            del a
    require(not path.is_symlink(), 'Redirected array')
    a = np.load(path, mmap_mode='r+', allow_pickle=False)
    require(a.shape == shape and a.dtype == np.dtype(dtype), 'Array layout drift')
    return a


def flush(array):
    array.flush()
    with Path(array.filename).open('r+b') as stream:
        stream.flush(); os.fsync(stream.fileno())


def save_rows(run, label, number, x, y, positions, values, targets, source):
    core.matrix(values, 46)
    targets = core.labels_checked(targets, len(values), 6)
    positions = np.asarray(positions)
    require(positions.shape == (len(values),) and np.issubdtype(positions.dtype, np.integer)
            and (positions >= 0).all() and (positions < len(x)).all()
            and len(np.unique(positions)) == len(positions), 'Invalid selected positions')
    expected = {'source': source, 'positions_sha256': array_hash(positions),
                'X_sha256': array_hash(values), 'y_sha256': array_hash(targets.astype(np.int64)), 'rows': len(values)}
    name = f'prepare/parts/{label}-{number:04d}.json'
    if (Path(run) / name).exists():
        rt.bound_json(run, name, expected)
        require(array_hash(x[positions]) == expected['X_sha256']
                and array_hash(y[positions]) == expected['y_sha256'], 'Checkpointed array rows drift')
    else:
        x[positions] = values; y[positions] = targets
        flush(x); flush(y)
        rt.bound_json(run, name, expected)


def standardize(run, x, validation, guard=lambda: None, emit=lambda _: None, block=8192):
    """Ordered block checkpoints preserve moments and transformed bytes on recovery."""
    folder = Path(run) / 'prepare'; folder.mkdir(exist_ok=True)
    stats = core.Standardizer.empty(46)
    for i, start in enumerate(range(0, len(x), block)):
        guard(); end = min(start + block, len(x))
        stats.update(x[start:end])
        rt.bound_json(run, f'prepare/moments/{i:04d}.json',
                      {'end': end, 'count': stats.count, 'mean': stats.mean.tolist(), 'm2': stats.m2.tolist()})
    rt.bound_json(run, 'prepare/scaler.json', {'count': stats.count, 'mean': stats.mean.tolist(),
                                            'm2': stats.m2.tolist(), 'scale': stats.scale.tolist()})
    for label, raw in [('train', x), ('validation', validation)]:
        out = mapped(folder / f'{label}-X.npy', raw.shape, np.float32)
        for i, start in enumerate(range(0, len(raw), block)):
            guard(); end = min(start + block, len(raw))
            values = stats.transform(raw[start:end])
            name = f'prepare/transforms/{label}-{i:04d}.json'
            expected = {'start': start, 'end': end, 'sha256': array_hash(values),
                        'scaler_sha256': file_hash(folder / 'scaler.json')}
            if (Path(run) / name).exists():
                rt.bound_json(run, name, expected)
                require(array_hash(out[start:end]) == expected['sha256'], 'Transformed checkpoint drift')
            else:
                out[start:end] = values; flush(out)
                rt.bound_json(run, name, expected)
            if i % 25 == 0: emit(f'scale_{label}_{end}_of_{len(raw)}')
        del out
    return stats


def validation_index(plan):
    reference = inputs.REFERENCE
    path = reference / 'prepared/validation.npz'
    require(file_hash(path) == plan['validation']['array_sha256'], 'Validation identity archive drift')
    with np.load(path, allow_pickle=False) as a:
        parts, shas, offsets, truth = [a[k] for k in ('source_part','source_sha256','row_offset','target')]
    require(len(truth) == 700000 and array_hash(truth) == plan['validation']['truth_sha256'], 'Validation truth drift')
    require(digest({k: array_hash(v) for k,v in zip(('source_part','source_sha256','row_offset'),(parts,shas,offsets))})
            == plan['validation']['identity_sha256'], 'Validation row identity drift')
    require(np.isin(truth,np.arange(7)).all(), 'Invalid validation classes')
    known = truth < 6
    require(int(known.sum()) == 599278, 'Known validation count differs')
    positions = np.flatnonzero(known).astype(np.int64)
    selected = (parts[known], shas[known], offsets[known], truth[known])
    inventory = {v['path']:v['sha256'] for v in plan['validation']['inventory']}
    require(set(selected[0]) <= set(inventory), 'Unregistered validation source')
    for part in np.unique(selected[0]):
        mask = selected[0] == part
        require((selected[1][mask] == inventory[part]).all() and len(np.unique(selected[2][mask])) == int(mask.sum()),
                'Validation source digest/duplicate identity drift')
    return (*selected, positions)


def prepare(run, plan, emit=lambda _: None):
    run = Path(run); folder = run/'prepare'; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        record = rt.read(folder/'complete.json')
        require(record['request_sha256'] == file_hash(run/'request.json'), 'Preparation request differs')
        rt.verify_files(folder, record)
        return record
    efc = inputs.OUTPUT / plan['train']['run_id']
    require(file_hash(efc/'complete.json') == plan['train']['completion_sha256'], 'EFC input completion drift')
    imported = inputs.verify_completed(efc)
    require(imported['rows'] == 1270025 and imported['columns'] == plan['columns'], 'EFC input layout differs')
    # Reconcile original selected identities as well as the prepared values.
    original = rt.read(efc/'request.json')['definition']
    train_parts, _, _ = inputs.load_index(original)
    x = mapped(folder/'train-raw.npy', (1270025,46), np.float64)
    y = mapped(folder/'train-y.npy', (1270025,), np.int64)
    seen = np.zeros(len(x),dtype=bool)
    for number, item in enumerate(imported['parts']):
        rt.guard(plan)
        source = safe_artifact(inputs.host.DATA_ROOT, item['source']['path'])
        require(file_hash(source) == item['source']['sha256'], 'TRAIN source drift')
        path = efc/item['part']/'values.npz'
        require(file_hash(path) == item['values_sha256'], 'TRAIN values drift')
        with np.load(path,allow_pickle=False) as a:
            positions, values, targets = a['positions'], a['X'], a['y']
        require(np.array_equal(positions,np.flatnonzero(train_parts == item['source']['path']))
                and not seen[positions].any(), 'TRAIN selected positions differ')
        save_rows(run,'train',number,x,y,positions,values,targets,item['source'])
        seen[positions] = True
        if number % 20 == 0: emit(f'import_train_part_{number+1}_of_{len(imported["parts"])}')
    require(seen.all() and np.bincount(y,minlength=6).tolist() == plan['train']['class_support'], 'TRAIN coverage/support differs')
    del train_parts, seen, positions, values, targets
    gc.collect()
    parts, shas, offsets, truth, selected_positions = validation_index(plan)
    v = mapped(folder/'validation-raw.npy',(599278,46),np.float64)
    vy = mapped(folder/'validation-y.npy',(599278,),np.int64)
    position_file = folder/'validation-sample-positions.npy'
    indices = mapped(position_file,(599278,),np.int64)
    # Fixed identity data can be safely rewritten before its immutable binding.
    indices[:] = selected_positions; flush(indices); del indices
    seen = np.zeros(len(v),dtype=bool)
    for number, entry in enumerate(plan['validation']['inventory']):
        rt.guard(plan)
        require(entry['path'].startswith('val/'), 'Only validation sources allowed')
        positions = np.flatnonzero(parts == entry['path'])
        if not len(positions): continue
        source = safe_artifact(inputs.host.DATA_ROOT,entry['path'])
        values, targets = inputs.extract_part(source, entry['sha256'], plan['columns'],
                                              offsets[positions], truth[positions] != 0)
        require(np.array_equal(targets,truth[positions]) and not seen[positions].any(), 'Known validation labels/coverage differ')
        save_rows(run,'validation',number,v,vy,positions,values,targets,entry)
        seen[positions] = True
        if number % 10 == 0: emit(f'extract_known_validation_part_{number+1}_of_{len(plan["validation"]["inventory"])}')
    require(seen.all() and np.array_equal(vy,truth), 'Incomplete known validation population')
    identity = digest({k:array_hash(a) for k,a in zip(('source_part','source_sha256','row_offset'),(parts,shas,offsets))})
    del seen, parts, shas, offsets, truth, selected_positions, positions, values, targets
    gc.collect()
    emit('fit_TRAIN_standardizer_and_transform')
    standardize(run,x,v,guard=lambda:rt.guard(plan),emit=emit)
    # Preserve exact raw imports and verify source inventory again before closure.
    for entry in [*original['inventory'], *plan['validation']['inventory']]:
        rt.guard(plan)
        require(file_hash(safe_artifact(inputs.host.DATA_ROOT,entry['path'])) == entry['sha256'], 'Source changed during preparation')
    result = {'rows':{'train':len(x),'validation':len(v)}, 'columns':plan['columns'],
              'class_support':{'train':np.bincount(y,minlength=6).tolist(),'validation':np.bincount(vy,minlength=6).tolist()},
              'train_identity_sha256':imported['identity_sha256'],'validation_identity_sha256':identity,
              'validation_parent_identity_sha256':plan['validation']['identity_sha256'],
              'files':{p.name:file_hash(p) for p in folder.iterdir() if p.is_file() and p.suffix in ('.npy','.json')},
              'scaling_fit_population':'TRAIN_ONLY','cutoffs_fitted':False,'test_read':False}
    del x,y,v,vy
    rt.guard(plan)
    return rt.bound_json(run,'prepare/complete.json',result)
