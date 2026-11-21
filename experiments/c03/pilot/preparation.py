"""Fresh scientific preparation using the already verified historical algorithm."""
import json
import numpy as np
import pyarrow.parquet as pq

from .contract import SCOPE
from experiments.c03.producer import validation as v, models
from experiments.c03.producer.artifacts import array_hash, safe_artifact
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.plan import canonical, digest, file_hash, require
from experiments.c03.producer.transforms import transform_selected, BLOCK_ROWS


def prepare(store, sources, emit):
    require(store.plan.get('evidence_scope') == SCOPE and store.plan.get('execution_authorized') is True,
            'Scientific preparation requires explicit fixed-case authorization')
    emit('verify_train_validation_sources')
    sources.verify()
    _, grid, engine = v.historical()
    store.verify_plan()
    directory = store.path/'prepared'
    directory.mkdir(exist_ok=False)
    preprocessing = directory/'preprocessor'
    preprocessing.mkdir()
    emit('fit_train_stream_preprocessor')
    prep = engine.fit_preprocessor(str(sources.root), str(preprocessing))
    prep_sha = file_hash(preprocessing/'preprocessor.joblib')
    sampler = sources.sampler()
    emit('sample_train')
    train = sampler.sample('train', store.plan['sampling']['train_cap'], store.plan['sampling']['train_seed'])
    emit('sample_validation')
    val = sampler.sample('val', store.plan['sampling']['val_cap'], store.plan['sampling']['val_seed'])
    yt, yv = v.truth_labels(train.frame), v.truth_labels(val.frame)
    for label in range(7):
        require(int((yv == label).sum()) >= 200, 'Insufficient validation support')
    for label in range(6):
        require(int((yt == label).sum()) >= 200, 'Insufficient supervised training support')
    masks = {'stage1': yt != 6, 'stage2': (yt >= 1) & (yt <= 5)}
    require(set(yt[masks['stage2']]) == set(range(1, 6)), 'Represented family missing')
    tensors, targets, ids, weights = {}, {}, {}, {}
    for stage, mask in masks.items():
        emit(stage+'_tensor')
        tensors[stage] = transform_selected(prep, train.frame, mask)
        targets[stage] = ((yt[mask] != 0).astype(int) if stage == 'stage1' else (yt[mask]-1).astype(int))
        ids[stage] = train.identities.loc[mask].reset_index(drop=True)
        for name, node in store.plan['graph']['components'].items():
            if node['stage'] == stage:
                weights[name] = (grid.make_stage1_sample_weights(targets[stage],
                    train.frame.loc[mask, 'y_stage2_family'].astype(str).to_numpy(dtype=object),
                    node['weight_mode'], float(grid.CFG['inv_family_clip_max'])) if stage == 'stage1'
                    else grid.stage2_sample_weights(targets[stage], 5, node['weight_mode']))
    emit('validation_tensor')
    Xv = transform_selected(prep, val.frame, np.ones(len(val.frame), dtype=bool))
    data = v.Prepared(tensors, targets, ids, weights, Xv, yv, val.identities, prep_sha)
    binding = v.input_bindings(data, store.plan)
    store.validate_inputs(binding)
    expected = store.plan['expected_support']
    require((int((yv != 6).sum()), int((yv == 6).sum())) ==
            (expected['validation_K'], expected['validation_U']), 'Validation support differs from frozen plan')
    for stage in masks:
        emit('serialize_'+stage)
        v.save_tensor(directory/f'{stage}.npz', tensors[stage], targets[stage], ids[stage],
            {k: w for k, w in weights.items() if store.plan['graph']['components'][k]['stage'] == stage})
    emit('serialize_validation')
    v.save_tensor(directory/'validation.npz', Xv, yv, val.identities, {})
    sources.verify()
    prefix, seen = [], 0
    for part in sources.parts['train']:
        rows = pq.ParquetFile(sources.path(part)).metadata.num_rows
        prefix.append({'source_part': part.name, 'rows': rows})
        seen += rows
        if seen >= engine.CFG['max_stat_rows']:
            break
    stat_rows = (np.random.RandomState(engine.CFG['global_seed']).choice(seen, size=engine.CFG['max_stat_rows'], replace=False)
                 if seen > engine.CFG['max_stat_rows'] else np.arange(seen))
    receipt = {'scope': SCOPE, 'plan_sha256': digest(store.plan), 'fresh_preparation': True,
        'source_inventory': sources.inventory(), 'bindings': binding, 'validation': v.validation_identity(data),
        'transform_execution': {'rows_per_block': BLOCK_ROWS, 'passes': 2, 'historical_transform_unchanged': True},
        'preprocessor_metadata': json.loads((preprocessing/'preprocessor_meta.json').read_text()),
        'preprocessor_statistics': {'ordered_prefix': prefix, 'selected_prefix_positions_sha256': array_hash(stat_rows),
            'selected_rows': len(stat_rows), 'inference_part': sources.parts['train'][0].name, 'inference_max_rows': 5000},
        'helper_config': models.json_parameters(engine.CFG),
        'files': {p.relative_to(store.path).as_posix(): file_hash(p) for p in sorted(directory.rglob('*')) if p.is_file()}}
    publish(store.path/'inputs.json', canonical({'plan_sha256': digest(store.plan),
        'bindings': binding, 'bindings_sha256': digest(binding)}))
    publish(directory/'complete.json', canonical(receipt))
    emit('preparation_committed')
    return receipt


def verify(store, *, array=None):
    store.verify_plan()
    receipt = store.read_json(store.path/'prepared/complete.json')
    require(receipt['scope'] == SCOPE and receipt['fresh_preparation'] is True
            and receipt['plan_sha256'] == digest(store.plan) and receipt['bindings'] == store.inputs(), 'Preparation binding drift')
    required = {'prepared/stage1.npz', 'prepared/stage2.npz', 'prepared/validation.npz',
                'prepared/preprocessor/preprocessor.joblib', 'prepared/preprocessor/preprocessor_meta.json'}
    require(set(receipt['files']) == required, 'Incomplete preparation')
    require(array in (None, 'stage1', 'stage2', 'validation'), 'Invalid preparation array')
    names = required if array is None else {f'prepared/{array}.npz'}
    for name in names:
        require(file_hash(safe_artifact(store.path, name)) == receipt['files'][name], 'Preparation artifact drift')
    return receipt
