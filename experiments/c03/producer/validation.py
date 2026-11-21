"""Historical preparation and synthetic-only model/validation-score production.

Engineering TRAIN/validation preparation is separately scoped; no test reader.
Model constructors,
preprocessing and score transforms are shared with the pinned historical code.
"""
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

from . import models
from .transforms import BLOCK_ROWS, transform_selected
from .artifacts import (array_hash, tensor_hash, identity_hash, write_arrays,
                        read_arrays, safe_artifact)
from .checkpoints import ENGINEERING_SCOPE, publish
from .plan import TOOLS, canonical, digest, file_hash, require

FAMILIES = ('BruteForce', 'DDoS', 'DoS', 'Other', 'Scan/Recon')


def historical():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import c03_paired_runner_core as core
    grid, engine = core.history_modules()
    return core, grid, engine


def fixture_gate(store, sources):
    require(store.plan.get('evidence_scope') == 'SYNTHETIC_ONLY' and
            store.plan.get('status') == 'SYNTHETIC_VALIDATION_PLAN' and store.plan.get('execution_authorized') is False,
            'Benchmark fitting is not authorized; only synthetic fixtures may execute')
    from .plan import ROOT
    require(not sources.root.is_relative_to(ROOT), 'Fixtures must be outside repository/data roots')
    for stage in ('train', 'val'):
        require(0 < store.plan['sampling'][f'{stage}_cap'] <= 10000, 'Synthetic row cap exceeded')
        count = sum(pq.ParquetFile(sources.path(part)).metadata.num_rows for part in sources.parts[stage])
        require(0 < count <= 10000, 'Synthetic source limit exceeded')
    for node in store.plan['graph']['components'].values():
        params = node['parameters']
        require(1 <= params['n_estimators'] <= 8 and params.get('n_jobs') == 1,
                'Synthetic model size/worker limit exceeded')
        if node['model_family'] == 'xgb':
            require(params.get('device') == 'cpu', 'Synthetic test requires explicit CPU')


def truth_labels(frame):
    y1 = frame['y_stage1_attack'].to_numpy()
    y2 = frame['y_stage2_family'].astype(str).to_numpy()
    require(np.isin(y1, [0, 1]).all(), 'Nonbinary Stage-1 truth')
    require(np.isin(y2[y1 == 1], (*FAMILIES, 'Botnet')).all(), 'Unexpected attack family')
    result = np.zeros(len(frame), dtype=np.int8)
    for i, name in enumerate(FAMILIES, 1):
        result[(y1 == 1) & (y2 == name)] = i
    result[(y1 == 1) & (y2 == 'Botnet')] = 6
    return result


@dataclass
class Prepared:
    tensors: dict
    targets: dict
    row_ids: dict
    weights: dict
    validation: csr_matrix
    truth: np.ndarray
    validation_ids: object
    preprocessor_sha: str
    receipt_sha: str = ''


@dataclass
class ComponentPrepared(Prepared):
    component: str = ''


def input_bindings(data, plan):
    return {'preprocessor': data.preprocessor_sha, **{
        stage: {'rows': len(data.targets[stage]), 'tensor': tensor_hash(data.tensors[stage]),
                'labels': array_hash(data.targets[stage]), 'row_ids': identity_hash(data.row_ids[stage]),
                'weights': {name: digest(None) if data.weights[name] is None else array_hash(data.weights[name])
                            for name, node in plan['graph']['components'].items() if node['stage'] == stage}}
        for stage in ('stage1', 'stage2')}}


def validation_identity(data):
    return {'tensor': tensor_hash(data.validation), 'truth': array_hash(data.truth),
            'rows': identity_hash(data.validation_ids), 'count': len(data.truth)}


def save_tensor(path, tensor, target, ids, weights):
    arrays = {'data': tensor.data, 'indices': tensor.indices, 'indptr': tensor.indptr,
              'shape': np.array(tensor.shape, dtype=np.int64), 'target': target,
              'source_part': ids.source_part.to_numpy(dtype=str),
              'source_sha256': ids.source_sha256.to_numpy(dtype=str),
              'row_offset': ids.row_offset.to_numpy(dtype=np.int64)}
    arrays.update({f'weight_{k}': v for k, v in weights.items() if v is not None})
    write_arrays(path, **arrays)


def prepare(store, sources):
    """Fit actual historical preprocessing on fixture train only; sample and persist."""
    fixture_gate(store, sources)
    return _prepare(store, sources, scope='SYNTHETIC_ONLY')


def prepare_engineering(store, sources, emit):
    """Preparation only. This scope cannot fit components, select or evaluate."""
    require(store.plan.get('status') == 'ENGINEERING_PREPARATION_PLAN' and
            store.plan.get('evidence_scope') == ENGINEERING_SCOPE and
            store.plan.get('execution_authorized') is False, 'Invalid engineering preparation scope')
    return _prepare(store, sources, scope=ENGINEERING_SCOPE, emit=emit)


def _prepare(store, sources, *, scope, emit=lambda _: None):
    require(scope in ('SYNTHETIC_ONLY', ENGINEERING_SCOPE) and
            store.plan.get('evidence_scope') == scope, 'Preparation scope mismatch')
    emit('verify_sources')
    sources.verify()
    from .sealing import current_code
    current_code(store.plan)
    core, grid, engine = historical()
    with store.lock():
        store.verify_plan()
        directory = store.path/'prepared'
        directory.mkdir(exist_ok=False)
        preprocessing = directory/'preprocessor'
        preprocessing.mkdir()
        emit('preprocessor')
        prep = engine.fit_preprocessor(str(sources.root), str(preprocessing))
        prep_sha = file_hash(preprocessing/'preprocessor.joblib')
        sampler = sources.sampler()
        emit('sample_train')
        train = sampler.sample('train', store.plan['sampling']['train_cap'], store.plan['sampling']['train_seed'])
        emit('sample_validation')
        val = sampler.sample('val', store.plan['sampling']['val_cap'], store.plan['sampling']['val_seed'])
        yt, yv = truth_labels(train.frame), truth_labels(val.frame)
        for label in range(7):
            require(int((yv == label).sum()) >= 200, 'Insufficient validation class support')
        for label in range(6):
            require(int((yt == label).sum()) >= 200, 'Insufficient supervised training class support')
        masks = {'stage1': yt != 6, 'stage2': (yt >= 1) & (yt <= 5)}
        require(set(yt[masks['stage2']]) == set(range(1, 6)), 'All represented families required')
        tensors, targets, ids, weights = {}, {}, {}, {}
        for stage, mask in masks.items():
            emit(stage+'_tensor')
            tensors[stage] = transform_selected(prep, train.frame, mask)
            targets[stage] = ((yt[mask] != 0).astype(int) if stage == 'stage1' else (yt[mask]-1).astype(int))
            ids[stage] = train.identities.loc[mask].reset_index(drop=True)
            for name, node in store.plan['graph']['components'].items():
                if node['stage'] != stage:
                    continue
                weights[name] = (grid.make_stage1_sample_weights(
                    targets[stage], train.frame.loc[mask, 'y_stage2_family'].astype(str).to_numpy(dtype=object),
                    node['weight_mode'], float(grid.CFG['inv_family_clip_max'])) if stage == 'stage1'
                    else grid.stage2_sample_weights(targets[stage], 5, node['weight_mode']))
        emit('validation_tensor')
        Xv = transform_selected(prep, val.frame, np.ones(len(val.frame), dtype=bool))
        data = Prepared(tensors, targets, ids, weights, Xv, yv, val.identities, prep_sha)
        emit('input_identity')
        binding = input_bindings(data, store.plan)
        store.validate_inputs(binding)
        expected = store.plan['expected_support']
        require((int((yv != 6).sum()), int((yv == 6).sum())) ==
                (expected['validation_K'], expected['validation_U']), 'Validation support drift')
        for stage in masks:
            emit('serialize_'+stage)
            save_tensor(directory/f'{stage}.npz', tensors[stage], targets[stage], ids[stage],
                        {k: v for k, v in weights.items() if store.plan['graph']['components'][k]['stage'] == stage})
        emit('serialize_validation')
        save_tensor(directory/'validation.npz', Xv, yv, val.identities, {})
        emit('verify_and_receipt')
        sources.verify()
        # Reconstruct the historical Parquet statistics-prefix row selection.
        # This records identities only; it does not replace preprocessing.
        prefix, seen = [], 0
        for part in sources.parts['train']:
            rows = pq.ParquetFile(sources.path(part)).metadata.num_rows
            prefix.append({'source_part': part.name, 'rows': rows})
            seen += rows
            if seen >= engine.CFG['max_stat_rows']:
                break
        stat_rows = (np.random.RandomState(engine.CFG['global_seed']).choice(
            seen, size=engine.CFG['max_stat_rows'], replace=False)
            if seen > engine.CFG['max_stat_rows'] else np.arange(seen))
        receipt = {'scope': scope, 'plan_sha256': digest(store.plan),
                   'transform_execution': {'rows_per_block': BLOCK_ROWS, 'passes': 2,
                                           'historical_transform_unchanged': True},
                   'source_inventory': sources.inventory(), 'bindings': binding,
                   'validation': validation_identity(data),
                   'preprocessor_metadata': json.loads((preprocessing/'preprocessor_meta.json').read_text()),
                   'preprocessor_statistics': {'ordered_prefix': prefix,
                       'selected_prefix_positions_sha256': array_hash(stat_rows),
                       'selected_rows': len(stat_rows),
                       'inference_part': sources.parts['train'][0].name,
                       'inference_max_rows': 5000},
                   'helper_config': models.json_parameters(engine.CFG),
                   'files': {p.relative_to(store.path).as_posix(): file_hash(p)
                             for p in sorted(directory.rglob('*')) if p.is_file()}}
        # Publish the binding first; the preparation receipt is the final commit
        # marker, so a completed preparation can never lack its input binding.
        publish(store.path/'inputs.json', canonical({'plan_sha256': digest(store.plan),
            'bindings': binding, 'bindings_sha256': digest(binding)}))
        publish(directory/'complete.json', canonical(receipt))
        data.receipt_sha = digest(receipt)
    return data


def verify_prepared(store):
    store.verify_plan()
    receipt = store.read_json(store.path/'prepared/complete.json')
    require(receipt['scope'] == store.plan.get('evidence_scope') and
            receipt['scope'] in ('SYNTHETIC_ONLY', ENGINEERING_SCOPE) and
            receipt['plan_sha256'] == digest(store.plan) and receipt['bindings'] == store.inputs(),
            'Prepared input/plan drift')
    required = {'prepared/stage1.npz', 'prepared/stage2.npz', 'prepared/validation.npz',
                'prepared/preprocessor/preprocessor.joblib', 'prepared/preprocessor/preprocessor_meta.json'}
    require(set(receipt['files']) == required, 'Incomplete prepared artifacts')
    for name, sha in receipt['files'].items():
        require(file_hash(safe_artifact(store.path, name)) == sha, 'Prepared artifact drift')
    return receipt


def load_prepared(store):
    """Resume arrays only after file receipt verification; no pickle load."""
    import pandas as pd
    receipt = verify_prepared(store)
    tensors, targets, ids, weights = {}, {}, {}, {}
    for stage in ('stage1', 'stage2', 'validation'):
        values = read_arrays(store.path/f'prepared/{stage}.npz')
        tensors[stage] = csr_matrix((values['data'], values['indices'], values['indptr']), shape=tuple(values['shape']))
        targets[stage] = values['target']
        ids[stage] = pd.DataFrame({k: values[k] for k in ('source_part', 'source_sha256', 'row_offset')})
        for name, node in store.plan['graph']['components'].items():
            if node['stage'] == stage:
                weights[name] = values.get('weight_'+name)
    data = Prepared({k: tensors[k] for k in ('stage1', 'stage2')},
                    {k: targets[k] for k in ('stage1', 'stage2')},
                    {k: ids[k] for k in ('stage1', 'stage2')}, weights,
                    tensors['validation'], targets['validation'], ids['validation'],
                    receipt['bindings']['preprocessor'], digest(receipt))
    require(input_bindings(data, store.plan) == store.inputs() and validation_identity(data) == receipt['validation'],
            'Prepared array identity differs')
    return data


def component_record(store, name):
    payload = store.verify_component(name)
    record = json.loads(payload.read_text(encoding='utf-8'))
    receipt = verify_prepared(store)
    require(record['prepared_sha256'] == digest(receipt) and
            record['validation'] == receipt['validation'], 'Score provenance drift')
    family = store.plan['graph']['components'][name]['model_family']
    require(record['model_family'] == family and record['scope'] == 'SYNTHETIC_ONLY', 'Model scope mismatch')
    expected = {f'{name}/model.'+('ubj' if family == 'xgb' else 'joblib'), f'{name}/validation.npz'}
    require(set(record['files']) == expected, 'Missing/foreign model or score file')
    for path, sha in record['files'].items():
        require(file_hash(safe_artifact(store.path, path)) == sha, 'Model/score checksum drift')
    return record


def load_component_prepared(store, name):
    """Load only this component's training stage plus common validation arrays."""
    import pandas as pd
    require(name in store.plan['graph']['components'], 'Unknown component')
    receipt = verify_prepared(store)
    stage = store.plan['graph']['components'][name]['stage']
    tensors, targets, ids = {}, {}, {}
    weight = None
    for key in (stage, 'validation'):
        values = read_arrays(store.path/f'prepared/{key}.npz')
        tensors[key] = csr_matrix((values['data'], values['indices'], values['indptr']), shape=tuple(values['shape']))
        targets[key] = values['target']
        ids[key] = pd.DataFrame({k: values[k] for k in ('source_part', 'source_sha256', 'row_offset')})
        if key == stage:
            weight = values.get('weight_'+name)
    data = ComponentPrepared({stage: tensors[stage]}, {stage: targets[stage]}, {stage: ids[stage]},
        {name: weight}, tensors['validation'], targets['validation'], ids['validation'],
        receipt['bindings']['preprocessor'], digest(receipt), name)
    verify_live_inputs(store, name, data, receipt)
    return data


def verify_live_inputs(store, name, data, receipt):
    require(data.receipt_sha == digest(receipt) and validation_identity(data) == receipt['validation'],
            'Live input identity drift')
    if isinstance(data, ComponentPrepared):
        stage = store.plan['graph']['components'][name]['stage']
        expected = store.inputs()[stage]
        require(data.component == name and set(data.tensors) == {stage} and
                data.preprocessor_sha == receipt['bindings']['preprocessor'] and
                tensor_hash(data.tensors[stage]) == expected['tensor'] and
                array_hash(data.targets[stage]) == expected['labels'] and
                identity_hash(data.row_ids[stage]) == expected['row_ids'] and
                (digest(None) if data.weights[name] is None else array_hash(data.weights[name])) == expected['weights'][name],
                'Component input identity drift')
    else:
        require(input_bindings(data, store.plan) == store.inputs(), 'Live input identity drift')


def fit_component(store, name, data):
    with store.component_lock(name):
        return _fit_component(store, name, data)


def _fit_component(store, name, data):
    require(store.plan.get('evidence_scope') == 'SYNTHETIC_ONLY' and
            store.plan.get('status') == 'SYNTHETIC_VALIDATION_PLAN', 'Full fit not authorized')
    from .sealing import current_code
    current_code(store.plan)
    receipt = verify_prepared(store)
    verify_live_inputs(store, name, data, receipt)
    if store.component_path(name).exists():
        component_record(store, name)  # Missing receipt means interrupted, not reusable.
        return False
    _, grid, _ = historical()
    node = store.plan['graph']['components'][name]
    store.begin(name)
    model = models.construct(grid, node)
    params = models.json_parameters(model.get_params())
    stage = node['stage']
    weight = data.weights[name]
    model.fit(data.tensors[stage], data.targets[stage], **({} if weight is None else {'sample_weight': weight}))
    probabilities = models.probabilities(model, data.validation, stage)
    verify_live_inputs(store, name, data, receipt)
    directory = store.component_path(name)
    model_path = directory/('model.ubj' if node['model_family'] == 'xgb' else 'model.joblib')
    models.save_local_model(model, model_path, node['model_family'])
    write_arrays(directory/'validation.npz', probabilities=probabilities)
    record = {'scope': 'SYNTHETIC_ONLY', 'model_family': node['model_family'],
              'prepared_sha256': data.receipt_sha, 'validation': validation_identity(data),
              'effective_parameters': params, 'fitted_classes': np.asarray(model.classes_).tolist(),
              'probability_order': [0, 1] if stage == 'stage1' else list(FAMILIES),
              'files': {p.relative_to(store.path).as_posix(): file_hash(p)
                        for p in (model_path, directory/'validation.npz')}}
    store.complete(name, canonical(record))
    del model
    gc.collect()
    return True


def verified_model(store, name):
    """Only this checked entrypoint is used for locally produced model reload."""
    record = component_record(store, name)
    family = record['model_family']
    path = safe_artifact(store.path, f'{name}/model.'+('ubj' if family == 'xgb' else 'joblib'))
    return models.load_verified_local_model(path, family)


def validation_bundle(store):
    core, _, _ = historical()
    receipt = verify_prepared(store)
    validation = read_arrays(store.path/'prepared/validation.npz')
    truth = validation['target']
    scores = {}
    cached = {}
    for name in store.plan['graph']['components']:
        component_record(store, name)
        cached[name] = read_arrays(store.path/f'{name}/validation.npz')['probabilities']
    for name, refs in store.plan['graph']['candidate_references'].items():
        scores[name] = core.CandidateScores(cached[refs['stage1']], cached[refs['stage2']])
    counts = store.plan['expected_support']
    core.validate_prediction_bundle(truth, scores, expected_counts=(counts['validation_K'], counts['validation_U']))
    require(array_hash(truth) == receipt['validation']['truth'], 'Truth identity drift')
    return truth, scores


def produce_validation(store, sources, *, resume=False):
    """Sequential prepare/fit/score/seal lifecycle, bounded to synthetic data.

    Existing incomplete artifacts fail closed. No automatic retry or replacement.
    A fully complete run may be verified with resume=True without fitting again.
    """
    from .sealing import select_and_seal, verify_seal, current_code
    current_code(store.plan)
    fixture_gate(store, sources)
    sources.verify()
    with store.pipeline_lock():
        if (store.path/'prepared').exists():
            require(resume, 'Explicit resume required for existing preparation')
            receipt = verify_prepared(store)
            require(receipt['source_inventory'] == sources.inventory(), 'Resume source inventory drift')
            data = load_prepared(store)
        else:
            data = prepare(store, sources)
        for name in store.plan['graph']['fit_order']:
            fit_component(store, name, data)
        if (store.path/'selection.json').exists():
            require(resume, 'Explicit resume required for existing selection')
            return verify_seal(store)
        return select_and_seal(store)
