"""Frozen full-count model fits and independent validation-only score checkpoints."""
import json
import time
import warnings
import numpy as np

from .contract import SCOPE, COMPONENTS
from . import preparation
from experiments.c03.producer import models as native, validation as v
from experiments.c03.producer.artifacts import array_hash, tensor_hash, safe_artifact, write_arrays
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.plan import canonical, digest, file_hash, require
from experiments.c03.engineering import lifecycle, xgb_full_row_sizing_probe as xcheck


def imported(store, name, phase):
    require(name in COMPONENTS and phase in ('fit', 'score'), 'Invalid model phase')
    key = store.plan['graph']['components'][name]['stage'] if phase == 'fit' else 'validation'
    receipt = preparation.verify(store, array=key)
    return {'scope': SCOPE, 'scientific_checkpoint': True, 'component': name, 'phase': phase,
        'plan_sha256': digest(store.plan), 'prepared_receipt_sha256': file_hash(store.path/'prepared/complete.json'),
        'array_path': f'prepared/{key}.npz', 'array_sha256': receipt['files'][f'prepared/{key}.npz'],
        'expected': receipt['bindings'][key] if phase == 'fit' else receipt['validation']}


def check_model(model, node, name, *, device='cuda:0', columns=46):
    expected_classes = list(range(2 if node['stage'] == 'stage1' else 5))
    require(model.classes_.tolist() == expected_classes and model.n_features_in_ == columns, 'Model class/feature drift')
    count = node['parameters']['n_estimators']
    if node['model_family'] == 'rf':
        require(len(model.estimators_) == count and model.n_jobs == 2, 'RF count/thread mismatch')
        return {'trees': count, 'device': 'cpu', 'classes': expected_classes}
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    xcheck.cuda.check_booster(config, name, device=device)
    require(booster.num_boosted_rounds() == count, 'XGBoost round count differs')
    trees = int(config['learner']['gradient_booster']['gbtree_model_param']['num_trees'])
    require(trees == count*(1 if node['stage'] == 'stage1' else 5), 'XGBoost tree count differs')
    return {'trees': trees, 'rounds': count, 'device': device, 'classes': expected_classes, 'config': config}


def fit(store, directory, name, record, X, y, weight, emit, *, device='cuda:0'):
    node = store.plan['graph']['components'][name]
    _, grid, _ = v.historical()
    # Only private generated tests supply CPU XGBoost plans. CLI uses frozen_plan.
    model = native.construct(grid, node)
    params = native.json_parameters(model.get_params())
    require(params['n_estimators'] == node['parameters']['n_estimators'] and params['n_jobs'] == 2
            and params['random_state'] == node['seed'], 'Effective fit definition differs')
    before = [tensor_hash(X), array_hash(y), digest(None) if weight is None else array_hash(weight)]
    expected = record['expected']
    require(before == [expected['tensor'], expected['labels'], expected['weights'][name]], 'Training identity differs')
    if node['model_family'] == 'xgb':
        import xgboost
        class Reporter(xgboost.callback.TrainingCallback):
            def after_iteration(self, booster, epoch, evals_log):
                xcheck.check_booster(json.loads(booster.save_config()), name, device=device)
                if (epoch+1) % 20 == 0 or epoch+1 == node['parameters']['n_estimators']:
                    emit(f'round_{epoch+1}_of_{node["parameters"]["n_estimators"]}')
                return False
        model.set_params(callbacks=[Reporter()])
    emit('fit_full_frozen_ensemble')
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(X, y, **({} if weight is None else {'sample_weight': weight}))
    require(not caught, 'Fit warnings: '+repr([str(x.message) for x in caught]))
    elapsed = time.monotonic()-start
    info = check_model(model, node, name, device=device, columns=X.shape[1])
    require(before == [tensor_hash(X), array_hash(y), digest(None) if weight is None else array_hash(weight)], 'Fit mutated inputs')
    if node['model_family'] == 'xgb':
        xcheck.check_booster(info['config'], name, device=device)
        model.set_params(callbacks=None)
    emit('save_full_model')
    path = directory/('model.joblib' if node['model_family'] == 'rf' else 'model.ubj')
    native.save_local_model(model, path, node['model_family'])
    return {'model_family': node['model_family'], 'effective_parameters': params, 'fitted': info,
        'input_hashes': before, 'rows': X.shape[0], 'features': X.shape[1], 'fit_seconds': elapsed,
        'model_file': path.name, 'model_sha256': file_hash(path), 'model_bytes': path.stat().st_size}


def score(store, directory, fit_dir, fit_record, name, X, emit, *, device='cuda:0'):
    node = store.plan['graph']['components'][name]
    path = safe_artifact(fit_dir, fit_record['model_file'])
    require(file_hash(path) == fit_record['model_sha256'], 'Model checksum drift before load')
    model = native.load_verified_local_model(path, node['model_family'])
    if node['model_family'] == 'xgb':
        model.set_params(device=device, n_jobs=2, tree_method='hist')
    check_model(model, node, name, device=device, columns=X.shape[1])
    before = tensor_hash(X)
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        probabilities = lifecycle.score_blocks(model, X, name, emit)
    require(not caught, 'Prediction warnings: '+repr([str(x.message) for x in caught]))
    info = check_model(model, node, name, device=device, columns=X.shape[1])
    require(tensor_hash(X) == before and file_hash(path) == fit_record['model_sha256'], 'Scoring input/model drift')
    write_arrays(directory/'validation.npz', probabilities=probabilities)
    return {'model_family': node['model_family'], 'model_sha256': fit_record['model_sha256'],
        'inference': info, 'rows': len(probabilities), 'shape': list(probabilities.shape),
        'dtype': str(probabilities.dtype), 'probabilities_sha256': array_hash(probabilities),
        'score_file': 'validation.npz', 'score_file_sha256': file_hash(directory/'validation.npz'),
        'score_seconds': time.monotonic()-start, 'quality_metrics': None}


def commit(store, directory, name, phase, record, result, fit_sha=None):
    require(imported(store, name, phase) == record, 'Inputs changed before checkpoint')
    publish(directory/'complete.json', canonical({'scope': SCOPE, 'schema_version': 1, 'phase': phase,
        'component': name, 'plan_sha256': digest(store.plan), 'import': record, 'result': result,
        'fit_receipt_sha256': fit_sha}))


def verify(store, directory, name, phase, fit_dir=None):
    record = store.read_json(directory/'complete.json')
    require(record['scope'] == SCOPE and record['component'] == name and record['phase'] == phase
            and record['schema_version'] == 1 and record['plan_sha256'] == digest(store.plan), 'Checkpoint scope/binding drift')
    require(record['import'] == imported(store, name, phase), 'Checkpoint import drift')
    node, result = store.plan['graph']['components'][name], record['result']
    require(result['model_family'] == node['model_family'], 'Model family differs')
    if phase == 'fit':
        expected = record['import']['expected']
        require(record['fit_receipt_sha256'] is None and result['rows'] == expected['rows']
                and result['input_hashes'] == [expected['tensor'], expected['labels'], expected['weights'][name]], 'Fit identity differs')
        params = result['effective_parameters']
        _, grid, _ = v.historical()
        require(params == native.json_parameters(native.construct(grid, node).get_params()), 'Effective parameter drift')
        info = result['fitted']
        count = node['parameters']['n_estimators']
        require(info['classes'] == list(range(2 if node['stage'] == 'stage1' else 5)), 'Fitted classes differ')
        require(info['trees'] == count*(5 if node['model_family'] == 'xgb' and node['stage'] == 'stage2' else 1), 'Tree count differs')
        if node['model_family'] == 'xgb':
            require(info['rounds'] == count and info['device'] == node['parameters']['device'], 'GPU/round definition differs')
            xcheck.check_booster(info['config'], name, device=node['parameters']['device'])
        filename = 'model.joblib' if node['model_family'] == 'rf' else 'model.ubj'
        require(result['model_file'] == filename, 'Foreign model format')
        path = safe_artifact(directory, filename)
        require(file_hash(path) == result['model_sha256'] and path.stat().st_size == result['model_bytes'], 'Model checksum drift')
    else:
        require(fit_dir is not None and record['fit_receipt_sha256'] == file_hash(fit_dir/'complete.json'), 'Fit receipt changed')
        fit_record = verify(store, fit_dir, name, 'fit')['result']
        require(result['model_sha256'] == fit_record['model_sha256'], 'Scored model differs')
        count = record['import']['expected']['count']
        info = result['inference']
        require(info['classes'] == fit_record['fitted']['classes']
                and info['trees'] == fit_record['fitted']['trees'], 'Inference ensemble differs')
        if node['model_family'] == 'xgb':
            require(info['rounds'] == node['parameters']['n_estimators']
                    and info['device'] == node['parameters']['device'], 'Inference device/rounds differ')
            xcheck.check_booster(info['config'], name, device=node['parameters']['device'])
        require(result['rows'] == count and result['shape'] == ([count] if node['stage'] == 'stage1' else [count, 5])
                and result['dtype'] == 'float64' and result['quality_metrics'] is None, 'Score definition differs')
        require(result['score_file'] == 'validation.npz'
                and file_hash(safe_artifact(directory, 'validation.npz')) == result['score_file_sha256'], 'Score checksum drift')
    return record
