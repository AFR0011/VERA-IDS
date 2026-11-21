"""Versioned engineering fit/score receipts. Never selects or evaluates test data."""
import copy
import json
from pathlib import Path
import time
import warnings

from . import rf_stage2_sizing_probe as prep
from . import xgb_full_row_sizing_probe as xprobe
from . import resource_policy
from experiments.c03.producer import models
from experiments.c03.producer.artifacts import safe_artifact, read_arrays, write_arrays
from experiments.c03.producer.checkpoints import ENGINEERING_SCOPE, publish
from experiments.c03.producer.plan import ROOT, build_plan, canonical, digest, file_hash, require

SCOPE = 'ENGINEERING_FIT_RELOAD_VALIDATION_REHEARSAL_ONLY'
COMPONENTS = ('stage1_rf_class_weight_balanced', 'stage1_rf_inv_family_clipped',
              'stage1_xgb_inv_family_clipped', 'stage2_rf_shared', 'stage2_xgb_shared')
BLOCK_ROWS = 8192
DEFINITION = {'components': list(COMPONENTS), 'n_estimators': 8, 'n_jobs': 2,
              'device': 'cuda:0', 'validation_rows': 700000, 'block_rows': BLOCK_ROWS,
              'features': 46, 'stage1_rows': 1270025, 'stage2_rows': 1201422,
              'seeds': [123, 133], 'selection_authorized': False, 'test_authorized': False}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_record():
    return {'prior': xprobe.source_record(), 'sources': {
        p.relative_to(ROOT).as_posix(): file_hash(p) for p in (
            Path(__file__), Path(__file__).with_name('lifecycle_probe.py'),
            Path(resource_policy.__file__),
            ROOT/'tests/test_c03_lifecycle.py', ROOT/'tests/test_c03_lifecycle_probe.py')}}


def request_base():
    return {'schema_version': 1, 'scope': SCOPE, 'scientific_checkpoint': False,
            'definition': copy.deepcopy(DEFINITION), 'limits': dict(resource_policy.LIMITS), 'source_record': source_record()}


def verify_import(source, success, plan, name, phase):
    """Only the selected array's bytes are read; other evidence is compact metadata."""
    require(name in COMPONENTS and phase in ('fit', 'score'), 'Unknown lifecycle phase')
    require(success['scope'] == ENGINEERING_SCOPE and
            success['status'] == 'VERIFIED_COMBINED_PREPARATION_COMPLETE', 'Invalid preparation scope')
    for path, sha in success['evidence_hashes'].items():
        require(file_hash(safe_artifact(source, path)) == sha, 'Original evidence drift')
    request, result, completion = [read(source/p) for p in ('request.json', 'worker-result.json', 'complete.json')]
    require(completion['scope'] == ENGINEERING_SCOPE and completion['scientific_checkpoint'] is False
            and read(source/'supervisor.json')['status'] == 'COMPLETE' and result['reload_verified'] is True,
            'Incomplete source preparation')
    for path, key in [('request.json', 'request_sha256'), ('worker-result.json', 'worker_result_sha256'),
                      ('supervisor.json', 'supervisor_sha256')]:
        require(completion[key] == file_hash(source/path), 'Source completion drift')
    old = request['plan']
    require(digest(old) == success['source_record']['plan_sha256'] == result['plan_sha256']
            and request['source_record'] == success['source_record'], 'Original plan drift')
    for key in ('environment', 'policy_sha256', 'original_snapshot_sha256', 'sampling', 'expected_support', 'graph'):
        require(old[key] == plan[key], 'Import policy/environment/model definition differs')
    for path in ('src/ids_eval_framework/_native/protocol_b_grid.py', 'src/ids_eval_framework/src/two_stage_engine.py'):
        require(old['sources'][path] == plan['sources'][path], 'Historical helper changed')
    receipt_path = safe_artifact(source, 'diagnostic/prepared/complete.json')
    require(file_hash(receipt_path) == result['files']['diagnostic/prepared/complete.json'], 'Preparation receipt drift')
    receipt = read(receipt_path)
    require(receipt['scope'] == ENGINEERING_SCOPE and receipt['plan_sha256'] == digest(old)
            and receipt['bindings'] == success['identities']['bindings'] == result['identities']['bindings']
            and receipt['validation'] == success['identities']['validation'] == result['identities']['validation'],
            'Preparation identities differ')
    key = plan['graph']['components'][name]['stage'] if phase == 'fit' else 'validation'
    array = f'diagnostic/prepared/{key}.npz'
    sha = result['files'][array]
    require(sha == receipt['files'][f'prepared/{key}.npz'] and file_hash(safe_artifact(source, array)) == sha,
            'Imported array byte drift')
    expected = receipt['bindings'][key] if phase == 'fit' else receipt['validation']
    rows = expected['rows'] if phase == 'fit' else expected['count']
    require(rows == DEFINITION[key+'_rows'], 'Approved row count differs')
    return {'scope': SCOPE, 'scientific_checkpoint': False, 'component': name, 'phase': phase,
            'source_run_id': success['run_id'], 'source_plan_sha256': digest(old),
            'current_plan_sha256': digest(plan), 'approved_success_sha256': prep.SUCCESS_SHA,
            'source_receipt_sha256': file_hash(receipt_path), 'preprocessor': receipt['bindings']['preprocessor'],
            'array_path': array, 'array_sha256': sha, 'expected': expected}


def load_input(path, imported, name, phase, *, columns=46):
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from experiments.c03.producer.artifacts import tensor_hash, array_hash, identity_hash
    require(imported['component'] == name and imported['phase'] == phase, 'Wrong imported component/phase')
    require(file_hash(path) == imported['array_sha256'], 'Array changed before load')
    values = read_arrays(path)
    X = csr_matrix((values['data'], values['indices'], values['indptr']), shape=tuple(values['shape']))
    y = values['target']
    ids = pd.DataFrame({k: values[k] for k in ('source_part', 'source_sha256', 'row_offset')})
    expected = imported['expected']
    count = expected['rows'] if phase == 'fit' else expected['count']
    require(X.shape == (count, columns) and X.dtype == np.dtype('float64'), 'Input shape/dtype differs')
    require(tensor_hash(X) == expected['tensor'] and array_hash(y) == expected['labels' if phase == 'fit' else 'truth']
            and identity_hash(ids) == expected['row_ids' if phase == 'fit' else 'rows'], 'Input identity drift')
    weight = values.get('weight_'+name) if phase == 'fit' else None
    if phase == 'fit':
        require(set(np.unique(y)) == ({0, 1} if name.startswith('stage1') else set(range(5))), 'Training support differs')
        require((digest(None) if weight is None else array_hash(weight)) == expected['weights'][name], 'Weight identity drift')
        require((weight is None) == (name == COMPONENTS[0]), 'Wrong sample weighting path')
        if weight is not None:
            require(weight.shape == y.shape and np.isfinite(weight).all() and (weight > 0).all(), 'Invalid weights')
        if name.startswith('stage2'):
            from experiments.c03.producer.validation import historical
            _, grid, _ = historical()
            require(array_hash(grid.stage2_sample_weights(y, 5, 'balanced')) == array_hash(weight), 'Stage-2 weights differ')
    else:
        require(set(np.unique(y)) == set(range(7)), 'Validation support differs')
    require(file_hash(path) == imported['array_sha256'], 'Array changed during load')
    return X, y, weight


def construct(plan, name, *, device='cuda:0'):
    from experiments.c03.producer.validation import historical
    require(name in COMPONENTS and device in ('cpu', 'cuda:0'), 'Invalid model request')
    node = copy.deepcopy(plan['graph']['components'][name])
    _, grid, _ = historical()
    original = models.json_parameters(models.construct(grid, node).get_params())
    override = {'n_estimators': 8, 'n_jobs': 2}
    if node['model_family'] == 'xgb':
        override['device'] = device
    node['parameters'].update(override)
    model = models.construct(grid, node)
    parameters = models.json_parameters(model.get_params())
    require(parameters == {**original, **override}, 'Unapproved estimator parameter change')
    return model, parameters


def check_model(model, name, *, device='cuda:0', columns=46, fitted=False):
    import numpy as np
    expected = np.arange(2 if name.startswith('stage1') else 5)
    require(np.array_equal(model.classes_, expected), 'Fitted class order differs')
    require(model.n_features_in_ == columns, 'Fitted feature count differs')
    if '_rf_' in name:
        require(len(model.estimators_) == 8 and model.n_jobs == 2, 'RF execution definition differs')
        return {'trees': 8, 'device': 'cpu'}
    booster = model.get_booster()
    config = read_config(model)
    xprobe.cuda.check_booster(config, name, device=device)
    if fitted:
        xprobe.check_booster(config, name, device=device)
    require(booster.num_boosted_rounds() == 8, 'Wrong boosting round count')
    trees = int(config['learner']['gradient_booster']['gbtree_model_param']['num_trees'])
    require(trees == (8 if name.startswith('stage1') else 40), 'Wrong XGBoost tree count')
    return {'trees': trees, 'rounds': 8, 'device': device, 'config': config}


def read_config(model):
    return json.loads(model.get_booster().save_config())


def fit_save(directory, plan, name, imported, X, y, weight, emit, *, device='cuda:0'):
    from experiments.c03.producer.artifacts import tensor_hash, array_hash
    model, params = construct(plan, name, device=device)
    family = plan['graph']['components'][name]['model_family']
    before = [tensor_hash(X), array_hash(y), digest(None) if weight is None else array_hash(weight)]
    if family == 'xgb':
        import xgboost
        class Reporter(xgboost.callback.TrainingCallback):
            def after_iteration(self, booster, epoch, evals_log):
                xprobe.check_booster(json.loads(booster.save_config()), name, device=device)
                emit(f'round_{epoch+1}_of_8')
                return False
        model.set_params(callbacks=[Reporter()])
    emit('fitting')
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(X, y, **({} if weight is None else {'sample_weight': weight}))
    elapsed = time.monotonic()-started
    require(not caught, 'Fit warnings: '+repr([str(w.message) for w in caught]))
    info = check_model(model, name, device=device, columns=X.shape[1], fitted=True)
    if family == 'xgb':
        model.set_params(callbacks=None)  # Progress observer is never part of the saved estimator.
    require(before == [tensor_hash(X), array_hash(y), digest(None) if weight is None else array_hash(weight)], 'Fit mutated inputs')
    emit('saving_model')
    path = directory/('model.ubj' if family == 'xgb' else 'model.joblib')
    models.save_local_model(model, path, family)
    return {'model_family': family, 'effective_parameters': params, 'fitted': info,
            'fitted_classes': model.classes_.tolist(),
            'input_hashes': before, 'fit_seconds': elapsed, 'rows': X.shape[0], 'features': X.shape[1],
            'model_file': path.name, 'model_sha256': file_hash(path), 'model_bytes': path.stat().st_size,
            'import_sha256': digest(imported)}


class DMatrixProbabilities:
    """Explicit XGBoost DMatrix inference; same logistic/softprob class expansion.

    Avoid a CPU-CSR / CUDA inplace-predict mismatch and its implicit DMatrix
    fallback. The configured booster stays on its approved device. See the
    installed sklearn.XGBModel.predict and XGBClassifier.predict_proba paths.
    """
    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_

    def predict_proba(self, X):
        import numpy as np
        import xgboost
        matrix = xgboost.DMatrix(X, missing=self.model.missing, nthread=2,
                                 feature_types=self.model.feature_types,
                                 enable_categorical=self.model.enable_categorical)
        raw = self.model.get_booster().predict(matrix, iteration_range=(0, 0), validate_features=True)
        return np.vstack((1-raw, raw)).T if len(self.classes_) == 2 else raw


def score_blocks(model, X, name, emit, *, block_rows=BLOCK_ROWS):
    import numpy as np
    require(type(block_rows) is int and block_rows > 0, 'Invalid scoring block size')
    stage = 'stage1' if name.startswith('stage1') else 'stage2'
    adapter = model if '_rf_' in name else DMatrixProbabilities(model)
    output = np.empty(X.shape[0] if stage == 'stage1' else (X.shape[0], 5), dtype=np.float64)
    for start in range(0, X.shape[0], block_rows):
        stop = min(start+block_rows, X.shape[0])
        output[start:stop] = models.probabilities(adapter, X[start:stop], stage)
        emit(f'scored_{stop}_of_{X.shape[0]}')
    return output


def reload_score(directory, fit_directory, fit, name, X, emit, *, device='cuda:0'):
    from experiments.c03.producer.artifacts import array_hash, tensor_hash
    path = safe_artifact(fit_directory, fit['model_file'])
    require(file_hash(path) == fit['model_sha256'], 'Model checksum drift before deserialization')
    emit('reloading_verified_model')
    model = models.load_verified_local_model(path, fit['model_family'])
    if fit['model_family'] == 'xgb':
        # Native model files do not preserve runtime device/thread configuration.
        model.set_params(device=device, n_jobs=2, tree_method='hist')
    before = check_model(model, name, device=device, columns=X.shape[1])
    input_sha = tensor_hash(X)
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scores = score_blocks(model, X, name, emit)
    require(not caught, 'Scoring warnings: '+repr([str(w.message) for w in caught]))
    after = check_model(model, name, device=device, columns=X.shape[1])
    require(before['device'] == after['device'], 'Inference device changed')
    require(tensor_hash(X) == input_sha, 'Scoring mutated validation input')
    elapsed = time.monotonic()-started
    require(file_hash(path) == fit['model_sha256'], 'Model changed during scoring')
    write_arrays(directory/'validation.npz', probabilities=scores)
    return {'model_sha256': fit['model_sha256'], 'model_family': fit['model_family'],
            'rows': X.shape[0], 'shape': list(scores.shape), 'dtype': str(scores.dtype),
            'fitted_classes': model.classes_.tolist(),
            'probabilities_sha256': array_hash(scores), 'score_seconds': elapsed,
            'score_file': 'validation.npz', 'score_file_sha256': file_hash(directory/'validation.npz'),
            'inference': after, 'inference_api': 'predict_proba' if '_rf_' in name else 'explicit_DMatrix_predict',
            'warnings': [], 'quality_metrics': None}


def commit_phase(directory, name, phase, request_sha, imported, result, *, fit_receipt_sha=None):
    require(name in COMPONENTS and phase in ('fit', 'score'), 'Invalid receipt phase')
    publish(directory/'complete.json', canonical({'schema_version': 1, 'scope': SCOPE,
        'scientific_checkpoint': False, 'component': name, 'phase': phase,
        'request_sha256': request_sha, 'import': imported, 'result': result,
        'fit_receipt_sha256': fit_receipt_sha}))


def verify_phase(directory, name, phase, request_sha, *, fit_receipt_sha=None):
    receipt = read(safe_artifact(directory, 'complete.json'))
    require(receipt['schema_version'] == 1 and receipt['scope'] == SCOPE and receipt['scientific_checkpoint'] is False
            and receipt['component'] == name and receipt['phase'] == phase
            and receipt['request_sha256'] == request_sha and receipt['fit_receipt_sha256'] == fit_receipt_sha,
            'Phase receipt binding drift')
    imported, result = receipt['import'], receipt['result']
    require(imported['scope'] == SCOPE and imported['component'] == name and imported['phase'] == phase
            and imported['scientific_checkpoint'] is False, 'Import scope differs')
    family = 'rf' if '_rf_' in name else 'xgb'
    require(result['model_family'] == family, 'Model family differs')
    require(result['fitted_classes'] == list(range(2 if name.startswith('stage1') else 5)), 'Class order drift')
    if phase == 'fit':
        expected = imported['expected']
        require(result['input_hashes'] == [expected['tensor'], expected['labels'], expected['weights'][name]]
                and result['rows'] == expected['rows'] and result['import_sha256'] == digest(imported), 'Fit input binding drift')
        filename = 'model.joblib' if family == 'rf' else 'model.ubj'
        require(result['model_file'] == filename and result['effective_parameters']['n_estimators'] == 8
                and result['effective_parameters']['n_jobs'] == 2
                and result['effective_parameters']['random_state'] == (123 if name.startswith('stage1') else 133)
                and result['features'] == 46, 'Fit definition drift')
        if family == 'xgb':
            require(result['fitted']['rounds'] == 8 and result['fitted']['trees'] == (8 if name.startswith('stage1') else 40),
                    'Fit tree count drift')
            xprobe.check_booster(result['fitted']['config'], name, device=result['effective_parameters']['device'])
        else:
            require(result['fitted'] == {'trees': 8, 'device': 'cpu'}, 'Fit RF definition drift')
        path = safe_artifact(directory, filename)
        require(file_hash(path) == result['model_sha256'] and path.stat().st_size == result['model_bytes'], 'Model checksum drift')
    else:
        require(result['score_file'] == 'validation.npz' and result['quality_metrics'] is None
                and result['warnings'] == [] and result['rows'] == imported['expected']['count']
                and result['shape'] == ([result['rows']] if name.startswith('stage1') else [result['rows'], 5])
                and result['dtype'] == 'float64', 'Score definition drift')
        require(file_hash(safe_artifact(directory, 'validation.npz')) == result['score_file_sha256'], 'Score checksum drift')
    return receipt
