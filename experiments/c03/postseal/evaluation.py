"""Post-seal streaming inference. One component per isolated worker; no selection."""
import hashlib
import warnings
import joblib
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

from . import contract, selection
from experiments.c03.pilot import models
from experiments.c03.engineering import lifecycle, preparation_probe as host
from experiments.c03.producer import models as native, validation as v, evaluator as reference
from experiments.c03.producer.artifacts import array_hash, safe_artifact, write_arrays
from experiments.c03.producer.checkpoints import publish
from experiments.c03.producer.plan import canonical, digest, file_hash, require


class PopulationHash:
    """Exact existing identity_hash, streamed with declared full-population dtypes."""
    def __init__(self, rows, parts):
        self.rows, self.seen = rows, 0
        self.dtypes = {'source_part': np.dtype(f'U{max(len(p["path"]) for p in parts)}'),
                       'source_sha256': np.dtype('U64'), 'row_offset': np.dtype('int64')}
        self.hashes = {}
        for name, dtype in self.dtypes.items():
            h = hashlib.sha256()
            h.update(str(dtype).encode('ascii'))
            h.update(str((rows,)).encode('ascii'))
            self.hashes[name] = h

    def add(self, part, offset, size):
        values = {'source_part': np.full(size, part['path'], dtype=self.dtypes['source_part']),
                  'source_sha256': np.full(size, part['sha256'], dtype='U64'),
                  'row_offset': np.arange(offset, offset+size, dtype=np.int64)}
        require(self.seen+size <= self.rows, 'Unexpected extra test rows')
        for name, array in values.items():
            self.hashes[name].update(memoryview(array).cast('B'))
        self.seen += size

    def finish(self):
        require(self.seen == self.rows, 'Test row count differs')
        return digest({name: h.hexdigest() for name, h in self.hashes.items()})


def test_batches(spec):
    # Invoked only AFTER seal and all producer artifacts have been verified.
    root = host.DATA_ROOT
    actual = sorted(p.relative_to(root).as_posix() for p in (root/'test').glob('*.parquet'))
    require(actual == [p['path'] for p in spec['parts']], 'Test inventory/order drift')
    for part in spec['parts']:
        path = safe_artifact(root, part['path'])
        require(file_hash(path) == part['sha256'], 'Test file changed before read')
        source = pq.ParquetFile(path)
        require(source.metadata.num_rows > 0, 'Empty test part')
        offset = 0
        for batch in source.iter_batches(batch_size=spec['block_rows'], use_threads=False):
            frame = batch.to_pandas()
            yield part, offset, frame
            offset += len(frame)
        require(offset == source.metadata.num_rows and file_hash(path) == part['sha256'], 'Test changed during read')


def score_component(store, name, directory, emit):
    record = selection.verify_seal(store)  # Before even resolving a test source path.
    require(name in selection.selected_components(store, record), 'Unselected component cannot read test')
    node = store.source.plan['graph']['components'][name]
    fit_dir = safe_artifact(store.source.path, store.plan['components'][name]['fit'])
    fit = store.source.read_json(fit_dir/'complete.json')['result']
    path = safe_artifact(fit_dir, fit['model_file'])
    require(file_hash(path) == fit['model_sha256'], 'Model checksum drift before deserialization')
    v.historical()  # Registers the pinned historical preprocessor class.
    prep = joblib.load(store.source.path/'prepared/preprocessor/preprocessor.joblib')
    model = native.load_verified_local_model(path, node['model_family'])
    device = node['parameters'].get('device', 'cpu')
    if node['model_family'] == 'xgb':
        model.set_params(device=device, n_jobs=2, tree_method='hist')
    models.check_model(model, node, name, device=device)
    adapter = model if node['model_family'] == 'rf' else lifecycle.DMatrixProbabilities(model)
    spec = store.plan['test']
    truth = np.empty(spec['rows'], dtype=np.int8)
    probabilities = np.empty(spec['rows'] if node['stage'] == 'stage1' else (spec['rows'], 5), dtype=np.float64)
    population = PopulationHash(spec['rows'], spec['parts'])
    cursor, part_rows = 0, {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for part, offset, frame in test_batches(spec):
            stop = cursor+len(frame)
            require(stop <= spec['rows'], 'Unexpected extra test rows')
            truth[cursor:stop] = v.truth_labels(frame)
            X = prep.transform(frame.drop(columns=['y_stage1_attack', 'y_stage2_family']))
            require(isinstance(X, csr_matrix) and X.shape == (len(frame), 46)
                    and X.dtype == np.float64, 'Test transform schema differs')
            probabilities[cursor:stop] = native.probabilities(adapter, X, node['stage'])
            population.add(part, offset, len(frame))
            part_rows[part['path']] = offset+len(frame)
            cursor = stop
            emit(f'{name}:test_rows_{cursor}_of_{spec["rows"]}')
    require(not caught, 'Test inference warnings: '+repr([str(w.message) for w in caught]))
    population_sha = population.finish()
    support = np.bincount(truth, minlength=7)
    require((support >= spec['minimum_per_class']).all(), 'Test support floor failed')
    info = models.check_model(model, node, name, device=device)
    require(selection.verify_seal(store) == record, 'Seal changed during test scoring')
    write_arrays(directory/'test.npz', truth=truth, probabilities=probabilities)
    result = {'scope': contract.SCOPE, 'contract_sha256': digest(store.plan),
        'selection_sha256': file_hash(store.path/'selection.json'), 'component': name,
        'model_sha256': fit['model_sha256'], 'rows': cursor, 'part_rows': part_rows,
        'population_sha256': population_sha, 'truth_sha256': array_hash(truth), 'support': support.tolist(),
        'probabilities_sha256': array_hash(probabilities), 'shape': list(probabilities.shape),
        'inference': info, 'file_sha256': file_hash(directory/'test.npz')}
    publish(directory/'complete.json', canonical(result))
    return result


def verify_component(store, name, directory):
    record = store.read((directory/'complete.json').relative_to(store.path).as_posix())
    require(record['scope'] == contract.SCOPE and record['contract_sha256'] == digest(store.plan)
            and record['selection_sha256'] == file_hash(store.path/'selection.json')
            and record['component'] == name and record['rows'] == store.plan['test']['rows'], 'Test checkpoint binding drift')
    fit_dir = safe_artifact(store.source.path, store.plan['components'][name]['fit'])
    fit = store.source.read_json(fit_dir/'complete.json')['result']
    require(record['model_sha256'] == fit['model_sha256'], 'Wrong evaluated model')
    node = store.source.plan['graph']['components'][name]
    info = record['inference']
    require(info['trees'] == fit['fitted']['trees'] and info['classes'] == fit['fitted']['classes'], 'Inference ensemble differs')
    if node['model_family'] == 'xgb':
        require(info['device'] == node['parameters']['device'] and info['rounds'] == node['parameters']['n_estimators'], 'GPU/round drift')
        models.xcheck.check_booster(info['config'], name, device=node['parameters']['device'])
    require(file_hash(directory/'test.npz') == record['file_sha256'], 'Test score artifact drift')
    with np.load(directory/'test.npz', allow_pickle=False) as arrays:
        truth, probabilities = arrays['truth'], arrays['probabilities']
    require(truth.shape == (record['rows'],) and truth.dtype == np.int8
            and np.isin(truth, np.arange(7)).all() and array_hash(truth) == record['truth_sha256'], 'Test truth content drift')
    expected = [record['rows']] if node['stage'] == 'stage1' else [record['rows'], 5]
    require(record['shape'] == expected == list(probabilities.shape) and probabilities.dtype == np.float64
            and array_hash(probabilities) == record['probabilities_sha256'] and np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all(), 'Test score content drift')
    if node['stage'] == 'stage2':
        require(np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-8), 'Unnormalized test probabilities')
    require(np.bincount(truth, minlength=7).tolist() == record['support']
            and min(record['support']) >= store.plan['test']['minimum_per_class'], 'Test support drift')
    require(set(record['part_rows']) == {p['path'] for p in store.plan['test']['parts']}
            and all(type(n) is int and n > 0 for n in record['part_rows'].values())
            and sum(record['part_rows'].values()) == record['rows'], 'Test part coverage differs')
    return record, truth, probabilities


def completed_component(store, name):
    matches = list((store.path/'attempts'/name).glob('a*/complete.json'))
    require(len(matches) <= 1, 'Duplicate completed component')
    if not matches:
        return None
    path = safe_artifact(store.path, matches[0].parent.relative_to(store.path).as_posix())
    verify_component(store, name, path)
    return path


def analyze(store, emit):
    selection_record = selection.verify_seal(store)
    names = selection.selected_components(store, selection_record)
    cached, checkpoints, truth, population, part_rows = {}, {}, None, None, None
    for name in names:
        directory = completed_component(store, name)
        require(directory is not None, 'Missing selected test scores')
        record, labels, scores = verify_component(store, name, directory)
        if truth is None:
            truth, population, part_rows = labels, record['population_sha256'], record['part_rows']
        require(np.array_equal(truth, labels) and population == record['population_sha256']
                and part_rows == record['part_rows'], 'Unpaired test populations')
        cached[name] = scores
        checkpoints[name] = file_hash(directory/'complete.json')
    scores = {}
    for arm in selection_record['decisions']['arms'].values():
        profile = arm['candidate_profile']
        refs = store.source.plan['graph']['candidate_references'][profile]
        if all(n in cached for n in refs.values()):
            scores[profile] = (cached[refs['stage1']], cached[refs['stage2']], population)
    emit('computing_frozen_paired_metrics')
    records = reference.paired_results(truth, scores, selection_record['decisions']['arms'], population)
    require(selection.verify_seal(store) == selection_record, 'Seal changed during analysis')
    result = {'scope': contract.SCOPE, 'contract_sha256': digest(store.plan),
        'selection_sha256': file_hash(store.path/'selection.json'), 'component_checkpoints': checkpoints,
        'rows': len(truth) if truth is not None else None, 'population_sha256': population,
        'truth_sha256': array_hash(truth) if truth is not None else None,
        'test_accessed': bool(names), 'records': records, 'uncertainty': store.plan['uncertainty'],
        'interpretation': 'One fixed holdout/seed on the prepared corpus; no source-independent or population causal claim'}
    if (store.path/'evaluation.json').exists():
        require(store.read('evaluation.json') == result, 'Interrupted analysis differs; no overwrite')
    else:
        publish(store.path/'evaluation.json', canonical(result))
    if not (store.path/'evaluation-complete.json').exists():
        publish(store.path/'evaluation-complete.json', canonical({'evaluation_sha256': file_hash(store.path/'evaluation.json')}))
    return result


def verify_result(store):
    record = selection.verify_seal(store)
    result = store.read('evaluation.json')
    require(store.read('evaluation-complete.json') == {'evaluation_sha256': file_hash(store.path/'evaluation.json')}
            and result['contract_sha256'] == digest(store.plan)
            and result['selection_sha256'] == file_hash(store.path/'selection.json'), 'Evaluation completion drift')
    names = selection.selected_components(store, record)
    require(result['component_checkpoints'] == {n: file_hash(completed_component(store, n)/'complete.json') for n in names},
            'Evaluation dependency drift')
    return result
