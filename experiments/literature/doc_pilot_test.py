"""Post-seal test export and GPU inference for the fixed DOC pilot."""
import argparse
import io
from pathlib import Path

import numpy as np

from . import doc_pilot_io as rt, doc_core as core
from experiments.c03.producer.locking import file_lock


def save_arrays(path, **arrays):
    stream = io.BytesIO(); np.savez(stream, **arrays)
    if path.exists():
        import hashlib
        rt.require(rt.file_hash(path) == hashlib.sha256(stream.getvalue()).hexdigest(), 'Interrupted array differs')
    else:
        rt.publish(path, stream.getvalue())


def verify_part(run, i, entry, kind):
    name = f'{kind}/p{i:04d}'
    value = rt.read(rt.safe_artifact(run, name + '.json'))
    rt.require(value['request_sha256'] == rt.file_hash(run / 'request.json')
               and value['seal_sha256'] == rt.file_hash(run / 'seal.json') and value['source'] == entry
               and rt.file_hash(rt.safe_artifact(run, name + '.npz')) == value['arrays_sha256'], 'DOC test checkpoint drift')
    return value


def export(run, plan, emit):
    # This process runs in the main data environment, only after a verified seal.
    import pyarrow.parquet as pq
    from experiments.c03.producer.artifacts import array_hash
    from experiments.c03.engineering.preparation_probe import DATA_ROOT
    from experiments.c03.producer.validation import truth_labels
    from experiments.c03.postseal.evaluation import PopulationHash
    rt.verify_seal(run, plan); folder, _ = rt.preparation(plan)
    value = rt.read(folder / 'scaler.json')
    scaler = core.Standardizer(value['count'], np.asarray(value['mean']), np.asarray(value['m2']))
    spec = plan['test']; population = PopulationHash(spec['rows'], spec['parts']); truths = []
    rt.require(sorted(p.relative_to(DATA_ROOT).as_posix() for p in (DATA_ROOT / 'test').glob('*.parquet'))
               == [e['path'] for e in spec['parts']], 'Test inventory drift')
    (run / 'test-inputs').mkdir(exist_ok=True)
    for i, entry in enumerate(spec['parts']):
        rt.guard(run, plan); emit('sealed_test_export', part=i + 1, total=len(spec['parts']))
        source = rt.safe_artifact(DATA_ROOT, entry['path'])
        rt.require(rt.file_hash(source) == entry['sha256'], 'Test source drift')
        name = f'test-inputs/p{i:04d}'
        if not (run / (name + '.json')).exists():
            xs, ys = [], []
            data = pq.ParquetFile(source)
            for batch in data.iter_batches(batch_size=8192, use_threads=False):
                rt.guard(run, plan); frame = batch.to_pandas()
                frame.columns = [str(c).strip().lower().replace(' ', '_') for c in frame.columns]
                rt.require(frame.columns.is_unique and set(plan['columns']) <= set(frame.columns), 'Test schema differs')
                xs.append(scaler.transform(frame[plan['columns']].to_numpy(dtype=np.float64)))
                ys.append(truth_labels(frame))
            x, y = np.concatenate(xs), np.concatenate(ys)
            rt.require(len(y) == data.metadata.num_rows and rt.file_hash(source) == entry['sha256'], 'Test source changed')
            save_arrays(run / (name + '.npz'), X=x, truth=y)
            rt.bound_json(run, name + '.json', {'seal_sha256': rt.file_hash(run / 'seal.json'), 'source': entry,
                'rows': len(y), 'arrays_sha256': rt.file_hash(run / (name + '.npz'))})
        value = verify_part(run, i, entry, 'test-inputs')
        with np.load(run / (name + '.npz'), allow_pickle=False) as a:
            y = a['truth']; rt.require(a['X'].shape == (len(y), 46) and len(y) == value['rows'], 'Test input shape drift')
        truths.append(y); population.add(entry, 0, len(y))
    truth = np.concatenate(truths)
    rt.require(population.finish() == plan['expected_population_sha256']
               and array_hash(truth) == plan['expected_truth_sha256'], 'Test identity/truth differs')
    rt.guard(run, plan)
    rt.bound_json(run, 'test-inputs/complete.json', {'seal_sha256': rt.file_hash(run / 'seal.json'),
        'rows': len(truth), 'population_sha256': population.finish(), 'truth_sha256': array_hash(truth),
        'parts': [rt.file_hash(run / f'test-inputs/p{i:04d}.json') for i in range(len(spec['parts']))]})


def score(run, plan, emit):
    from . import doc_training as training
    from .doc_pilot_fit import load_tensor
    from .doc_readiness_fit import environment
    environment(); seal = rt.verify_seal(run, plan)
    exported = rt.read(run / 'test-inputs/complete.json')
    rt.require(exported['seal_sha256'] == rt.file_hash(run / 'seal.json')
               and exported['request_sha256'] == rt.file_hash(run / 'request.json'), 'Test export binding drift')
    training.configure('cuda:0'); model = training.network('cuda:0')
    model.load_state_dict(load_tensor(run / 'fit', seal['model_ref'])['model'])
    (run / 'test-scores').mkdir(exist_ok=True)
    for i, entry in enumerate(plan['test']['parts']):
        rt.guard(run, plan); emit('sealed_test_score', part=i + 1, total=len(plan['test']['parts']))
        verify_part(run, i, entry, 'test-inputs')
        rt.require(rt.file_hash(run / f'test-inputs/p{i:04d}.json') == exported['parts'][i], 'Test input receipt drift')
        name = f'test-scores/p{i:04d}'
        if not (run / (name + '.json')).exists():
            with np.load(run / f'test-inputs/p{i:04d}.npz', allow_pickle=False) as a:
                x, y = a['X'], a['truth']
            probabilities = training.probabilities(model, x, 'cuda:0', block=8192)
            pred = core.predict(probabilities, seal['thresholds']).astype(np.int8)
            save_arrays(run / (name + '.npz'), truth=y, prediction=pred, probabilities=probabilities)
            rt.bound_json(run, name + '.json', {'seal_sha256': rt.file_hash(run / 'seal.json'), 'source': entry,
                'rows': len(y), 'arrays_sha256': rt.file_hash(run / (name + '.npz'))})
        verify_part(run, i, entry, 'test-scores')
    rt.guard(run, plan)
    rt.bound_json(run, 'test-scores/complete.json', {'seal_sha256': rt.file_hash(run / 'seal.json'),
        'parts': [rt.file_hash(run / f'test-scores/p{i:04d}.json') for i in range(len(plan['test']['parts']))]})


def reconcile(run, plan):
    from experiments.c03.producer.artifacts import array_hash
    from .efc_fidelity import metrics
    from experiments.c03.postseal.evaluation import PopulationHash
    seal = rt.verify_seal(run, plan); spec = plan['test']
    population = PopulationHash(spec['rows'], spec['parts']); ys, ps = [], []
    complete = rt.read(run / 'test-scores/complete.json')
    rt.require(complete['seal_sha256'] == rt.file_hash(run / 'seal.json')
               and complete['request_sha256'] == rt.file_hash(run / 'request.json'), 'Score completion drift')
    for i, entry in enumerate(spec['parts']):
        record = verify_part(run, i, entry, 'test-scores')
        rt.require(rt.file_hash(run / f'test-scores/p{i:04d}.json') == complete['parts'][i], 'Score receipt drift')
        with np.load(run / f'test-scores/p{i:04d}.npz', allow_pickle=False) as a:
            y, p, probabilities = a['truth'], a['prediction'], a['probabilities']
        rt.require(len(y) == record['rows'] and np.array_equal(core.predict(probabilities, seal['thresholds']), p),
                   'DOC prediction rule reconciliation failed')
        ys.append(y); ps.append(p); population.add(entry, 0, len(y))
    truth, pred = np.concatenate(ys), np.concatenate(ps)
    rt.require(population.finish() == plan['expected_population_sha256']
               and array_hash(truth) == plan['expected_truth_sha256'], 'Completed population drift')
    result_metrics = metrics(truth, pred)
    result_metrics['auc_not_reported_reason'] = 'DOC sigmoid outputs have no adopted Stage-1 P(attack) definition'
    rt.require(min(result_metrics['label_support']) >= spec['minimum_per_class'], 'Missing test support')
    return {'scope': rt.SCOPE, 'status': 'COMPLETE', 'rows': len(truth), 'metrics': result_metrics,
            'population_sha256': population.finish(), 'truth_sha256': array_hash(truth),
            'prediction_sha256': array_hash(pred), 'seal_sha256': rt.file_hash(run / 'seal.json'),
            'comparison': 'Paper-rule tabular adaptation; native operating point, not policy-matched to VERA',
            'uncertainty': 'One fixed holdout/seed; no generalization or independence claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--attempt', required=True)
    parser.add_argument('--stage', choices=('export', 'score'), required=True)
    args = parser.parse_args(); plan, done = rt.supervise_owner(args.run, args.attempt)
    try:
        with file_lock(args.run / '.worker.guard'):
            (export if args.stage == 'export' else score)(args.run, plan,
                lambda stage, **info: rt.emit(args.run, args.attempt, stage, **info))
    finally:
        done.set()


if __name__ == '__main__':
    main()
