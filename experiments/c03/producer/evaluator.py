"""Separate-process synthetic evaluator. Seal verification precedes test resolution.

Usage: python -m experiments.c03.producer.evaluator --run EXTERNAL_RUN_DIRECTORY
No benchmark evaluation mode exists. Older seals without an evaluator contract
cannot be upgraded or used here.
"""
import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .artifacts import array_hash, identity_hash, safe_artifact
from .checkpoints import Checkpoints, publish
from .models import probabilities
from .plan import ROOT, canonical, digest, file_hash, require, output_location
from .sealing import verify_seal
from .validation import truth_labels, verified_model, verify_prepared


def number(value):
    if value == '+inf':
        return float('inf')
    result = float(value)
    require(np.isfinite(result), 'Invalid frozen threshold')
    return result


def apply_policy(p1, p2, arm, selection):
    require(selection['status'] == 'feasible', 'Infeasible policy cannot produce predictions')
    require(p1.ndim == 1 and p2.shape == (len(p1), 5), 'Misaligned scores')
    require(np.isfinite(p1).all() and np.isfinite(p2).all() and
            ((p1 >= 0) & (p1 <= 1)).all() and ((p2 >= 0) & (p2 <= 1)).all() and
            np.allclose(p2.sum(axis=1), 1, atol=1e-8, rtol=0), 'Invalid test probabilities')
    q = np.clip(p2, 1e-12, 1.)
    q = q/q.sum(axis=1, keepdims=True)
    family = np.argmax(q, axis=1)
    maximum = q.max(axis=1)
    gate = p1 >= number(arm['stage1_threshold'])
    method = selection['method']
    if method == 'family_conditional':
        thresholds = np.asarray(selection['family_thresholds'], dtype=float)
        require(thresholds.shape == (5,) and np.isfinite(thresholds).all(), 'Invalid family thresholds')
        rejected = gate & (maximum < thresholds[family])
    else:
        top = np.partition(q, -2, axis=1)
        scores = {'max_confidence': 1-maximum,
                  'margin': 1-(top[:, -1]-top[:, -2]),
                  'entropy': -(q*np.log(np.clip(q, 1e-12, 1))).sum(axis=1)/np.log(5)}
        require(method in scores, 'Unknown rejection method')
        rejected = gate & (scores[method] >= number(selection['threshold']))
    return np.where(rejected, 6, np.where(gate, family+1, 0)).astype(np.int8)


def metrics(truth, prediction, p1):
    require(truth.shape == prediction.shape == p1.shape and len(truth) > 0, 'Misaligned metric population')
    require(np.issubdtype(truth.dtype, np.integer) and np.issubdtype(prediction.dtype, np.integer) and
            np.isin(truth, np.arange(7)).all() and np.isin(prediction, np.arange(7)).all(), 'Invalid labels')
    matrix = np.bincount(truth.astype(int)*7+prediction, minlength=49).reshape(7, 7)
    support, predicted, tp = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    den = support+predicted
    f1 = np.divide(2.*tp, den, out=np.zeros(7), where=den > 0)
    benign, known, attacks, unknown = truth == 0, truth != 6, (truth > 0) & (truth < 6), truth == 6
    rejected = prediction == 6
    def rate(hit, population):
        n, d = int((hit & population).sum()), int(population.sum())
        return {'value': n/d if d else None, 'numerator': n, 'denominator': d,
                'undefined_reason': None if d else 'No supported true population'}
    result = {'confusion_matrix': matrix.tolist(), 'label_support': support.tolist(),
              'macro_f1': float(f1.mean()), 'accuracy': float(tp.sum()/len(truth)),
              'udr': rate(rejected, unknown), 'fur_known': rate(rejected, known),
              'fur_known_attacks': rate(rejected, attacks),
              'benign_family_fpr': rate((prediction > 0) & (prediction < 6), benign),
              'benign_rejection': rate(rejected, benign),
              'realized_rejection': rate(rejected, np.ones(len(truth), dtype=bool))}
    target = truth != 0
    defined = np.unique(target).size == 2
    result['stage1_attack_auroc'] = float(roc_auc_score(target, p1)) if defined else None
    result['stage1_attack_average_precision'] = float(average_precision_score(target, p1)) if defined else None
    result['auc_definition'] = 'Stage-1 raw P(attack); positive = any true attack including Unknown; all test rows'
    result['auc_undefined_reason'] = None if defined else 'Both binary classes required'
    return result


def paired_results(truth, scores, arms, population_sha):
    """All policies consume the same ordered population; cache is by profile."""
    records = {}
    for budget in ('0.01', '0.03', '0.05', '0.10'):
        for method in ('max_confidence', 'margin', 'entropy', 'family_conditional'):
            pair = {}
            for name in ('blind', 'visible'):
                arm = arms[name]
                selection = arm['selections'][budget][method]
                if selection['status'] == 'infeasible':
                    pair[name] = {'status': 'infeasible', 'metrics': None, 'reason': 'No feasible validation policy'}
                    continue
                require(selection['method'] == method and float(selection['budget']) == float(budget), 'Policy key mismatch')
                p1, p2, row_sha = scores[arm['candidate_profile']]
                require(row_sha == population_sha, 'Candidate test-row identity mismatch')
                prediction = apply_policy(p1, p2, arm, selection)
                pair[name] = {'status': 'feasible', 'candidate_profile': arm['candidate_profile'],
                              'prediction_sha256': array_hash(prediction), 'metrics': metrics(truth, prediction, p1)}
            deltas = None
            if all(pair[a]['status'] == 'feasible' for a in pair):
                deltas = {}
                for key, value in pair['visible']['metrics'].items():
                    other = pair['blind']['metrics'][key]
                    if isinstance(value, dict) and 'value' in value:
                        deltas[key] = (value['value']-other['value']
                                       if value['value'] is not None and other['value'] is not None else None)
                    elif isinstance(value, float):
                        deltas[key] = value-other if other is not None else None
                    elif value is None:
                        deltas[key] = None
            records[f'{budget}/{method}'] = {'arms': pair, 'visible_minus_blind': deltas,
                'delta_undefined_reason': None if deltas is not None else 'At least one arm infeasible',
                'feasibility_transition': pair['blind']['status']+'->'+pair['visible']['status']}
    return records


def read_test(contract):
    """Called only after verification. Hash and ordered source-row identities bind the read."""
    root = Path(contract['root']).resolve()
    require(not root.is_relative_to(ROOT), 'Synthetic test sources must be outside repository')
    parts = contract['parts']
    require(parts and len({p['path'] for p in parts}) == len(parts), 'Duplicate/empty test inventory')
    require([p['path'] for p in parts] == sorted(p['path'] for p in parts), 'Unordered test inventory')
    require(0 < sum(p['rows'] for p in parts) <= 10000, 'Synthetic test row limit exceeded')
    frames, identities = [], []
    for part in parts:
        path = safe_artifact(root, part['path'])
        require(file_hash(path) == part['sha256'], 'Test source byte drift')
        frame = pd.read_parquet(path)
        require(len(frame) == part['rows'] and file_hash(path) == part['sha256'], 'Test row count/byte drift')
        frames.append(frame)
        identities.append(pd.DataFrame({'source_part': [part['path']]*len(frame),
            'source_sha256': [part['sha256']]*len(frame), 'row_offset': np.arange(len(frame), dtype=np.int64)}))
    return pd.concat(frames, ignore_index=True), pd.concat(identities, ignore_index=True)


def evaluate(store):
    selection = verify_seal(store)  # Must precede test path resolution, model load and reads.
    contract = selection.get('synthetic_evaluation_contract')
    require(contract is not None and contract.get('scope') == 'SYNTHETIC_ONLY', 'No prospective synthetic evaluation contract')
    require(selection['evaluator_revision'] == file_hash(__file__) == contract['evaluator_revision'], 'Evaluator revision drift')
    require(store.plan.get('status') == 'SYNTHETIC_VALIDATION_PLAN', 'Benchmark evaluation not authorized')
    lock = store.path/'.evaluation-lock'
    with lock.open('x'):
        pass
    try:
        require(not (store.path/'evaluation.json').exists(), 'Evaluation already exists; no overwrite')
        frame, identities = read_test(contract)
        population = identity_hash(identities)
        truth = truth_labels(frame)
        verify_prepared(store)
        prep = joblib.load(safe_artifact(store.path, 'prepared/preprocessor/preprocessor.joblib'))
        X = prep.transform(frame.drop(columns=['y_stage1_attack', 'y_stage2_family']))
        component_cache, scores = {}, {}
        for arm in selection['decisions']['arms'].values():
            if not any(s['status'] == 'feasible' for per_budget in arm['selections'].values() for s in per_budget.values()):
                continue
            profile = arm['candidate_profile']
            if profile in scores:
                continue
            refs = store.plan['graph']['candidate_references'][profile]
            for stage, name in refs.items():
                if name not in component_cache:
                    model = verified_model(store, name)
                    component_cache[name] = probabilities(model, X, stage)
                    del model
                    gc.collect()
            scores[profile] = (component_cache[refs['stage1']], component_cache[refs['stage2']], population)
        results = paired_results(truth, scores, selection['decisions']['arms'], population)
        require(verify_seal(store) == selection, 'Seal changed during evaluation')
        for part in contract['parts']:
            require(file_hash(safe_artifact(Path(contract['root']).resolve(), part['path'])) == part['sha256'], 'Test changed during scoring')
        result = {'scope': 'SYNTHETIC_ONLY', 'selection_sha256': file_hash(store.path/'selection.json'),
                  'evaluator_revision': file_hash(__file__), 'test_contract_sha256': digest(contract),
                  'population_sha256': population, 'truth_sha256': array_hash(truth), 'rows': len(truth),
                  'records': results, 'uncertainty': 'Not estimated; descriptive synthetic verification only'}
        with store.lock():
            publish(store.path/'evaluation.json', canonical(result))
            publish(store.path/'evaluation-complete.json', canonical({'evaluation_sha256': file_hash(store.path/'evaluation.json')}))
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = output_location(args.run)
    plan = json.loads((run/'plan.json').read_text(encoding='utf-8'))
    result = evaluate(Checkpoints(run, plan))
    print(f"Synthetic evaluation complete: {result['rows']} aligned rows; 16 paired policies")


if __name__ == '__main__':
    main()
