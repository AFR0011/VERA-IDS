"""Historical constructors/weights and verified, locally produced checkpoints."""
import joblib
import numpy as np

from .artifacts import generated_file
from .plan import require


def json_parameters(values):
    # Some estimator defaults contain NaN. Record it explicitly without changing
    # the constructor values, and without non-standard JSON numeric literals.
    def normalize(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return {'nonfinite_float': str(value)}
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        if isinstance(value, (set, frozenset)):
            require(all(isinstance(v, str) for v in value), 'Unsupported metadata set')
            return {'set_of_strings': sorted(value)}
        require(value is None or isinstance(value, (str, int, float, bool)),
                'Unsupported constructor metadata')
        return value
    return normalize(values)


def construct(grid, node):
    if node['stage'] == 'stage1':
        return grid.build_stage1_model(node['model_family'], dict(node['parameters']),
                                       node['weight_mode'], seed=node['seed'])
    return grid.build_stage2_model(node['model_family'], dict(node['parameters']),
                                   n_classes=5, seed=node['seed'])


def probabilities(model, tensor, stage):
    classes = np.asarray(model.classes_)
    expected = np.arange(2 if stage == 'stage1' else 5)
    require(classes.ndim == 1 and len(classes) == len(expected) and
            set(classes.tolist()) == set(expected.tolist()), 'Missing/duplicate fitted classes')
    raw = np.asarray(model.predict_proba(tensor), dtype=np.float64)
    require(raw.shape == (tensor.shape[0], len(classes)), 'Probability shape mismatch')
    require(np.isfinite(raw).all() and ((raw >= 0) & (raw <= 1)).all(), 'Invalid probabilities')
    ordered = raw[:, [int(np.flatnonzero(classes == label)[0]) for label in expected]]
    require(np.allclose(ordered.sum(axis=1), 1., atol=1e-6, rtol=0), 'Unnormalized probabilities')
    if stage == 'stage1':
        return ordered[:, 1]  # Historical binary helper returns raw P(attack).
    clipped = np.clip(ordered, 1e-12, 1.)
    return clipped / clipped.sum(axis=1, keepdims=True)  # Historical multiclass helper.


def save_local_model(model, path, family):
    with generated_file(path) as temporary:
        if family == 'xgb':
            model.save_model(str(temporary))  # .ubj extension retained by temporary.
        else:
            require(family == 'rf', 'Unsupported model family')
            joblib.dump(model, temporary, compress=0)


def load_verified_local_model(path, family):
    """Internal adapter; caller MUST verify receipt/hash before this invocation."""
    if family == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier()
        model.load_model(str(path))
        return model
    require(family == 'rf', 'Unsupported model family')
    return joblib.load(path)
