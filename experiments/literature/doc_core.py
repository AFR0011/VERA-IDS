"""Approved DOC paper-rule numerical core; no benchmark I/O or model fitting."""
from dataclasses import dataclass

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matrix(values, width=None):
    values = np.asarray(values)
    require(values.ndim == 2 and len(values) > 0, 'Expected nonempty matrix')
    require(width is None or values.shape[1] == width, 'Feature/class width differs')
    require(np.issubdtype(values.dtype, np.number) and np.isfinite(values).all(), 'Nonfinite/nonnumeric input')
    return values


@dataclass
class Standardizer:
    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls, width):
        require(type(width) is int and width > 0, 'Invalid width')
        return cls(0, np.zeros(width, dtype=np.float64), np.zeros(width, dtype=np.float64))

    def update(self, train_block):
        x = matrix(train_block, len(self.mean)).astype(np.float64)
        n = len(x)
        mean = x.mean(axis=0)
        centered = x - mean
        m2 = np.sum(centered * centered, axis=0)
        delta = mean - self.mean
        total = self.count + n
        new_m2 = self.m2 + m2 + delta * delta * (self.count * n / total)
        new_mean = self.mean + delta * (n / total)
        require(np.isfinite(new_mean).all() and np.isfinite(new_m2).all(), 'Scaling moment overflow')
        self.mean, self.m2, self.count = new_mean, new_m2, total

    @property
    def scale(self):
        require(self.count > 0, 'Unfitted scaling')
        variance = self.m2 / self.count
        require(np.isfinite(variance).all() and (variance >= 0).all(), 'Invalid scaling moments')
        return np.where(variance == 0, 1.0, np.sqrt(variance))

    def transform(self, block):
        x = matrix(block, len(self.mean)).astype(np.float64)
        out = ((x - self.mean) / self.scale).astype(np.float32)
        require(np.isfinite(out).all(), 'Transformed float32 overflow')
        return out


@dataclass
class CutoffMoments:
    counts: np.ndarray
    squares: np.ndarray

    @classmethod
    def empty(cls, classes):
        require(type(classes) is int and classes > 1, 'Invalid class count')
        return cls(np.zeros(classes, dtype=np.int64), np.zeros(classes, dtype=np.float64))

    def update(self, probabilities, true_labels):
        p = probabilities_checked(probabilities, len(self.counts)).astype(np.float64)
        y = labels_checked(true_labels, len(p), len(self.counts))
        # Include all positives, not only correctly classified rows.
        for index in range(len(self.counts)):
            own = p[y == index, index]
            self.counts[index] += len(own)
            self.squares[index] += np.sum((1.0 - own) ** 2, dtype=np.float64)

    def thresholds(self):
        require((self.counts > 0).all(), 'Missing TRAIN class support')
        require(np.isfinite(self.squares).all() and (self.squares >= 0).all(), 'Invalid cutoff moments')
        return np.maximum(0.5, 1.0 - 3.0 * np.sqrt(self.squares / self.counts))


def labels_checked(labels, rows, classes):
    y = np.asarray(labels)
    require(y.shape == (rows,) and np.issubdtype(y.dtype, np.integer), 'Invalid label shape/type')
    require(((y >= 0) & (y < classes)).all(), 'Unknown/out-of-range class in known population')
    return y


def probabilities_checked(values, classes=None):
    p = matrix(values, classes)
    require(p.shape[1] > 1 and ((p >= 0) & (p <= 1)).all(), 'Invalid sigmoid probabilities')
    return p


def predict(probabilities, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    require(thresholds.ndim == 1 and np.isfinite(thresholds).all()
            and ((thresholds >= .5) & (thresholds <= 1)).all(), 'Invalid cutoffs')
    p = probabilities_checked(probabilities, len(thresholds)).astype(np.float64)
    result = np.argmax(p, axis=1)  # First index wins ties, including failed winner.
    result[np.all(p < thresholds, axis=1)] = len(thresholds)
    return result
