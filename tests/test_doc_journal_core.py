"""More standardization/threshold edge cases without a neural dependency."""
import numpy as np
import pytest
from experiments.literature.doc_core import Standardizer, CutoffMoments


def test_resume_scaling_and_cutoff_accumulators_without_double_counting():
    rng = np.random.default_rng(52)
    x = rng.normal(size=(65, 46))
    a = Standardizer.empty(46)
    a.update(x[:32])
    restored = Standardizer(a.count, a.mean.copy(), a.m2.copy())
    a.update(x[32:]); restored.update(x[32:])
    np.testing.assert_array_equal(a.mean, restored.mean)
    np.testing.assert_array_equal(a.scale, restored.scale)
    p = rng.uniform(size=(65, 6)); labels = np.arange(65) % 6
    m = CutoffMoments.empty(6); m.update(p[:32], labels[:32])
    reloaded = CutoffMoments(m.counts.copy(), m.squares.copy())
    m.update(p[32:], labels[32:]); reloaded.update(p[32:], labels[32:])
    np.testing.assert_array_equal(m.thresholds(), reloaded.thresholds())


def test_nonfinite_scaling_and_width_drift_fail():
    s = Standardizer.empty(2)
    with pytest.raises(ValueError):
        s.update([[1, np.inf]])
    with pytest.raises(ValueError):
        s.update([[1, 2, 3]])
    with pytest.raises(ValueError, match='Unfitted'):
        s.transform([[1,2]])
