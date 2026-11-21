"""DOC rules on constructed arrays; no benchmark I/O."""
import numpy as np
import pytest

from experiments.literature.doc_core import Standardizer, CutoffMoments, predict


def test_paper_rule_accepts_failed_winner_if_another_class_passes():
    assert predict([[.9, .7]], [.95, .6]).tolist() == [0]
    assert predict([[.5, .2]], [.5, .8]).tolist() == [0]
    assert predict([[.4, .3]], [.5, .5]).tolist() == [2]
    assert predict([[.8, .8]], [.7, .7]).tolist() == [0]


def test_mirrored_mle_includes_misclassified_positives_and_streams():
    p = np.array([[.1, .9], [.95, .1], [.9, .2], [.1, 1.]])
    y = np.array([0, 0, 1, 1])
    m = CutoffMoments.empty(2)
    m.update(p[:1], y[:1]); m.update(p[1:], y[1:])
    expected = []
    for i in range(2):
        positive = p[y == i, i]
        expected.append(max(.5, 1 - 3 * np.std(np.r_[positive, 2 - positive], ddof=0)))
    np.testing.assert_allclose(m.thresholds(), expected, atol=1e-15, rtol=0)
    assert m.counts.tolist() == [2, 2]
    perfect = CutoffMoments.empty(2)
    perfect.update(np.eye(2), np.arange(2))
    np.testing.assert_array_equal(perfect.thresholds(), [1., 1.])


def test_scaling_population_variance_constant_column_and_no_query_refit():
    x = np.array([[2., 7.], [4., 7.], [9., 7.]])
    s = Standardizer.empty(2)
    s.update(x[:2]); s.update(x[2:])
    np.testing.assert_allclose(s.mean, x.mean(axis=0))
    np.testing.assert_allclose(s.scale, [x[:, 0].std(ddof=0), 1.])
    before = (s.count, s.mean.copy(), s.m2.copy())
    np.testing.assert_allclose(s.transform(x)[:, 0], ((x[:, 0] - x[:, 0].mean()) / x[:, 0].std()).astype('float32'))
    s.transform([[1000., 8.]])
    assert s.count == before[0]
    np.testing.assert_array_equal(s.mean, before[1])
    np.testing.assert_array_equal(s.m2, before[2])


@pytest.mark.parametrize('p,t', [([[np.nan, .1]], [.5,.5]), ([[1.01,.1]], [.5,.5]),
                              ([[.5,.1]], [.4,.5]), ([[.5,.1]], [np.inf,.5])])
def test_invalid_scores_cutoffs_rejected(p, t):
    with pytest.raises(ValueError):
        predict(p, t)


def test_missing_class_and_unknown_labels_fail():
    m = CutoffMoments.empty(2)
    m.update([[.8,.2]], [0])
    with pytest.raises(ValueError, match='Missing'):
        m.thresholds()
    with pytest.raises(ValueError, match='Unknown'):
        m.update([[.8,.2]], [2])
