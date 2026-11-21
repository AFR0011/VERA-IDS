"""User-adopted 2026-11-18 EFC quantiles, explicit across dependency versions.

This is the numerical preprocessing constructor for a future EFC adapter.
It does not fit EFC, read benchmark data, or modify the upstream package.
"""
from sklearn.preprocessing import KBinsDiscretizer


def make_discretizer(n_bins=30):
    return KBinsDiscretizer(n_bins=n_bins,encode='ordinal',strategy='quantile',
                           subsample=None,quantile_method='linear',random_state=None)
