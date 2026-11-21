"""Check the approved numerical rule beyond the modern subsampling boundary."""
import numpy as np
from experiments.literature.efc_discretization import make_discretizer


def test_all_rows_linear_edges_above_subsampling_boundary():
    # Distinct nonlinear values and an extreme tail make missed rows observable.
    x=np.linspace(0,1,200017,dtype=np.float64)**3
    x[-1]=1000
    matrix=np.column_stack((x,3*x[::-1]+7))
    first=make_discretizer().fit(matrix)
    expected=np.percentile(matrix,np.linspace(0,100,31),axis=0,method='linear')
    assert first.subsample is None and first.quantile_method=='linear'
    for j in range(2):np.testing.assert_array_equal(first.bin_edges_[j],expected[:,j])
    np.random.seed(987)
    second=make_discretizer().fit(matrix)
    for a,b in zip(first.bin_edges_,second.bin_edges_,strict=True):np.testing.assert_array_equal(a,b)
    queries=matrix[::997]
    np.testing.assert_array_equal(first.transform(queries),
        np.concatenate([first.transform(queries[i:i+17]) for i in range(0,len(queries),17)]))


def test_transform_does_not_refit_on_query_extremes():
    train=np.arange(600,dtype=float).reshape(300,2)
    fitted=make_discretizer().fit(train)
    before=[a.copy() for a in fitted.bin_edges_]
    result=fitted.transform([[-1e9,-1e9],[1e9,1e9]])
    np.testing.assert_array_equal(result,[[0,0],[29,29]])
    for a,b in zip(before,fitted.bin_edges_,strict=True):np.testing.assert_array_equal(a,b)
