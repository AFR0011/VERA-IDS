import ast
import numpy as np
import pytest
from experiments.literature import efc_adapter as adapter


def test_constructor_patch_changes_only_the_three_approved_keywords():
    source=b'def fit(self):\n    x=KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="quantile")\n    return native_energy(x)\n'
    patched=adapter._insert_settings(source)
    tree=ast.parse(patched)
    call=tree.body[0].body[0].value
    settings={kw.arg:ast.literal_eval(kw.value) for kw in call.keywords if kw.arg!='n_bins'}
    assert settings=={'encode':'ordinal','strategy':'quantile','subsample':None,'quantile_method':'linear','random_state':None}
    assert b'return native_energy(x)' in patched


@pytest.mark.parametrize('source',[b'changed',b'',adapter.OLD])
def test_unreviewed_sources_cannot_be_patched(source):
    with pytest.raises(ValueError,match='Unreviewed'):adapter.patched_classifier(source)


def test_ambiguous_constructor_is_rejected():
    source=b'def fit(self):\n    x=KBinsDiscretizer('+adapter.OLD+b')\n    y=KBinsDiscretizer('+adapter.OLD+b')\n'
    with pytest.raises(ValueError):adapter._insert_settings(source)


def test_nonfinite_training_is_rejected_before_preprocessing():
    with pytest.raises(ValueError):adapter.preprocess_training(np.array([[1.,np.nan],[2.,3.]]))
