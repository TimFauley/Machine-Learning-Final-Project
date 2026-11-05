"""Check the shared settings contract and train-only transforms."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess import notebook_settings, make_prep, Standardizer


def test_settings_follow_saved_notebook(tmp_path):
    path = tmp_path / "02.ipynb"
    for k in (0.95, 2):
        path.write_text(json.dumps({"cells": [
            {"cell_type": "code", "source": ["%autoreload 2\n", "STANDARDIZE = True\n"]},
            {"cell_type": "code", "source": [f"N_COMPONENTS = {k}\n"]}]}))
        settings = notebook_settings(path)
        assert settings == {"STANDARDIZE": True, "N_COMPONENTS": k}
        assert make_prep(settings["N_COMPONENTS"], settings["STANDARDIZE"])[-1].n_components == k


def test_pipeline_uses_training_statistics_and_columns():
    X = pd.DataFrame({"a": [0., 1., 2., 3.], "copy": [0., 1., 2., 3.],
                      "b": [3., 0., 2., 1.]})
    test = pd.DataFrame({"a": [50.], "copy": [-100.], "b": [30.]})
    prep = make_prep(2)
    prep.fit(X)
    mean = prep[1].mean_.copy()
    centres = prep[-1].components_.copy()
    scores = prep.transform(test)
    assert prep[0].dropped_ == {"copy": "a"}
    assert_allclose(prep[1].mean_, mean)
    assert_allclose(prep[-1].components_, centres)
    assert_allclose(prep[-1].inverse_transform(scores), (test[["a", "b"]] - mean) / prep[1].std_)


def test_standardizer_handles_constant_features():
    X = np.array([[1., 2.], [1., 4.], [1., 6.]])
    for with_std in (True, False):
        ours = Standardizer(with_std).fit(X)
        ref = StandardScaler(with_std=with_std).fit(X)
        assert_allclose(ours.transform(X), ref.transform(X))
        assert_allclose(ours.transform([[5., 8.]]), ref.transform([[5., 8.]]))
