"""Preprocessing for UCI HAR: label remap, duplicate-column removal, standardization, PCA.

Rule for everything in this file: fit() only ever sees TRAINING data. transform() is
then applied unchanged to the training set, to validation folds, and to the test set.

The classes follow scikit-learn's fit / transform convention (they subclass
BaseEstimator + TransformerMixin). That is only plumbing: it lets the whole chain go
into a sklearn Pipeline later, so it can be refit inside every cross-validation fold
(Nov 11-13) without leaking validation data into the scaler or the PCA.

    from sklearn.pipeline import make_pipeline
    prep = make_pipeline(DropDuplicateColumns(), Standardizer(), PCA(n_components=0.95))
    T_train = prep.fit_transform(X_train)   # fit on train
    T_test = prep.transform(X_test)          # apply to test

Everything marked TODO is mine to implement (NumPy / pandas only, no sklearn inside).
notebooks/02_preprocessing_pca.ipynb checks each piece against scikit-learn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def remap_labels(y: pd.Series) -> np.ndarray:
    """Map activity ids 1..6 to 0..5 (row/column index for one-vs-all and the confusion matrix).

    Returns a 1-D integer NumPy array.
    """
    # TODO
    raise NotImplementedError


class DropDuplicateColumns(BaseEstimator, TransformerMixin):
    """Remove feature columns that are exact copies of an earlier column.

    fit(X):        decide which columns to keep, using X (the training data) only. Sets
                     keep_     list[str]       column names kept, in their original order
                     dropped_  dict[str, str]  dropped column -> the earlier column it copies
    transform(X):  return X restricted to keep_ (still a DataFrame).
    """

    def fit(self, X: pd.DataFrame, y=None):
        # TODO
        raise NotImplementedError
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # TODO
        raise NotImplementedError


class Standardizer(BaseEstimator, TransformerMixin):
    """z = (x - mean) / std, where mean and std come from the training data.

    with_std=False only centers (x - mean): the "centered only" curve in 01_eda.

    fit(X):        sets mean_ and std_, 1-D arrays with one value per feature.
                   Use the population std (ddof=0), as in 01_eda and sklearn.
                   If with_std=False, std_ should be all ones.
    transform(X):  returns a NumPy array of the same shape as X.
    Accepts a DataFrame or an array (np.asarray(X) handles both).
    """

    def __init__(self, with_std: bool = True):
        self.with_std = with_std

    def fit(self, X, y=None):
        # TODO
        raise NotImplementedError
        return self

    def transform(self, X) -> np.ndarray:
        # TODO
        raise NotImplementedError


class PCA(BaseEstimator, TransformerMixin):
    """Principal component analysis via the SVD of the centered training matrix.

    n_components:
        None          keep every component
        int k         keep the first k components
        float in (0,1) keep the smallest k whose cumulative explained variance >= that fraction

    fit(Z) sets:
        mean_                          (d,)    column means of Z. About 0 if Z is already
                                               standardized, but subtract it anyway so the
                                               class is correct on any input.
        components_                    (k, d)  rows = principal directions (unit vectors),
                                               ordered by variance explained
        explained_variance_ratio_      (k,)    share of total variance per kept component
        explained_variance_ratio_all_  (r,)    the same for ALL components (for the scree plot)
        n_components_                  int     the k actually kept
    transform(Z):          (n, k) scores = (Z - mean_) @ components_.T
    inverse_transform(T):  (n, d) back in feature space = T @ components_ + mean_
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, Z, y=None):
        # TODO
        raise NotImplementedError
        return self

    def transform(self, Z) -> np.ndarray:
        # TODO
        raise NotImplementedError

    def inverse_transform(self, T) -> np.ndarray:
        # TODO
        raise NotImplementedError
