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

Numerical implementations use NumPy / pandas; sklearn supplies estimator plumbing.
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
    return np.asarray(y, dtype=int) - 1


class DropDuplicateColumns(BaseEstimator, TransformerMixin):
    """Remove feature columns that are exact copies of an earlier column.

    fit(X):        decide which columns to keep, using X (the training data) only. Sets
                     keep_     list[str]       column names kept, in their original order
                     dropped_  dict[str, str]  dropped column -> the earlier column it copies
    transform(X):  return X restricted to keep_ (still a DataFrame).
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.keep_ = []
        self.dropped_ = {}
        for col in X.columns:
            original = next((old for old in self.keep_ if X[col].equals(X[old])), None)
            if original is None:
                self.keep_.append(col)
            else:
                self.dropped_[col] = original
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.keep_]


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
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0, ddof=0) if self.with_std else np.ones(X.shape[1])
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_


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
        Z = np.asarray(Z, dtype=float)
        self.mean_ = Z.mean(axis=0)
        _, singular, directions = np.linalg.svd(Z - self.mean_, full_matrices=False)
        variance = singular ** 2
        self.explained_variance_ratio_all_ = variance / variance.sum()
        r = len(singular)
        if self.n_components is None:
            k = r
        elif isinstance(self.n_components, (int, np.integer)) and 1 <= self.n_components <= r:
            k = int(self.n_components)
        elif isinstance(self.n_components, (float, np.floating)) and 0 < self.n_components < 1:
            k = int(np.searchsorted(np.cumsum(self.explained_variance_ratio_all_), self.n_components) + 1)
        else:
            raise ValueError("n_components must be None, an integer in [1, rank], or a fraction in (0, 1)")
        self.n_components_ = k
        self.components_ = directions[:k]
        self.explained_variance_ratio_ = self.explained_variance_ratio_all_[:k]
        return self

    def transform(self, Z) -> np.ndarray:
        return (np.asarray(Z, dtype=float) - self.mean_) @ self.components_.T

    def inverse_transform(self, T) -> np.ndarray:
        return np.asarray(T, dtype=float) @ self.components_ + self.mean_
