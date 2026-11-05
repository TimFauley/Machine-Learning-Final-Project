"""K-means and fuzzy C-means ported from Homework 1/HW1.ipynb.

Random training rows initialize K-means; FCM starts with random normalized
memberships. Restarts minimize the training SSE / fuzzy objective, never a
label-based score. Prediction uses the fitted centres without updating them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass
class ClusterResult:
    centres: np.ndarray
    labels: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    objective_history: np.ndarray
    memberships: np.ndarray | None = None


def _matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.size or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite 2-D matrix")
    return X


def _settings(X, k, max_iter, tol):
    X = _matrix(X)
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= len(X):
        raise ValueError("k must be an integer between 1 and the number of rows")
    if max_iter < 1 or tol <= 0:
        raise ValueError("max_iter and tol must be positive")
    return X


def squared_distances(X, centres):
    """Euclidean squared distances, with direct subtraction near cancellation."""
    X, centres = _matrix(X), _matrix(centres)
    if X.shape[1] != centres.shape[1]:
        raise ValueError("X and centres must have the same feature count")
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    cc = np.einsum("ij,ij->i", centres, centres)[None, :]
    D = xx + cc - 2 * X @ centres.T
    rows, cols = np.where(D <= 1e-12 * np.maximum(xx + cc, 1))
    if len(rows):
        delta = X[rows] - centres[cols]
        D[rows, cols] = np.einsum("ij,ij->i", delta, delta)
    return np.maximum(D, 0)


def predict_kmeans(X, centres):
    """Assign new rows to their nearest training centroid."""
    return squared_distances(X, centres).argmin(axis=1)


def kmeanx(X, k, max_iter=300, tol=1e-5, seed=0):
    """One HW1 Lloyd run; stop on unchanged labels or maximum centre shift."""
    X = _settings(X, k, max_iter, tol)
    rng = np.random.default_rng(seed)
    centres = X[rng.choice(len(X), k, replace=False)].copy()
    old_labels = None
    history = []
    converged = False
    for iteration in range(1, max_iter + 1):
        labels = predict_kmeans(X, centres)
        updated = np.array([X[labels == j].mean(axis=0) if np.any(labels == j)
                            else X[rng.integers(len(X))] for j in range(k)])
        shift = np.linalg.norm(updated - centres, axis=1).max()
        stable = old_labels is not None and np.array_equal(labels, old_labels)
        centres = updated
        final_labels = predict_kmeans(X, centres)
        objective = float(((X - centres[final_labels]) ** 2).sum())
        history.append(objective)
        if stable or shift < tol:
            converged = True
            break
        old_labels = labels
    return ClusterResult(centres, final_labels, objective, iteration, converged, np.array(history))


def _best_run(run, n_init, seed):
    if not isinstance(n_init, (int, np.integer)) or n_init < 1:
        raise ValueError("n_init must be a positive integer")
    rng = np.random.default_rng(seed)
    results = [run(int(s)) for s in rng.integers(0, 2**32, size=n_init)]
    return min(results, key=lambda result: result.objective)


def best_kmeanx(X, k, max_iter=300, n_init=10, tol=1e-5, seed=0):
    """Choose the smallest training SSE across reproducible random restarts."""
    return _best_run(lambda s: kmeanx(X, k, max_iter, tol, s), n_init, seed)


def update_centre(X, U, m=2):
    """HW1 weighted mean: centre_j = sum(u_ij**m * x_i) / sum(u_ij**m)."""
    if m <= 1:
        raise ValueError("m must be greater than 1")
    weights = np.asarray(U, dtype=float) ** m
    return weights.T @ np.asarray(X, dtype=float) / weights.sum(axis=0)[:, None]


def update_membership(X, centre, m=2):
    """HW1 inverse-distance ratio update, vectorized and normalized in log space.

    Retains HW1's distance floor of 1e-10 and exponent 2/(m-1).
    Normalized inverse powers equal 1 / sum_k((d_j/d_k)**(2/(m-1))).
    """
    if m <= 1:
        raise ValueError("m must be greater than 1")
    distance = np.maximum(np.sqrt(squared_distances(X, centre)), 1e-10)
    log_weights = -2 / (m - 1) * np.log(distance)
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def predict_fcm(X, centres, m=2):
    """Memberships for new rows using fixed training centres."""
    return update_membership(X, centres, m)


def fcm(X, k, m=2, max_iter=500, tol=1e-5, seed=0):
    """One HW1 FCM run; convergence uses max absolute membership change."""
    X = _settings(X, k, max_iter, tol)
    if m <= 1:
        raise ValueError("m must be greater than 1")
    rng = np.random.default_rng(seed)
    U = rng.random((len(X), k))
    U /= U.sum(axis=1, keepdims=True)
    history = []
    converged = False
    for iteration in range(1, max_iter + 1):
        centres = update_centre(X, U, m)
        updated = update_membership(X, centres, m)
        change = np.abs(updated - U).max()
        U = updated
        objective = float((U ** m * squared_distances(X, centres)).sum())
        history.append(objective)
        if change < tol:
            converged = True
            break
    return ClusterResult(centres, U.argmax(axis=1), objective, iteration, converged,
                         np.array(history), U)


def best_fcm(X, k, m=2, max_iter=500, n_init=10, tol=1e-5, seed=0):
    """Choose the smallest training fuzzy objective across random restarts."""
    return _best_run(lambda s: fcm(X, k, m, max_iter, tol, s), n_init, seed)


def contingency_table(y_true, clusters, labels=None, cluster_ids=None):
    """Counts with true labels as rows and cluster IDs as columns, including zeros."""
    table = pd.crosstab(pd.Series(np.asarray(y_true), name="Activity"),
                        pd.Series(np.asarray(clusters), name="Cluster"))
    return table.reindex(index=labels, columns=cluster_ids, fill_value=0)


def hungarian_mapping(y_true, clusters):
    """One-to-one cluster -> label map maximizing counts on the supplied split."""
    table = contingency_table(y_true, clusters)
    rows, cols = linear_sum_assignment(-table.to_numpy())
    return {table.columns[j]: table.index[i] for i, j in zip(rows, cols)}


def matched_accuracy(y_true, clusters, mapping=None):
    """Hungarian accuracy; pass the TRAIN mapping to evaluate held-out windows.

    Without a mapping, this optimizes the permutation on the supplied labels
    (a descriptive clustering score, not a fixed-map predictive accuracy).
    Unmatched clusters count as incorrect.
    """
    y_true, clusters = np.asarray(y_true), np.asarray(clusters)
    if mapping is None:
        mapping = hungarian_mapping(y_true, clusters)
    return float(np.mean([cluster in mapping and mapping[cluster] == truth
                          for truth, cluster in zip(y_true, clusters)]))


def clustering_metrics(y_true, clusters, mapping=None):
    return {"ARI": adjusted_rand_score(y_true, clusters),
            "NMI": normalized_mutual_info_score(y_true, clusters),
            "matched accuracy": matched_accuracy(y_true, clusters, mapping)}


def partition_coefficient(U):
    """Mean sum of squared memberships; ranges from 1/k to 1."""
    U = np.asarray(U)
    return float(np.mean(np.sum(U ** 2, axis=1)))


def mean_max_membership(y_true, U):
    """Per-true-class mean of each window's largest membership."""
    return pd.Series(np.asarray(U).max(axis=1)).groupby(np.asarray(y_true)).mean().rename("mean max membership")
