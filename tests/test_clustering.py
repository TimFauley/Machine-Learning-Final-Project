"""Small, known-data checks for the HW1 ports and evaluation helpers."""
import sys
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clustering import (best_kmeanx, best_fcm, kmeanx, fcm, predict_kmeans,
                        predict_fcm, update_centre, update_membership,
                        contingency_table, hungarian_mapping, matched_accuracy,
                        partition_coefficient, mean_max_membership)


def blobs():
    return make_blobs(n_samples=180, centers=[[-6, -6], [0, 6], [6, -3]],
                      cluster_std=0.5, random_state=12)


def test_kmeans_agrees_with_sklearn():
    X, y = blobs()
    ours = best_kmeanx(X, 3, n_init=8, seed=42)
    reference = KMeans(3, init="random", n_init=8, random_state=42).fit(X)
    assert abs(adjusted_rand_score(y, ours.labels) -
               adjusted_rand_score(y, reference.labels_)) < 1e-6
    assert ours.converged
    assert_allclose(ours.objective, reference.inertia_, rtol=1e-6)
    assert_allclose(predict_kmeans(X, ours.centres), ours.labels)
    assert_allclose(best_kmeanx(X, 3, n_init=8, seed=42).centres, ours.centres)


def test_fcm_memberships_and_convergence():
    X, y = blobs()
    result = best_fcm(X, 3, n_init=4, seed=42)
    assert result.converged and result.n_iter < 500
    assert_allclose(result.memberships.sum(axis=1), 1, atol=1e-12)
    assert np.all((result.memberships >= 0) & (result.memberships <= 1))
    assert adjusted_rand_score(y, result.labels) > 0.99
    assert np.all(np.diff(result.objective_history) <= 1e-8)
    assert_allclose(predict_fcm(X, result.centres), result.memberships)
    next_u = update_membership(X, update_centre(X, result.memberships), 2)
    assert np.max(np.abs(next_u - result.memberships)) < 1e-5
    assert partition_coefficient(result.memberships) > 0.95


def test_fcm_matches_homework_equations():
    X = np.array([[0., 0.], [2., 3.], [6., 1.]])
    U = np.array([[.8, .2], [.4, .6], [.1, .9]])
    for m in (1.5, 2., 3.):
        centres = np.array([np.average(X, axis=0, weights=U[:, j] ** m) for j in range(2)])
        assert_allclose(update_centre(X, U, m), centres)
        expected = []
        for x in X:
            d = np.maximum(np.linalg.norm(x - centres, axis=1), 1e-10)
            expected.append([1 / np.sum((dj / d) ** (2 / (m - 1))) for dj in d])
        assert_allclose(update_membership(X, centres, m), expected)


def test_zero_distances_and_coincident_centres():
    X = np.array([[0., 0.], [4., 4.]])
    U = predict_fcm(X, np.array([[0., 0.], [0., 0.], [4., 4.]]))
    assert np.isfinite(U).all()
    assert_allclose(U.sum(axis=1), 1)
    assert_allclose(U[0, :2], [.5, .5])
    assert_allclose(U[1, 2], 1)


def test_hungarian_accuracy_and_fixed_test_mapping():
    y = np.array([1, 1, 1, 2, 2, 2])
    clusters = np.array([9, 9, 4, 4, 4, 4])
    assert_allclose(contingency_table(y, clusters).to_numpy(), [[1, 2], [3, 0]])
    mapping = hungarian_mapping(y, clusters)
    assert mapping == {9: 1, 4: 2}
    assert matched_accuracy(y, clusters) == 5 / 6
    assert matched_accuracy([1, 2], [4, 9], mapping) == 0
    assert matched_accuracy([1, 2], [4, 9]) == 1
    assert matched_accuracy([1, 2, 2], [9, 4, 7], mapping) == 2 / 3


def test_membership_summaries():
    U = np.array([[.8, .2], [.6, .4], [.1, .9]])
    assert_allclose(partition_coefficient(U), (.68 + .52 + .82) / 3)
    assert_allclose(mean_max_membership([1, 1, 2], U), [.7, .9])


def test_iteration_cap():
    X, _ = blobs()
    for result in (kmeanx(X, 3, max_iter=1), fcm(X, 3, max_iter=1)):
        assert result.n_iter == 1
        assert np.isfinite(result.objective)
