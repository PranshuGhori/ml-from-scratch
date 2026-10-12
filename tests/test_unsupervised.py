import numpy as np
import pytest
from sklearn import cluster, decomposition
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from mlscratch import KMeans, PCA


@pytest.fixture
def blobs():
    return make_blobs(n_samples=600, centers=4, cluster_std=1.0, random_state=3)


def test_kmeans_recovers_blobs(blobs):
    X, y = blobs
    km = KMeans(n_clusters=4).fit(X)
    assert adjusted_rand_score(y, km.labels_) > 0.95
    ref = cluster.KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    assert km.inertia_ <= ref.inertia_ * 1.001
    np.testing.assert_array_equal(km.predict(X), km.labels_)


def test_kmeans_inertia_is_consistent(blobs):
    X, _ = blobs
    km = KMeans(n_clusters=3, seed=1).fit(X)
    inertia = sum(np.sum((X[km.labels_ == j] - c) ** 2) for j, c in enumerate(km.centers))
    np.testing.assert_allclose(km.inertia_, inertia)
    for j, c in enumerate(km.centers):  # converged: each centre is the mean of its cluster
        np.testing.assert_allclose(c, X[km.labels_ == j].mean(0), atol=1e-8)


def test_kmeans_edge_cases():
    X = np.zeros((5, 2))  # identical points
    assert KMeans(n_clusters=2, n_init=1).fit(X).inertia_ == 0
    x = np.r_[np.zeros(10), np.ones(10) * 10]  # 1-D input
    assert len(set(KMeans(n_clusters=2).fit_predict(x))) == 2
    with pytest.raises(ValueError):
        KMeans(n_clusters=10).fit(np.ones((3, 2)))


@pytest.fixture
def correlated(rng):
    return rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))


def test_pca_matches_sklearn(correlated):
    X = correlated
    ours, ref = PCA(3).fit(X), decomposition.PCA(3).fit(X)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-10)
    np.testing.assert_allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_,
                               rtol=1e-10)
    # components are defined up to sign
    signs = np.sign(np.sum(ours.components * ref.components_, axis=1))
    np.testing.assert_allclose(ours.components, signs[:, None] * ref.components_, atol=1e-10)
    np.testing.assert_allclose(ours.transform(X), ref.transform(X) * signs, atol=1e-10)


def test_pca_full_rank_roundtrip(correlated):
    X = correlated
    p = PCA().fit(X)
    np.testing.assert_allclose(p.inverse_transform(p.transform(X)), X, atol=1e-10)
    np.testing.assert_allclose(p.components @ p.components.T, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(p.explained_variance_ratio_.sum(), 1)
    assert np.all(np.diff(p.explained_variance_) <= 0)


def test_pca_wide_and_invalid(rng):
    X = rng.normal(size=(4, 10))  # more features than samples
    assert PCA().fit(X).components.shape == (4, 10)
    with pytest.raises(ValueError):
        PCA(5).fit(X)
