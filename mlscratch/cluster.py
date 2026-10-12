"""k-means clustering (Lloyd's algorithm) with k-means++ initialisation."""

import numpy as np

from ._base import check_fitted, check_X


def _sq_dists(X, C):
    """Squared Euclidean distances between rows of X and rows of C, shape (n, k)."""
    return np.maximum((X**2).sum(1)[:, None] - 2 * X @ C.T + (C**2).sum(1)[None, :], 0)


class KMeans:
    """Minimises inertia sum_i ||x_i - mu_{c_i}||^2; keeps the best of ``n_init`` runs."""

    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-6, seed=0):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centers = None

    def _init_plus_plus(self, X, rng):
        """Pick each next centre with probability proportional to D(x)^2."""
        C = [X[rng.integers(len(X))]]
        d2 = _sq_dists(X, np.array(C))[:, 0]
        for _ in range(1, self.n_clusters):
            p = d2 / d2.sum() if d2.sum() > 0 else None  # all points coincide: uniform
            C.append(X[rng.choice(len(X), p=p)])
            d2 = np.minimum(d2, _sq_dists(X, C[-1][None])[:, 0])
        return np.array(C)

    def _lloyd(self, X, C):
        for it in range(1, self.max_iter + 1):
            labels = _sq_dists(X, C).argmin(1)
            new = np.array([X[labels == j].mean(0) if np.any(labels == j) else C[j]
                            for j in range(self.n_clusters)])
            shift = np.sum((new - C) ** 2)
            C = new
            if shift <= self.tol:
                break
        d = _sq_dists(X, C)
        labels = d.argmin(1)
        return C, labels, d[np.arange(len(X)), labels].sum(), it

    def fit(self, X):
        X = check_X(X)
        if not 1 <= self.n_clusters <= len(X):
            raise ValueError("n_clusters must be between 1 and n_samples")
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            run = self._lloyd(X, self._init_plus_plus(X, rng))
            if best is None or run[2] < best[2]:
                best = run
        self.centers, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        check_fitted(self, "centers")
        return _sq_dists(check_X(X), self.centers).argmin(1)

    def fit_predict(self, X):
        return self.fit(X).labels_
