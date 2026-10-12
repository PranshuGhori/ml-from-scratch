"""Principal component analysis via the SVD of the centred data."""

import numpy as np

from ._base import check_fitted, check_X


class PCA:
    """X - mean = U S V^T; components are the top rows of V^T, variances S^2 / (n - 1)."""

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components = None

    def fit(self, X):
        X = check_X(X)
        n, d = X.shape
        k = min(n, d) if self.n_components is None else self.n_components
        if not 1 <= k <= min(n, d):
            raise ValueError("n_components must be between 1 and min(n_samples, n_features)")
        self.mean_ = X.mean(0)
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        # SVD signs are arbitrary; make the largest-magnitude loading of each component positive.
        signs = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(1)])
        Vt *= signs[:, None]
        var = S**2 / max(n - 1, 1)
        self.components = Vt[:k]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / var.sum()
        return self

    def transform(self, X):
        check_fitted(self, "components")
        return (check_X(X) - self.mean_) @ self.components.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        check_fitted(self, "components")
        return np.asarray(Z, float) @ self.components + self.mean_
