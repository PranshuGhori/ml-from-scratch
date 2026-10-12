"""Linear regression: gradient descent, normal equation, ridge, locally weighted."""

import numpy as np

from ._base import check_fitted, check_X, check_X_y


def _add_bias(X):
    return np.c_[np.ones(len(X)), X]


class _Linear:
    weights = None
    bias = None

    def predict(self, X):
        check_fitted(self, "weights")
        return check_X(X) @ self.weights + self.bias


class LinearRegressionGD(_Linear):
    """Batch gradient descent on J = 1/(2n) ||Xw + b - y||^2. Standardise X first."""

    def __init__(self, learning_rate=0.1, n_iters=5000, tol=1e-8):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.tol = tol

    def loss(self, X, y, w, b):
        return 0.5 * np.mean((X @ w + b - y) ** 2)

    def gradient(self, X, y, w, b):
        r = X @ w + b - y
        return X.T @ r / len(y), r.mean()

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        w, b = np.zeros(X.shape[1]), 0.0
        self.loss_history = []
        for self.n_iter_ in range(1, self.n_iters + 1):
            dw, db = self.gradient(X, y, w, b)
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
            self.loss_history.append(self.loss(X, y, w, b))
            if np.sqrt(dw @ dw + db**2) < self.tol:
                break
        self.weights, self.bias = w, b
        return self


class LinearRegression(_Linear):
    """Normal equation theta = X^+ y, solved with lstsq (SVD) rather than an inverse."""

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        theta = np.linalg.lstsq(_add_bias(X), y, rcond=None)[0]
        self.bias, self.weights = theta[0], theta[1:]
        return self


class Ridge(_Linear):
    """Minimises ||y - Xw - b||^2 + alpha ||w||^2; the bias is not penalised."""

    def __init__(self, alpha=1.0):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        mx, my = X.mean(axis=0), y.mean()
        Xc = X - mx
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.weights = np.linalg.lstsq(A, Xc.T @ (y - my), rcond=None)[0]
        self.bias = my - mx @ self.weights
        return self


class LocallyWeightedRegression:
    """Solves theta = (X^T W X)^{-1} X^T W y for every query, w_i = exp(-||x_i - x||^2 / 2tau^2)."""

    def __init__(self, tau=1.0):
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau
        self.X_ = None

    def fit(self, X, y):
        self.X_, self.y_ = check_X_y(X, y)
        return self

    def _predict_one(self, x):
        d2 = np.sum((self.X_ - x) ** 2, axis=1)
        s = np.exp(-d2 / (4 * self.tau**2))  # sqrt of the kernel weight
        theta = np.linalg.lstsq(_add_bias(self.X_) * s[:, None], self.y_ * s, rcond=None)[0]
        return theta[0] + x @ theta[1:]

    def predict(self, X):
        check_fitted(self, "X_")
        X = check_X(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("X has a different number of features than the training data")
        return np.array([self._predict_one(x) for x in X])
