"""Binary logistic regression with an L2 penalty, trained by gradient descent."""

import numpy as np

from ._base import check_fitted, check_X, check_X_y


def sigmoid(z):
    """Numerically stable 1 / (1 + e^-z)."""
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


class LogisticRegression:
    """Minimises mean cross-entropy + (l2/2) ||w||^2.

    Equivalent to ``sklearn.linear_model.LogisticRegression(C=C)`` with l2 = 1 / (C n).
    """

    def __init__(self, learning_rate=0.5, n_iters=10000, l2=0.0, tol=1e-7):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l2 = l2
        self.tol = tol
        self.weights = None

    def loss(self, X, y, w, b):
        z = X @ w + b
        return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * self.l2 * (w @ w)

    def gradient(self, X, y, w, b):
        r = sigmoid(X @ w + b) - y
        return X.T @ r / len(y) + self.l2 * w, r.mean()

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_dtype=None)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("LogisticRegression expects exactly two classes")
        t = (y == self.classes_[1]).astype(float)
        w, b = np.zeros(X.shape[1]), 0.0
        for self.n_iter_ in range(1, self.n_iters + 1):
            dw, db = self.gradient(X, t, w, b)
            w -= self.learning_rate * dw
            b -= self.learning_rate * db
            if np.sqrt(dw @ dw + db**2) < self.tol:
                break
        self.weights, self.bias = w, b
        return self

    def decision_function(self, X):
        check_fitted(self, "weights")
        return check_X(X) @ self.weights + self.bias

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.c_[1 - p, p]

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
