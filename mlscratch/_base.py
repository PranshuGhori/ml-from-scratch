"""Small shared helpers: input validation and the not-fitted check."""

import numpy as np


def check_X(X):
    """Return X as a 2-D float array. A 1-D input is treated as a single feature."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("X has no samples")
    return X


def check_X_y(X, y, y_dtype=float):
    """Validate a feature matrix and a 1-D target of matching length."""
    X = check_X(X)
    y = np.asarray(y) if y_dtype is None else np.asarray(y, dtype=y_dtype)
    if y.ndim != 1:
        y = y.ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}")
    return X, y


def check_fitted(model, attr):
    if getattr(model, attr, None) is None:
        raise RuntimeError(f"{type(model).__name__} is not fitted; call fit() first")
