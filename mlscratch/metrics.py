"""Evaluation metrics."""

import numpy as np


def mse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true, y_pred):
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def train_test_split(X, y, test_size=0.2, seed=0):
    """Shuffle and split arrays into train and test parts."""
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(round(test_size * len(X)))
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]


def standardize(X_train, *others):
    """Scale features to zero mean / unit variance using training statistics."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    out = [(X_train - mu) / sd] + [(X - mu) / sd for X in others]
    return out[0] if not others else tuple(out)
