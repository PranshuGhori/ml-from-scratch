import numpy as np
import pytest
from sklearn import linear_model
from sklearn.datasets import make_classification

from conftest import numerical_grad
from mlscratch import LogisticRegression
from mlscratch.logistic import sigmoid


@pytest.fixture
def data():
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=0)
    return (X - X.mean(0)) / X.std(0), y


@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_matches_sklearn(data, C):
    X, y = data
    ours = LogisticRegression(l2=1 / (C * len(y)), n_iters=200000, tol=1e-9).fit(X, y)
    ref = linear_model.LogisticRegression(C=C, tol=1e-10, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(ours.weights, ref.coef_[0], atol=1e-5)
    np.testing.assert_allclose(ours.bias, ref.intercept_[0], atol=1e-5)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-6)
    assert np.array_equal(ours.predict(X), ref.predict(X))


@pytest.mark.parametrize("l2", [0.0, 0.1])
def test_gradient_matches_finite_differences(data, l2):
    X, y = data
    m = LogisticRegression(l2=l2)
    theta = np.random.default_rng(1).normal(size=X.shape[1] + 1)
    dw, db = m.gradient(X, y, theta[:-1], theta[-1])
    num = numerical_grad(lambda t: m.loss(X, y, t[:-1], t[-1]), theta)
    np.testing.assert_allclose(np.r_[dw, db], num, rtol=1e-6, atol=1e-8)


def test_string_labels_and_proba_shape(data):
    X, y = data
    labels = np.where(y == 1, "spam", "ham")
    m = LogisticRegression(l2=0.01).fit(X, labels)
    assert set(m.predict(X)) <= {"ham", "spam"}
    P = m.predict_proba(X)
    assert P.shape == (len(X), 2)
    np.testing.assert_allclose(P.sum(1), 1)


def test_sigmoid_is_stable():
    z = np.array([-1000.0, -30.0, 0.0, 30.0, 1000.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s[[0, 2, 4]], [0, 0.5, 1])
    np.testing.assert_allclose(s + sigmoid(-z), 1)


def test_rejects_multiclass():
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.ones((6, 1)), [0, 1, 2, 0, 1, 2])
