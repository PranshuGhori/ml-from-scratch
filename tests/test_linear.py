import numpy as np
import pytest
from sklearn import linear_model

from conftest import numerical_grad
from mlscratch import LinearRegression, LinearRegressionGD, LocallyWeightedRegression, Ridge


@pytest.fixture
def data(rng):
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.5, -2.0, 0.5, 3.0]) + 0.7 + 0.1 * rng.normal(size=200)
    return X, y


def test_normal_equation_matches_sklearn(data):
    X, y = data
    ours, ref = LinearRegression().fit(X, y), linear_model.LinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.weights, ref.coef_, atol=1e-10)
    np.testing.assert_allclose(ours.bias, ref.intercept_, atol=1e-10)


def test_normal_equation_rank_deficient_gives_min_norm(rng):
    x = rng.normal(size=(50, 1))
    X = np.c_[x, x]  # duplicated column: X^T X is singular
    y = 3 * x[:, 0] + 1
    m = LinearRegression().fit(X, y)
    np.testing.assert_allclose(m.weights, [1.5, 1.5], atol=1e-10)
    np.testing.assert_allclose(m.predict(X), y, atol=1e-10)


def test_gd_converges_to_closed_form(data):
    X, y = data
    gd = LinearRegressionGD(learning_rate=0.1, n_iters=20000, tol=1e-10).fit(X, y)
    ne = LinearRegression().fit(X, y)
    np.testing.assert_allclose(gd.weights, ne.weights, atol=1e-8)
    np.testing.assert_allclose(gd.bias, ne.bias, atol=1e-8)
    assert np.all(np.diff(gd.loss_history) <= 1e-15)  # monotone for a small enough step


def test_gd_gradient_matches_finite_differences(data):
    X, y = data
    m = LinearRegressionGD()
    theta = np.random.default_rng(1).normal(size=X.shape[1] + 1)
    dw, db = m.gradient(X, y, theta[:-1], theta[-1])
    num = numerical_grad(lambda t: m.loss(X, y, t[:-1], t[-1]), theta)
    np.testing.assert_allclose(np.r_[dw, db], num, rtol=1e-6, atol=1e-8)


def test_original_toy_example():
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)
    m = LinearRegression().fit(X, y)
    np.testing.assert_allclose([m.bias, m.weights[0]], [0, 2], atol=1e-12)
    gd = LinearRegressionGD(learning_rate=0.05, n_iters=50000, tol=1e-12).fit(X, y)
    np.testing.assert_allclose(gd.predict(X), y, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
def test_ridge_matches_sklearn(data, alpha):
    X, y = data
    ours, ref = Ridge(alpha).fit(X, y), linear_model.Ridge(alpha=alpha).fit(X, y)
    np.testing.assert_allclose(ours.weights, ref.coef_, atol=1e-10)
    np.testing.assert_allclose(ours.bias, ref.intercept_, atol=1e-10)


def test_ridge_shrinks_weights(data):
    X, y = data
    norms = [np.linalg.norm(Ridge(a).fit(X, y).weights) for a in [0, 10, 1000]]
    assert norms[0] > norms[1] > norms[2]


def test_lwr_matches_explicit_formula():
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([1, 3, 2, 5, 7], dtype=float)
    w = np.exp(-((X - 3) ** 2) / 2)
    Xb, W = np.c_[np.ones(5), X], np.diag(w)
    theta = np.linalg.inv(Xb.T @ W @ Xb) @ Xb.T @ W @ y
    pred = LocallyWeightedRegression(tau=1).fit(X, y).predict([3.0])
    np.testing.assert_allclose(pred, [theta[0] + 3 * theta[1]], rtol=1e-12)


def test_lwr_large_tau_is_ols(data):
    X, y = data
    lwr = LocallyWeightedRegression(tau=1e6).fit(X, y)
    np.testing.assert_allclose(lwr.predict(X[:10]), LinearRegression().fit(X, y).predict(X[:10]),
                               atol=1e-6)


def test_lwr_fits_nonlinear_curve(rng):
    x = np.sort(rng.uniform(0, 2 * np.pi, 300))
    y = np.sin(x) + 0.05 * rng.normal(size=300)
    pred = LocallyWeightedRegression(tau=0.3).fit(x, y).predict(x)
    assert np.sqrt(np.mean((pred - np.sin(x)) ** 2)) < 0.05


@pytest.mark.parametrize("Model", [LinearRegression, LinearRegressionGD, Ridge,
                                   LocallyWeightedRegression])
def test_shapes_and_1d_input(Model):
    x = np.linspace(0, 1, 20)
    m = Model().fit(x, 2 * x + 1)          # 1-D X is a single feature
    assert m.predict(x).shape == (20,)
    assert m.predict(x[:, None]).shape == (20,)


@pytest.mark.parametrize("Model", [LinearRegression, LinearRegressionGD, Ridge,
                                   LocallyWeightedRegression])
def test_errors(Model):
    with pytest.raises(RuntimeError):
        Model().predict([[1.0]])
    with pytest.raises(ValueError):
        Model().fit(np.ones((5, 2)), np.ones(4))
