import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def numerical_grad(f, theta, eps=1e-6):
    """Central finite differences of a scalar function f at the vector theta."""
    g = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = eps
        g[i] = (f(theta + e) - f(theta - e)) / (2 * eps)
    return g
