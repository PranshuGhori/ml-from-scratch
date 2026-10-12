"""Classic machine learning algorithms in plain NumPy."""

from .cluster import KMeans
from .decomposition import PCA
from .linear import LinearRegression, LinearRegressionGD, LocallyWeightedRegression, Ridge
from .logistic import LogisticRegression
from .tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "KMeans",
    "LinearRegression",
    "LinearRegressionGD",
    "LocallyWeightedRegression",
    "LogisticRegression",
    "PCA",
    "Ridge",
]
__version__ = "0.1.0"
