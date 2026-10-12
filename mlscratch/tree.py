"""CART decision trees: greedy binary splits minimising Gini impurity or squared error."""

import numpy as np

from ._base import check_fitted, check_X, check_X_y


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value, feature=None, threshold=None, left=None, right=None):
        self.value, self.feature, self.threshold = value, feature, threshold
        self.left, self.right = left, right


class _CART:
    """Shared tree growing. Subclasses define per-row statistics, impurity and leaf value."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def _best_split(self, X, S):
        """Exhaustive search over features and midpoint thresholds using prefix sums."""
        n, leaf = len(X), self.min_samples_leaf
        total = S.sum(0)
        best_cost, best = self._cost(total, n), None
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="stable")
            x = X[order, j]
            left = np.cumsum(S[order], axis=0)[:-1]       # stats of rows [0..i]
            n_left = np.arange(1, n)
            ok = (x[1:] > x[:-1]) & (n_left >= leaf) & (n - n_left >= leaf)
            if not ok.any():
                continue
            cost = self._cost(left, n_left) + self._cost(total - left, n - n_left)
            cost[~ok] = np.inf
            i = int(np.argmin(cost))
            if cost[i] < best_cost - 1e-12:
                t = (x[i] + x[i + 1]) / 2
                best_cost, best = cost[i], (j, x[i] if t == x[i + 1] else t)  # float guard
        return best

    def _grow(self, X, y, depth):
        node = _Node(self._leaf_value(y))
        if (len(X) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node
        split = self._best_split(X, self._stats(y))
        if split is None:
            return node
        node.feature, node.threshold = split
        mask = X[:, node.feature] <= node.threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_dtype=None)
        self.n_features_ = X.shape[1]
        self.root = self._grow(X, self._encode(y), 0)
        return self

    def _leaves(self, X):
        check_fitted(self, "root")
        X = check_X(X)
        out = []
        for x in X:
            node = self.root
            while node.feature is not None:
                node = node.left if x[node.feature] <= node.threshold else node.right
            out.append(node.value)
        return np.array(out)

    def depth(self):
        def d(node):
            return 0 if node.feature is None else 1 + max(d(node.left), d(node.right))
        return d(self.root)

    def n_leaves(self):
        def c(node):
            return 1 if node.feature is None else c(node.left) + c(node.right)
        return c(self.root)


class DecisionTreeClassifier(_CART):
    """Gini impurity: n_t * (1 - sum_k p_k^2), summed over the two children."""

    def _encode(self, y):
        self.classes_, idx = np.unique(y, return_inverse=True)
        return idx

    def _stats(self, y):
        return np.eye(len(self.classes_))[y]                 # one-hot counts

    @staticmethod
    def _cost(counts, n):
        n = np.asarray(n, float)
        return n - (counts**2).sum(-1) / n

    def _leaf_value(self, y):
        return np.bincount(y, minlength=len(self.classes_)) / len(y)   # class proportions

    def predict_proba(self, X):
        return self._leaves(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class DecisionTreeRegressor(_CART):
    """Squared error: sum (y - mean)^2 = sum y^2 - (sum y)^2 / n, summed over the children."""

    def _encode(self, y):
        return np.asarray(y, float)

    @staticmethod
    def _stats(y):
        c = y - y.mean()                                     # centring avoids cancellation
        return np.c_[c, c**2]

    @staticmethod
    def _cost(stats, n):
        return stats[..., 1] - stats[..., 0] ** 2 / np.asarray(n, float)

    @staticmethod
    def _leaf_value(y):
        return y.mean()

    def predict(self, X):
        return self._leaves(X)
