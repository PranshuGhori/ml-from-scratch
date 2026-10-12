import numpy as np
import pytest
from sklearn import tree
from sklearn.datasets import load_iris, make_classification, make_regression

from mlscratch import DecisionTreeClassifier, DecisionTreeRegressor


@pytest.mark.parametrize("params", [dict(max_depth=3), dict(max_depth=6, min_samples_leaf=5)])
def test_classifier_matches_sklearn(params):
    X, y = make_classification(n_samples=400, n_features=6, random_state=0)
    ours = DecisionTreeClassifier(**params).fit(X, y)
    ref = tree.DecisionTreeClassifier(random_state=0, **params).fit(X, y)
    # Gini often ties between features and sklearn breaks ties at random, so compare the
    # fitted partition of the training set rather than predictions on unseen points.
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X))
    assert ours.depth() == ref.get_depth()
    assert ours.n_leaves() == ref.get_n_leaves()


@pytest.mark.parametrize("params", [dict(max_depth=4), dict(max_depth=8, min_samples_leaf=3)])
def test_regressor_matches_sklearn(params):
    X, y = make_regression(n_samples=400, n_features=5, noise=5, random_state=0)
    ours = DecisionTreeRegressor(**params).fit(X, y)
    ref = tree.DecisionTreeRegressor(random_state=0, **params).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-10, atol=1e-10)
    assert ours.n_leaves() == ref.get_n_leaves()


@pytest.mark.parametrize("Model", [DecisionTreeClassifier, DecisionTreeRegressor])
def test_size_limits_are_respected(Model):
    X, y = make_classification(n_samples=400, n_features=6, random_state=1)
    m = Model(min_samples_split=30, min_samples_leaf=7).fit(X, y)

    def walk(node, rows):
        if node.feature is None:
            assert rows.sum() >= 7
            return
        assert rows.sum() >= 30
        go_left = X[:, node.feature] <= node.threshold
        walk(node.left, rows & go_left)
        walk(node.right, rows & ~go_left)

    walk(m.root, np.ones(len(X), bool))


def test_unlimited_tree_memorises_training_set():
    X, y = load_iris(return_X_y=True)
    X = X + 1e-6 * np.arange(len(X))[:, None]  # break exact duplicates with different labels
    assert np.array_equal(DecisionTreeClassifier().fit(X, y).predict(X), y)


def test_regressor_large_offset_target_is_stable(rng):
    x = rng.uniform(size=200)
    y = 1e8 + (x > 0.5)  # a step on a large offset
    m = DecisionTreeRegressor(max_depth=1).fit(x, y)
    np.testing.assert_allclose(m.predict([0.1, 0.9]), [1e8, 1e8 + 1])


def test_edge_cases():
    m = DecisionTreeClassifier().fit(np.ones((5, 1)), [0, 1, 0, 1, 1])  # no valid split
    assert m.depth() == 0 and m.predict([[1.0]])[0] == 1
    m = DecisionTreeClassifier(max_depth=0).fit([[0.0], [1.0]], ["a", "b"])
    assert m.depth() == 0
    with pytest.raises(RuntimeError):
        DecisionTreeRegressor().predict([[0.0]])
