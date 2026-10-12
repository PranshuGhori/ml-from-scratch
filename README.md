# ml-from-scratch

Classic machine learning algorithms written from scratch in NumPy, each one tested against scikit-learn.

| Algorithm | File | Key idea |
| --- | --- | --- |
| `LinearRegressionGD` | `mlscratch/linear.py` | batch gradient descent on the squared error |
| `LinearRegression` | `mlscratch/linear.py` | normal equation, solved with `lstsq` (pseudoinverse) |
| `Ridge` | `mlscratch/linear.py` | closed form with an L2 penalty, bias unpenalised |
| `LocallyWeightedRegression` | `mlscratch/linear.py` | a Gaussian-weighted least-squares fit per query point |
| `LogisticRegression` | `mlscratch/logistic.py` | cross-entropy + L2, gradient descent, `predict_proba` |
| `KMeans` | `mlscratch/cluster.py` | Lloyd's algorithm with k-means++ seeding |
| `PCA` | `mlscratch/decomposition.py` | SVD of the centred data |
| `DecisionTreeClassifier` / `Regressor` | `mlscratch/tree.py` | CART: greedy splits on Gini or squared error |

Every model follows the same interface: `fit(X, y)` (or `fit(X)` for k-means and PCA) returns the
model, then `predict(X)` or `transform(X)`.
NumPy is the only runtime dependency.

## Quickstart

```bash
pip install -e .
```

```python
import numpy as np
from mlscratch import LinearRegression

X = np.array([[1], [2], [3], [4], [5]], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)
model = LinearRegression().fit(X, y)
print(model.weights, model.bias)   # [2.] and a bias of ~1e-15
```

The notebook [`Machine_learning_from_scratch.ipynb`](Machine_learning_from_scratch.ipynb) is a short tour:
each model on a toy dataset, a few plots, and a comparison with scikit-learn.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The tests check each model against scikit-learn or a closed-form answer, verify the gradient-descent
gradients with finite differences, and cover edge cases (1-D input, singular `X^T X`, identical points,
more features than samples). CI runs them on Python 3.11 and 3.12.

## Results

From the notebook run (80/20 split, best of 5 fits, Apple M5).

| Model | Dataset | Metric | mlscratch | scikit-learn | Fit time (ms) ours / sklearn |
| --- | --- | --- | --- | --- | --- |
| Linear, gradient descent | diabetes | test RMSE | 53.2601 | 53.2601 | 212.23 / 0.17 |
| Linear, normal equation | diabetes | test RMSE | 53.2601 | 53.2601 | 0.05 / 0.17 |
| Ridge, alpha = 10 | diabetes | test RMSE | 53.1368 | 53.1368 | 0.03 / 0.16 |
| Logistic regression, C = 1 | breast cancer | test accuracy | 0.9912 | 0.9912 | 98.92 / 1.63 |
| k-means, k = 4 | blobs | inertia | 1205.1708 | 1205.1708 | 3.04 / 3.59 |
| PCA, 2 of 64 components | digits | explained variance | 0.2851 | 0.2851 | 3.24 / 0.38 |
| Decision tree, depth 4 | wine | test accuracy | 0.8889 | 0.8889 | 2.16 / 0.33 |
| Regression tree, depth 4 | diabetes | test RMSE | 61.4789 | 61.4789 | 1.91 / 0.35 |

Gradient descent is slow by design: it takes about 20,000 full-batch steps to reach the normal-equation
weights (max difference 7.8e-07).

---

## Linear regression

Linear regression models the relationship between input features and the target as:

$$
\hat{y} = Xw + b, \qquad X \in \mathbb{R}^{n \times d}, \quad w \in \mathbb{R}^{d}, \quad b \in \mathbb{R}
$$

where $X$ is the feature matrix, $w$ the weight vector, $b$ the bias and $\hat{y}$ the prediction.
The objective is to learn $w$ and $b$ so that predictions are as close as possible to the true targets.

### Mean squared error

To measure prediction error, linear regression uses the mean squared error:

$$
J(w,b) = \frac{1}{2n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
$$

Larger errors are penalised more heavily because the difference is squared. The factor $\tfrac{1}{2}$
does not move the minimum; it only cancels the 2 that appears when differentiating.

### Gradient descent

Gradient descent minimises $J$ iteratively. At each step the predictions are $\hat{y} = Xw + b$ and the
gradients are:

$$
\frac{\partial J}{\partial w} = \frac{1}{n}X^T(\hat{y} - y), \qquad
\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)
$$

The parameters are then updated with learning rate $\alpha$:

$$
w := w - \alpha \frac{\partial J}{\partial w}, \qquad
b := b - \alpha \frac{\partial J}{\partial b}
$$

Gradient descent avoids matrix inversion and scales to large datasets, but it needs standardised
features and many steps.

### Normal equation

The normal equation computes the optimal parameters directly. With a column of ones prepended to $X$:

$$
\theta = (X^T X)^{-1} X^T y, \qquad
\theta = \begin{bmatrix} b \\ w \end{bmatrix}
$$

This is exact for small problems but expensive when the number of features is very large. A numerically
more stable version uses the Moore-Penrose pseudoinverse, which also handles a singular $X^T X$:

$$
\theta = X^{+} y
$$

`LinearRegression` computes this with `np.linalg.lstsq`, which solves the problem through an SVD.

### Ridge regression

Adding an L2 penalty shrinks the weights and keeps the problem well conditioned:

$$
\min_{w,b} \; \lVert y - Xw - b \rVert^2 + \lambda \lVert w \rVert^2
\quad\Rightarrow\quad
w = (X_c^T X_c + \lambda I)^{-1} X_c^T y_c, \qquad b = \bar{y} - \bar{x}^T w
$$

where $X_c$, $y_c$ are centred so that the bias is not penalised.

## Locally weighted regression

Locally weighted regression is non-parametric: instead of one global line it fits a local model around
each query point $x$, giving more importance to nearby training points through a Gaussian kernel:

$$
w^{(i)} = \exp\left(-\frac{\lVert x^{(i)} - x \rVert^2}{2\tau^2}\right)
$$

where $\tau$ is the bandwidth that controls how local the fit is. With $W = \mathrm{diag}(w^{(1)}, \dots, w^{(n)})$:

$$
\theta = (X^T W X)^{-1} X^T W y, \qquad \hat{y} = \theta^T x
$$

This captures local nonlinear behaviour, but a new weighted regression is solved for every query point.
As $\tau \to \infty$ every weight tends to 1 and the model becomes ordinary least squares.

## Logistic regression

For binary labels $y \in \{0, 1\}$ the model outputs a probability through the sigmoid
$\sigma(z) = 1 / (1 + e^{-z})$ and minimises the cross-entropy with an L2 penalty:

$$
p = \sigma(Xw + b), \qquad
J(w,b) = -\frac{1}{n}\sum_{i=1}^{n}\big[y_i \log p_i + (1-y_i)\log(1-p_i)\big] + \frac{\lambda}{2}\lVert w \rVert^2
$$

$$
\frac{\partial J}{\partial w} = \frac{1}{n}X^T(p - y) + \lambda w, \qquad
\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(p_i - y_i)
$$

The gradient has the same form as in linear regression. $J$ is convex, so gradient descent reaches the
global minimum; $\lambda = 1/(Cn)$ gives the same solution as scikit-learn's `C`.

## k-means

k-means partitions the data into $k$ clusters by minimising the inertia

$$
\sum_{i=1}^{n} \lVert x^{(i)} - \mu_{c_i} \rVert^2
$$

alternating two steps until the centres stop moving: assign each point to its nearest centre, then move
each centre to the mean of its points. k-means++ seeds the centres by picking each new one with
probability proportional to its squared distance from the nearest centre already chosen.

## Principal component analysis

PCA finds the orthogonal directions of maximum variance. With the centred data $X_c = U S V^T$:

$$
\text{components} = \text{rows of } V^T, \qquad
\text{variance}_j = \frac{s_j^2}{n - 1}, \qquad
Z = X_c V_k
$$

## Decision trees

CART grows a binary tree greedily. At each node it tries every feature $j$ and threshold $t$ and keeps the
split that minimises the size-weighted impurity of the two children:

$$
\text{Gini: } \; n_t \Big(1 - \sum_k p_k^2\Big), \qquad
\text{squared error: } \; \sum_{i \in t} (y_i - \bar{y}_t)^2
$$

Growth stops at `max_depth`, below `min_samples_split`, or when no split leaves `min_samples_leaf` rows
on each side. Leaves predict the majority class (or class proportions) and the mean target.
