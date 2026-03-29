# Linear Regression from Scratch
##
This file contains simple implementations of **Linear Regression using Gradient Descent**, **Linear Regression using the Normal Equation**, and **Locally Weighted Regression**.

---

## Linear Regression Model

Linear regression models the relationship between input features and the target as:

$$
\hat{y} = Xw + b
$$

where:

$$
X \in \mathbb{R}^{n \times d}, \quad
w \in \mathbb{R}^{d}, \quad
b \in \mathbb{R}
$$

Here:
- $X$ is the feature matrix
- $w$ is the weight vector
- $b$ is the bias term
- $\hat{y}$ is the predicted output

The objective is to learn $w$ and $b$ such that predictions are as close as possible to the true target values.

---

## Mean Squared Error Loss

To measure prediction error, linear regression commonly uses the **Mean Squared Error (MSE)** loss:

$$
J(w,b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

This loss penalizes larger errors more heavily because the difference is squared.

---

## Gradient Descent

Gradient descent is an iterative optimization algorithm used to minimize the loss function.

At each step, predictions are computed as:

$$
\hat{y} = Xw + b
$$

The gradients of the loss with respect to the parameters are:

$$
\frac{\partial J}{\partial w} = \frac{1}{n}X^T(\hat{y} - y)
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)
$$

The parameters are then updated using:

$$
w := w - \alpha \frac{\partial J}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

where $\alpha$ is the learning rate.

Gradient descent is useful when working with larger datasets because it avoids direct matrix inversion and updates parameters gradually.

---

## Normal Equation

The normal equation provides a closed-form solution for linear regression parameters.

Instead of iteratively updating the parameters, it computes them directly:

$$
\theta = (X^T X)^{-1} X^T y
$$

where:

$$
\theta =
\begin{bmatrix}
b \\
w
\end{bmatrix}
$$

This approach is simple and exact for small datasets, but it becomes computationally expensive when the number of features is very large.

In practice, a numerically more stable version often uses the pseudoinverse:

$$
\theta = X^{+} y
$$

where $X^{+}$ is the Moore-Penrose pseudoinverse of $X$.

---

## Locally Weighted Regression

Locally Weighted Regression is a non-parametric regression method that fits a local model around the query point.

Instead of learning one global line for the entire dataset, it gives higher importance to points that are closer to the point being predicted.

The weight assigned to each training example is computed using a Gaussian kernel:

$$
w^{(i)} = \exp\left(-\frac{(x^{(i)} - x)^2}{2\tau^2}\right)
$$

where:
- $x$ is the query point
- $x^{(i)}$ is a training point
- $\tau$ is the bandwidth parameter controlling how local the fit is

These weights form a diagonal matrix:

$$
W =
\begin{bmatrix}
w^{(1)} & 0 & \cdots & 0 \\
0 & w^{(2)} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & w^{(n)}
\end{bmatrix}
$$

The locally weighted parameters are then computed as:

$$
\theta = (X^T W X)^{-1} X^T W y
$$

The prediction at the query point is:

$$
\hat{y} = \theta^T x
$$

This method is powerful for capturing local nonlinear behavior, but it is slower because a new weighted regression must be solved for every query point.

---

## Summary

- **Gradient Descent** learns parameters iteratively by minimizing the loss step by step.
- **Normal Equation** computes the optimal parameters directly using matrix operations.
- **Locally Weighted Regression** fits a separate local model for each query point using nearby points with higher weights.

These methods help build intuition for how regression works mathematically and computationally from scratch.
