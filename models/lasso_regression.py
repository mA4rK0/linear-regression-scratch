"""
Lasso Regression (L1 Regularization) — from scratch (pure Python).

Model:  y = b0 + w1*x1 + ... + wm*xm
Loss:   L = MSE + λ * sum(|w_j|)
Method: Gradient Descent with subgradient for |w_j|

Why Lasso?
----------
Lasso adds a penalty proportional to the *absolute* value of each weight.
Unlike Ridge (L2), the L1 penalty can shrink weights all the way to **zero**,
effectively selecting only the most important features.

Notation
--------
b0      : intercept (not regularized)
weights : feature weights [w1, ..., wm]
alpha   : learning rate
lambda_ : regularization strength (λ)
MSE     : (1/n) * sum((y_hat - y)^2)
loss    : MSE + λ * sum(|w_j|)
sign(w) : subgradient of |w|  →  1 if w>0, -1 if w<0, 0 if w==0
gradient w.r.t. wj: (∂MSE/∂wj) + λ * sign(wj)
"""

from typing import List, Tuple, Dict, Any


def _sign(w: float) -> float:
    """Subgradient of |w|: 1 if w>0, -1 if w<0, 0 if w==0."""
    if w > 0:
        return 1.0
    if w < 0:
        return -1.0
    return 0.0


class LassoRegression:
    """
    Lasso Regression (L1) trained via gradient descent with subgradients.

    Parameters
    ----------
    alpha : float
        Learning rate. Default: 0.01.
    lambda_ : float
        L1 regularization strength. Larger values → more sparsity.
        Default: 0.1.
    max_epochs : int
        Maximum iterations. Default: 1000.
    tolerance : float
        Early-stopping on |prev_loss - loss|. Default: 1e-6.

    Attributes
    ----------
    b0 : float
        Intercept (never regularized).
    weights : list of float
        Learned feature weights (may be exactly 0 with large lambda_).
    history : list of dict
        Per-epoch record: {'epoch', 'mse', 'loss', 'b0', 'weights'}.
        'mse'  = pure mean squared error.
        'loss' = regularized objective = mse + λ*Σ|wj|.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        lambda_: float = 0.1,
        max_epochs: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.lambda_ = lambda_
        self.max_epochs = max_epochs
        self.tolerance = tolerance

        self.b0: float = 0.0
        self.weights: List[float] = []
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self, X: List[List[float]], y: List[float]
    ) -> "LassoRegression":
        """
        Train the model.

        Parameters
        ----------
        X : list of list of float
            Training features, shape (n_samples, n_features).
        y : list of float
            Target values.

        Returns
        -------
        self
        """
        n = len(X)
        m = len(X[0])

        self.b0 = 0.0
        self.weights = [0.0] * m
        self.history = []

        prev_loss = float("inf")

        for epoch in range(self.max_epochs):
            sum_error = 0.0
            sum_error_w = [0.0] * m
            sum_sq_error = 0.0

            for xi, yi in zip(X, y):
                y_hat = self.b0 + sum(w * xij for w, xij in zip(self.weights, xi))
                error = y_hat - yi

                sum_error += error
                sum_sq_error += error ** 2

                for j in range(m):
                    sum_error_w[j] += error * xi[j]

            # Separate MSE from regularized loss
            mse = sum_sq_error / n
            l1_penalty = self.lambda_ * sum(abs(w) for w in self.weights)
            loss = mse + l1_penalty  # ← what we actually minimise

            # Gradients
            # Intercept: no regularization  
            grad_b0 = sum_error / n

            # Weights: subgradient of L1 term is λ * sign(wj)
            grad_w = [
                (sum_error_w[j] / n) + self.lambda_ * _sign(self.weights[j])
                for j in range(m)
            ]

            # Update
            self.b0 -= self.alpha * grad_b0
            for j in range(m):
                self.weights[j] -= self.alpha * grad_w[j]

            self.history.append(
                {
                    "epoch": epoch,
                    "mse": mse,
                    "loss": loss,
                    "b0": self.b0,
                    "weights": list(self.weights),
                }
            )

            # Convergence on the regularized loss (same as ridge — now consistent)
            if abs(prev_loss - loss) < self.tolerance:
                break

            prev_loss = loss

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        """Generate predictions."""
        return [
            self.b0 + sum(w * xij for w, xij in zip(self.weights, xi))
            for xi in X
        ]

    def equation(self) -> str:
        """Return model equation as a human-readable string."""
        terms = [f"{w:+.4f}·x{j+1}" for j, w in enumerate(self.weights)]
        return "y = " + f"{self.b0:.4f} " + " ".join(terms)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LassoRegression(alpha={self.alpha}, lambda_={self.lambda_}, "
            f"max_epochs={self.max_epochs}, tolerance={self.tolerance})"
        )


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data: List[Tuple[List[float], float]] = [
        ([1, 2], 5),
        ([2, 1], 6),
        ([3, 3], 10),
        ([4, 2], 11),
        ([5, 3], 14),
    ]
    X_demo = [x for x, _ in data]
    y_demo = [y for _, y in data]

    model = LassoRegression(alpha=0.01, lambda_=0.1, max_epochs=1000, tolerance=1e-6)
    model.fit(X_demo, y_demo)

    last = model.history[-1]
    print("=== Lasso Regression ===")
    print(f"Epochs run     : {len(model.history)}")
    print(f"Final MSE      : {last['mse']:.6f}   (pure prediction error)")
    print(f"Final Loss     : {last['loss']:.6f}   (regularized objective = MSE + L1)")
    print(f"Model          : {model.equation()}")
    print(f"Predictions    : {[round(p, 4) for p in model.predict(X_demo)]}")
    print(f"Sparse weights : {[round(w, 6) for w in model.weights]}")
