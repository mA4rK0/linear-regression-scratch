"""
Ridge Regression (L2 Regularization) — from scratch (pure Python).

Model:  y = b0 + w1*x1 + ... + wm*xm
Loss:   L = MSE + λ * sum(w_j^2)
Method: Gradient Descent

Why Ridge?
----------
Without regularization, weights can grow very large when features are
correlated, causing the model to overfit. Ridge adds a penalty proportional
to the *squared* magnitude of every weight, gently pushing them toward zero
without eliminating any feature entirely.

Notation
--------
b0      : intercept (NOT regularized — standard practice)
weights : feature weights [w1, ..., wm]
alpha   : learning rate
lambda_ : regularization strength (λ)
MSE     : (1/n) * sum((y_hat - y)^2)          — pure fit quality
loss    : MSE + λ * sum(w_j^2)                  — optimization objective
gradient of loss w.r.t. wj: (∂MSE/∂wj) + 2λ*wj
"""

from typing import List, Tuple, Dict, Any


class RidgeRegression:
    """
    Ridge Regression (L2) trained via gradient descent.

    Parameters
    ----------
    alpha : float
        Learning rate. Default: 0.01.
    lambda_ : float
        L2 regularization strength. Larger values → more shrinkage.
        Default: 0.1.
    max_epochs : int
        Maximum iterations. Default: 1000.
    tolerance : float
        Early-stopping threshold on |prev_loss - loss|. Default: 1e-6.

    Attributes
    ----------
    b0 : float
        Intercept (not regularized).
    weights : list of float
        Learned feature weights.
    history : list of dict
        Per-epoch record: {'epoch', 'mse', 'loss', 'b0', 'weights'}.
        'mse'  = pure mean squared error (fit quality alone).
        'loss' = regularized objective (mse + λ‖w‖²) used for training.

    Bug Fixed vs. Original
    ----------------------
    The original script stored the *regularized* objective in a variable
    named `mse`, which was misleading.  Here we clearly separate:
        - `mse`  : pure prediction error  — tells you how well you fit the data
        - `loss` : regularized objective  — what gradient descent actually minimises
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
    ) -> "RidgeRegression":
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

            # --- Separate MSE (pure fit) from regularized loss ---
            mse = sum_sq_error / n
            l2_penalty = self.lambda_ * sum(w ** 2 for w in self.weights)
            loss = mse + l2_penalty  # ← this is what we minimize

            # --- Gradients ---
            # Intercept: NOT regularized (standard convention)
            grad_b0 = sum_error / n

            # Weights: gradient of loss includes L2 term  2λ*wj
            # Note: derivative of λ‖w‖² w.r.t. wj is 2λwj.
            # Many implementations absorb the 2 into λ; here we keep it explicit.
            grad_w = [
                (sum_error_w[j] / n) + 2 * self.lambda_ * self.weights[j]
                for j in range(m)
            ]

            # --- Update ---
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

            # Convergence on the regularized loss (what we actually optimise)
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
            f"RidgeRegression(alpha={self.alpha}, lambda_={self.lambda_}, "
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

    model = RidgeRegression(alpha=0.01, lambda_=0.1, max_epochs=1000, tolerance=1e-6)
    model.fit(X_demo, y_demo)

    last = model.history[-1]
    print("=== Ridge Regression ===")
    print(f"Epochs run     : {len(model.history)}")
    print(f"Final MSE      : {last['mse']:.6f}   (pure prediction error)")
    print(f"Final Loss     : {last['loss']:.6f}   (regularized objective = MSE + L2)")
    print(f"Model          : {model.equation()}")
    print(f"Predictions    : {[round(p, 4) for p in model.predict(X_demo)]}")
