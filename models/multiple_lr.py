"""
Multiple Linear Regression — from scratch (pure Python).

Model:  y = b0 + w1*x1 + w2*x2 + ... + wm*xm
Method: Gradient Descent

Notation
--------
b0      : intercept (bias)
weights : list of feature weights [w1, w2, ..., wm]
alpha   : learning rate
n       : number of training samples
m       : number of features
MSE     : Mean Squared Error = (1/n) * sum((y_hat - y)^2)
gradient: partial derivative of MSE w.r.t. each parameter
"""

from typing import List, Tuple, Dict, Any


class MultipleLinearRegression:
    """
    Multiple Linear Regression trained via gradient descent.

    Supports any number of input features.
    The model learns:  y_hat = b0 + w1*x1 + w2*x2 + ... + wm*xm

    Parameters
    ----------
    alpha : float
        Learning rate. Default: 0.01.
    max_epochs : int
        Maximum gradient descent iterations. Default: 1000.
    tolerance : float
        Early-stopping threshold on |prev_loss - loss|. Default: 1e-6.

    Attributes
    ----------
    b0 : float
        Learned intercept after calling fit().
    weights : list of float
        Learned feature weights [w1, ..., wm].
    history : list of dict
        Per-epoch record: {'epoch', 'mse', 'b0', 'weights'}.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        max_epochs: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        self.alpha = alpha
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
    ) -> "MultipleLinearRegression":
        """
        Train the model.

        Parameters
        ----------
        X : list of list of float
            2-D input array, shape (n_samples, n_features).
        y : list of float
            Target values, shape (n_samples,).

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
                # Forward pass: y_hat = b0 + dot(weights, xi)
                y_hat = self.b0 + sum(w * xij for w, xij in zip(self.weights, xi))
                error = y_hat - yi

                sum_error += error
                sum_sq_error += error ** 2

                for j in range(m):
                    sum_error_w[j] += error * xi[j]

            mse = sum_sq_error / n

            # Gradients
            grad_b0 = sum_error / n
            grad_w = [sum_error_w[j] / n for j in range(m)]

            # Update
            self.b0 -= self.alpha * grad_b0
            for j in range(m):
                self.weights[j] -= self.alpha * grad_w[j]

            # Snapshot history (copy weights to avoid mutation)
            self.history.append(
                {
                    "epoch": epoch,
                    "mse": mse,
                    "b0": self.b0,
                    "weights": list(self.weights),
                }
            )

            if abs(prev_loss - mse) < self.tolerance:
                break

            prev_loss = mse

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Generate predictions.

        Parameters
        ----------
        X : list of list of float
            Input samples.

        Returns
        -------
        list of float
        """
        return [
            self.b0 + sum(w * xij for w, xij in zip(self.weights, xi))
            for xi in X
        ]

    def equation(self) -> str:
        """Return the model equation as a human-readable string."""
        terms = [f"{w:+.4f}·x{j+1}" for j, w in enumerate(self.weights)]
        return "y = " + f"{self.b0:.4f} " + " ".join(terms)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MultipleLinearRegression(alpha={self.alpha}, "
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

    model = MultipleLinearRegression(alpha=0.01, max_epochs=1000, tolerance=1e-6)
    model.fit(X_demo, y_demo)

    print("=== Multiple Linear Regression ===")
    print(f"Epochs run  : {len(model.history)}")
    print(f"Final MSE   : {model.history[-1]['mse']:.6f}")
    print(f"Model       : {model.equation()}")
    print(f"Predictions : {[round(p, 4) for p in model.predict(X_demo)]}")
