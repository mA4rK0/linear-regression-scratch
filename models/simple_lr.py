"""
Simple Linear Regression — from scratch (pure Python).

Model:  y = b0 + b1 * x
Method: Gradient Descent

Notation
--------
b0      : intercept (bias)
b1      : slope (weight for the single feature)
alpha   : learning rate
n       : number of training samples
MSE     : Mean Squared Error = (1/n) * sum((y_hat - y)^2)
gradient: partial derivative of MSE w.r.t. parameter
"""

from typing import List, Tuple, Dict, Any


class SimpleLinearRegression:
    """
    Simple Linear Regression trained via gradient descent.

    Only supports a **single** input feature x.
    The model learns:  y_hat = b0 + b1 * x

    Parameters
    ----------
    alpha : float
        Learning rate (step size for gradient descent). Default: 0.01.
    max_epochs : int
        Maximum number of gradient descent iterations. Default: 1000.
    tolerance : float
        Early-stopping threshold. Training stops when |prev_loss - loss|
        drops below this value. Default: 1e-6.

    Attributes
    ----------
    b0 : float
        Learned intercept after calling fit().
    b1 : float
        Learned slope after calling fit().
    history : list of dict
        Per-epoch record: {'epoch', 'mse', 'b0', 'b1'}.
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

        # Learned parameters (set after fit())
        self.b0: float = 0.0
        self.b1: float = 0.0
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: List[float], y: List[float]) -> "SimpleLinearRegression":
        """
        Train the model on the provided data.

        Parameters
        ----------
        X : list of float
            Input features (one value per sample).
        y : list of float
            Target values.

        Returns
        -------
        self
            Returns itself so calls can be chained: model.fit(X, y).predict(X_new)
        """
        self.b0 = 0.0
        self.b1 = 0.0
        self.history = []

        n = len(X)
        prev_loss = float("inf")

        for epoch in range(self.max_epochs):
            sum_error = 0.0
            sum_error_x = 0.0
            sum_sq_error = 0.0

            for xi, yi in zip(X, y):
                y_hat = self.b0 + self.b1 * xi
                error = y_hat - yi

                sum_error += error
                sum_error_x += error * xi
                sum_sq_error += error ** 2

            mse = sum_sq_error / n

            # Gradients: ∂MSE/∂b0 and ∂MSE/∂b1
            grad_b0 = sum_error / n
            grad_b1 = sum_error_x / n

            # Gradient descent update
            self.b0 -= self.alpha * grad_b0
            self.b1 -= self.alpha * grad_b1

            self.history.append(
                {"epoch": epoch, "mse": mse, "b0": self.b0, "b1": self.b1}
            )

            # Early stopping
            if abs(prev_loss - mse) < self.tolerance:
                break

            prev_loss = mse

        return self

    def predict(self, X: List[float]) -> List[float]:
        """
        Generate predictions for new inputs.

        Parameters
        ----------
        X : list of float
            Input features.

        Returns
        -------
        list of float
            Predicted y values.
        """
        return [self.b0 + self.b1 * xi for xi in X]

    def equation(self) -> str:
        """Return the model equation as a human-readable string."""
        sign = "+" if self.b1 >= 0 else "-"
        return f"y = {self.b0:.4f} {sign} {abs(self.b1):.4f}·x"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SimpleLinearRegression(alpha={self.alpha}, "
            f"max_epochs={self.max_epochs}, tolerance={self.tolerance})"
        )


# ---------------------------------------------------------------------------
# Standalone demo (run this file directly: python -m models.simple_lr)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data: List[Tuple[float, float]] = [
        (1, 3),
        (3, 4),
        (4, 7),
        (5, 9),
        (6, 10),
    ]
    X_demo = [x for x, _ in data]
    y_demo = [y for _, y in data]

    model = SimpleLinearRegression(alpha=0.01, max_epochs=1000, tolerance=1e-6)
    model.fit(X_demo, y_demo)

    print("=== Simple Linear Regression ===")
    print(f"Epochs run  : {len(model.history)}")
    print(f"Final MSE   : {model.history[-1]['mse']:.6f}")
    print(f"Model       : {model.equation()}")
    print(f"Predictions : {[round(p, 4) for p in model.predict(X_demo)]}")
