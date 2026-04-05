"""
Linear Regression Models — from scratch (pure Python, no NumPy/scikit-learn).

Available models:
    - SimpleLinearRegression   : y = b0 + b1*x  (one feature)
    - MultipleLinearRegression : y = b0 + w1*x1 + ... + wn*xn
    - RidgeRegression          : Multiple LR + L2 penalty on weights
    - LassoRegression          : Multiple LR + L1 penalty on weights
"""

from .simple_lr import SimpleLinearRegression
from .multiple_lr import MultipleLinearRegression
from .ridge_regression import RidgeRegression
from .lasso_regression import LassoRegression

__all__ = [
    "SimpleLinearRegression",
    "MultipleLinearRegression",
    "RidgeRegression",
    "LassoRegression",
]
