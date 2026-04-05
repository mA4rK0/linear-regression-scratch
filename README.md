# 📐 Linear Regression from Scratch

> **Complete implementation of four linear regression variants in pure Python — no scikit-learn, no magic. Just math, code, and intuition.**

An interactive learning toolkit with a beautiful Streamlit visualizer that lets you watch gradient descent work in real time, step by step, epoch by epoch.

---

## 📑 Table of Contents

1. [Project Overview](#-project-overview)
2. [Folder Structure](#-folder-structure)
3. [Quick Start](#-quick-start)
4. [What is Linear Regression?](#-what-is-linear-regression)
5. [Simple Linear Regression](#1-simple-linear-regression)
6. [Multiple Linear Regression](#2-multiple-linear-regression)
7. [Ridge Regression (L2)](#3-ridge-regression-l2-regularization)
8. [Lasso Regression (L1)](#4-lasso-regression-l1-regularization)
9. [Ridge vs Lasso — Key Differences](#-ridge-vs-lasso--key-differences)
10. [Bugs Fixed from Original Code](#-bugs-fixed-from-original-code)
11. [The Streamlit App](#-the-streamlit-app)
12. [API Reference](#-api-reference)

---

## 📦 Project Overview

| What             | Detail                                                         |
| ---------------- | -------------------------------------------------------------- |
| Language         | Python 3.10+                                                   |
| Dependencies     | `streamlit`, `matplotlib`, `numpy` (visualization only)        |
| Models           | Pure Python — zero scikit-learn                                |
| Interface        | Streamlit interactive app                                      |
| Concepts covered | Gradient descent, MSE, L1/L2 regularization, feature selection |

---

## 🗂 Folder Structure

```
linear_regression_ml/
├── .venv/                        # Virtual environment (excluded from git)
│
├── models/                       # ← Core implementations (pure Python)
│   ├── __init__.py               # Package exports
│   ├── simple_lr.py              # SimpleLinearRegression class
│   ├── multiple_lr.py            # MultipleLinearRegression class
│   ├── ridge_regression.py       # RidgeRegression class (L2)
│   └── lasso_regression.py       # LassoRegression class (L1)
│
├── app.py                        # Streamlit visualization app
├── requirements.txt              # Python dependencies
├── LICENSE
└── README.md                     # This file
```

---

## 🚀 Quick Start

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd linear_regression_ml
```

### 2. Create the virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the interactive app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 6. Run models standalone (optional)

```bash
python -m models.simple_lr
python -m models.multiple_lr
python -m models.ridge_regression
python -m models.lasso_regression
```

---

## 🧠 What is Linear Regression?

### The Feynman Way — Start with the Simplest Idea

**Imagine you are a weather forecaster.** You notice that every time the temperature goes up by 1°C, ice cream sales increase by roughly 50 units. You write this pattern down: `sales = 50 × temperature`. That's linear regression.

More formally, linear regression is a **supervised learning algorithm** that finds the best-fitting straight line (or hyperplane in multiple dimensions) through a set of data points. "Best fitting" means the line that minimizes the total prediction error across all training examples.

### The Three Core Ingredients

```
1. A MODEL        : ŷ = b₀ + b₁x₁ + ... + bₘxₘ
2. A LOSS         : MSE = (1/n) Σ (ŷᵢ − yᵢ)²
3. AN OPTIMIZER   : Gradient Descent
```

**The model** is a mathematical formula with knobs (called weights or parameters). You turn the knobs until the formula fits the data. **The loss** is a scoreboard — it measures how wrong the model is right now (lower is better). **The optimizer** is the automatic knob-turner — it reads the loss and adjusts the weights in the direction that reduces it.

### Gradient Descent — The Rolling Ball Analogy

Picture a hilly landscape. Your current position is your current parameter values (weights). The elevation at any point is the loss (how wrong you are). Your goal is to reach the lowest valley.

You can't see the whole landscape (it's foggy). But you can feel which direction is downhill under your feet — that's the **gradient** (the partial derivative of the loss w.r.t. each parameter). You take a small step in the opposite direction of the gradient (downhill). Repeat. Eventually you reach a valley.

```
Step size = α (learning rate)
Direction = -∇Loss (negative gradient = downhill)
Update rule: w ← w − α · ∂Loss/∂w
```

---

## 1. Simple Linear Regression

### What it is

The most basic form — one input feature, one output. The model draws a straight line through 2-D data.

```
ŷ = b₀ + b₁ · x
```

- **b₀** (intercept): the y-value when x = 0. Your revenue when temperature is 0.
- **b₁** (slope): for every +1 unit of x, how much does y change?

### The Analogy

You're measuring how study hours affect exam scores. You collected this data:

| Hours studied (x) | Score (y) |
| ----------------- | --------- |
| 1                 | 55        |
| 2                 | 60        |
| 4                 | 75        |
| 5                 | 80        |

Simple LR finds the equation `score = 50 + 6 × hours` automatically from this data.

### How the math works (step by step)

**Step 1 — Initialize:** Start with b₀ = 0, b₁ = 0 (complete ignorance).

**Step 2 — Predict:** For each sample, compute ŷᵢ = b₀ + b₁ · xᵢ.

**Step 3 — Measure Error:**

```
error_i = ŷᵢ − yᵢ          (positive = overestimated, negative = underestimated)
MSE = (1/n) Σ error_i²     (always positive, punishes big errors more)
```

**Step 4 — Compute Gradients:**

```
∂MSE/∂b₀ = (1/n) Σ errorᵢ
∂MSE/∂b₁ = (1/n) Σ errorᵢ · xᵢ
```

Intuition: if all predictions are too high (positive errors), grad_b₀ is positive → step b₀ down.

**Step 5 — Update:**

```
b₀ ← b₀ − α · ∂MSE/∂b₀
b₁ ← b₁ − α · ∂MSE/∂b₁
```

**Step 6 — Repeat** steps 2–5 until convergence (loss stops improving).

### Key parameters

| Parameter               | What it does                                                    | Typical range    |
| ----------------------- | --------------------------------------------------------------- | ---------------- | ------------- | ---- |
| `alpha` (learning rate) | Step size per update. Too large → overshoots. Too small → slow. | 0.001 – 0.1      |
| `max_epochs`            | Maximum iterations. Training stops early if converged.          | 500 – 5000       |
| `tolerance`             | Convergence threshold: stop when                                | prev_loss - loss | < tolerance. | 1e-6 |


---

## 2. Multiple Linear Regression

### What it is

The generalization of simple LR to many input features simultaneously.

```
ŷ = b₀ + w₁x₁ + w₂x₂ + ··· + wₘxₘ
```

### The Analogy

Now you're predicting house prices. One feature (size) wasn't enough. You also have number of bedrooms, distance to city center, age of the building... Multiple LR learns a separate weight for each feature, answering:

> "Holding everything else constant, how much does adding one more bedroom affect the price?"

This is called **the ceteris paribus interpretation** of each coefficient.

### How the math works

The math is identical to simple LR but extended across m features. The gradient for each weight wⱼ is:

```
∂MSE/∂wⱼ = (1/n) Σ (ŷᵢ − yᵢ) · xᵢⱼ
```

You compute this gradient for every feature independently, and update every weight simultaneously. This is called **batch gradient descent** — you compute one update using all n samples before moving.

### Understanding the weight vector

If your trained model gives `w = [2.5, -1.2]` for features `[size, distance]`:

- A 1-unit increase in size → +2.5 in price (positive relationship)
- A 1-unit increase in distance to city → -1.2 in price (farther = cheaper)

### The curse: multicollinearity

When features are highly correlated (e.g., both `size` and `rooms` grow together), gradients fight each other and weights become unstable. This is why we need regularization (Ridge & Lasso).

---

## 3. Ridge Regression (L2 Regularization)

### What it is

Multiple Linear Regression with an **L2 penalty** added to the loss:

```
Loss = MSE + λ · Σ wⱼ²
```

The optimization objective is now **two competing goals**:

1. Minimize prediction error (MSE)
2. Keep weights small (penalty term)

### The Analogy

Imagine you're training an overenthusiastic intern. Without constraints, they might memorize every quirk of your past clients (overfit) but fail with new clients. Ridge is like giving them a rule: "Don't put **all** your faith in any single pattern." It gently nudges all learned patterns toward being more modest.

The leash analogy: every weight is on a leash (the L2 penalty). Each weight can still go left or right, but the further it strays from zero, the harder the leash pulls it back. No weight ever reaches exactly zero — every feature keeps some influence.

### The gradient with L2

The addition of λΣwⱼ² changes the gradient:

```
∂Loss/∂wⱼ = (1/n)Σ(ŷᵢ−yᵢ)xᵢⱼ  +  2λwⱼ
              ^--- same as before ---^  ^--- new L2 term ---^

Update: wⱼ ← wⱼ − α · [(1/n)Σeᵢxᵢⱼ + 2λwⱼ]
```

> **Important:** The intercept `b₀` is **NOT regularized** by convention. We only penalize feature weights, not the bias term.

### MSE vs Loss — what we actually track

```
mse  = (1/n) Σ (ŷᵢ − yᵢ)²          ← pure fit quality  (how well we predict)
loss = mse + λ · Σ wⱼ²              ← optimization objective (what we minimize)
```

Gradient descent minimizes `loss`. But `mse` is what you report to a client — it tells them how accurate your predictions are, independent of the regularization trick you used.

### Effect of λ

| λ value          | Effect                                                 |
| ---------------- | ------------------------------------------------------ |
| 0                | Reduces to plain Multiple LR                           |
| Small (0.01–0.1) | Gentle shrinkage, barely affects fit                   |
| Medium (0.5–1.0) | Noticeable shrinkage, better generalization            |
| Very large (>>1) | All weights → 0, model becomes constant (underfitting) |

---

## 4. Lasso Regression (L1 Regularization)

### What it is

Multiple Linear Regression with an **L1 penalty**:

```
Loss = MSE + λ · Σ |wⱼ|
```

### The Analogy

Lasso is Ridge's more decisive sibling. While Ridge shrinks weights _toward_ zero, Lasso shoves them all the way _to_ zero when they're not earning their keep.

The knife analogy: Ridge is a rubber band (it can stretch, but always pulls back toward zero smoothly). Lasso is a knife — if a weight is small enough, it gets guillotined off completely.

Why does L1 do this but L2 doesn't? It's in the shape of the penalty:

```
L2 penalty: w²     → gradient = 2w      (approaches 0 as w→0, gentle near zero)
L1 penalty: |w|    → gradient = sign(w) (always ±λ, constant push toward zero)
```

The constant push of sign(w) can overpower a small gradient, forcing the weight all the way to zero. L2's gentle gradient weakens as the weight shrinks, so it never actually reaches zero.

### The subgradient

The absolute value `|w|` is not differentiable at `w = 0`. We use the **subgradient** instead:

```
sign(w) = +1  if w > 0
sign(w) = -1  if w < 0
sign(w) =  0  if w = 0
```

Gradient update:

```
∂Loss/∂wⱼ = (1/n)Σ(ŷᵢ−yᵢ)xᵢⱼ + λ·sign(wⱼ)

wⱼ ← wⱼ − α · [(1/n)Σeᵢxᵢⱼ + λ·sign(wⱼ)]
```

### Sparsity — automatic feature selection

When a feature genuinely doesn't help predict the target, its gradient will be near zero. With L2, the weight stays tiny but nonzero. With L1, the constant `±λ` push overwhelms a tiny gradient and drives the weight to exactly 0. The feature is **eliminated from the model entirely**.

This is incredibly powerful: if you have 100 features and only 10 matter, Lasso will automatically find those 10 and zero out the rest.

### Effect of λ on sparsity

| λ value | Effect                                                |
| ------- | ----------------------------------------------------- |
| 0       | Reduces to plain Multiple LR                          |
| Small   | Slight shrinkage, most weights survive                |
| Medium  | Some weights reach exactly 0 (feature selection!)     |
| Large   | Most weights = 0, only the strongest feature survives |

---

## ⚔️ Ridge vs Lasso — Key Differences

| Property                    | Ridge (L2)                     | Lasso (L1)                              |
| --------------------------- | ------------------------------ | --------------------------------------- |
| Penalty                     | λ · Σ wⱼ²                      | λ · Σ \|wⱼ\|                            |
| Gradient of penalty         | 2λwⱼ (proportional to w)       | λ·sign(wⱼ) (constant magnitude)         |
| Weights reach exactly 0?    | ❌ Never                       | ✅ Yes — feature selection              |
| Good when...                | All features are relevant      | Many features are irrelevant            |
| Handles correlated features | ✅ Better (distributes weight) | ⚠️ Picks one, zeros others              |
| Solution                    | Closed-form exists             | No closed form — needs iterative solver |

**Rule of thumb:**

- You have many features and suspect most are noise → **Lasso**
- You know all features matter but want to prevent overfitting → **Ridge**
- Not sure? → Try both and compare validation error

---

## 🖥 The Streamlit App

Launch with:

```bash
streamlit run app.py
```

### Pages

| Page               | What you see                                                                                               |
| ------------------ | ---------------------------------------------------------------------------------------------------------- |
| 🏠 **Home**        | Concept overview, gradient descent visualization, navigation guide                                         |
| 📈 **Simple LR**   | Step-through epoch explorer, regression line animation, loss curve, b₀/b₁ trajectories                     |
| 📊 **Multiple LR** | Predictions vs actuals bar chart, loss curve, per-weight evolution plot                                    |
| 🔵 **Ridge**       | MSE vs Loss comparison, weight shrinkage over epochs, **coefficient path** (final weight vs λ sweep)       |
| 🟠 **Lasso**       | Same as Ridge + **L1 vs L2 penalty shape comparison**, sparsity visualization, Ridge vs Lasso side-by-side |

### Interactive Controls (all pages)

| Control                | Effect                                          |
| ---------------------- | ----------------------------------------------- |
| Learning Rate α slider | Adjusts gradient descent step size in real time |
| Max Epochs slider      | Controls how long training runs                 |
| λ slider (Ridge/Lasso) | Adjusts regularization strength                 |
| **Epoch step slider**  | Replays the training history frame by frame     |

---

## 📖 API Reference

All models share a consistent interface:

```python
from models import SimpleLinearRegression, MultipleLinearRegression, RidgeRegression, LassoRegression

# 1. Instantiate with hyperparameters
model = RidgeRegression(alpha=0.01, lambda_=0.1, max_epochs=1000, tolerance=1e-6)

# 2. Train
model.fit(X_train, y_train)   # X: list[list[float]], y: list[float]

# 3. Predict
predictions = model.predict(X_test)

# 4. Inspect parameters
print(model.b0)       # intercept
print(model.weights)  # feature weights

# 5. Human-readable equation
print(model.equation())   # "y = 0.1234 +2.3456·x1 +1.2345·x2"

# 6. Training history (per epoch)
for record in model.history:
    # record = {'epoch': int, 'mse': float, 'loss': float, 'b0': float, 'weights': list}
    print(record)
```

### SimpleLinearRegression

```python
SimpleLinearRegression(alpha=0.01, max_epochs=1000, tolerance=1e-6)
model.fit(X: list[float], y: list[float])
model.predict(X: list[float]) -> list[float]
# history keys: epoch, mse, b0, b1
```

### MultipleLinearRegression

```python
MultipleLinearRegression(alpha=0.01, max_epochs=1000, tolerance=1e-6)
model.fit(X: list[list[float]], y: list[float])
model.predict(X: list[list[float]]) -> list[float]
# history keys: epoch, mse, b0, weights
```

### RidgeRegression

```python
RidgeRegression(alpha=0.01, lambda_=0.1, max_epochs=1000, tolerance=1e-6)
model.fit(X: list[list[float]], y: list[float])
model.predict(X: list[list[float]]) -> list[float]
# history keys: epoch, mse, loss, b0, weights
# mse  = pure prediction error
# loss = mse + λ·Σw²  (the objective being minimized)
```

### LassoRegression

```python
LassoRegression(alpha=0.01, lambda_=0.1, max_epochs=1000, tolerance=1e-6)
model.fit(X: list[list[float]], y: list[float])
model.predict(X: list[list[float]]) -> list[float]
# history keys: epoch, mse, loss, b0, weights
# mse  = pure prediction error
# loss = mse + λ·Σ|w|  (the objective being minimized)
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).
