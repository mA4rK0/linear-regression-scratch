"""
app.py — Linear Regression from Scratch: Interactive Streamlit Visualizer

A beautiful, interactive learning tool that walks through each linear
regression variant step-by-step, showing gradient descent live.

Pages:
    🏠  Home           — What is linear regression?
    📈  Simple LR      — One feature, animated gradient descent
    📊  Multiple LR    — Two features, weight evolution
    🔵  Ridge (L2)     — Weight shrinkage vs. lambda
    🟠  Lasso (L1)     — Sparsity demo, coefficient paths
"""

import sys
import os

# Allow `from models import ...` regardless of how streamlit is launched
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from models import (
    SimpleLinearRegression,
    MultipleLinearRegression,
    RidgeRegression,
    LassoRegression,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Linear Regression from Scratch",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Adaptive CSS — works in both light and dark Streamlit themes
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Cards ── */
    .info-card {
        background: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.20);
        border-radius: 16px;
        padding: 22px 24px;
        margin: 10px 0;
    }

    /* ── Formula block ── */
    .formula-box {
        background: rgba(99, 102, 241, 0.08);
        border: 1.5px solid rgba(99, 102, 241, 0.30);
        border-radius: 12px;
        padding: 18px 24px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.98rem;
        margin: 12px 0;
        text-align: center;
        line-height: 2;
    }

    /* ── Info callout (green left border) ── */
    .insight-box {
        background: rgba(16, 185, 129, 0.07);
        border-left: 4px solid #10b981;
        border-radius: 0 12px 12px 0;
        padding: 14px 20px;
        margin: 12px 0;
        line-height: 1.7;
    }

    /* ── Note callout (amber left border) ── */
    .note-box {
        background: rgba(245, 158, 11, 0.07);
        border-left: 4px solid #f59e0b;
        border-radius: 0 12px 12px 0;
        padding: 14px 20px;
        margin: 12px 0;
        line-height: 1.7;
    }

    /* ── Hero ── */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #0ea5e9 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 1.05rem;
        opacity: 0.65;
        margin-bottom: 1.8rem;
    }

    /* ── Chips ── */
    .metric-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.10);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 9999px;
        padding: 4px 16px;
        font-size: 0.84rem;
        margin: 4px 4px 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib — clean neutral theme (readable on white background)
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#374151",
        "axes.titlecolor": "#111827",
        "xtick.color": "#6b7280",
        "ytick.color": "#6b7280",
        "grid.color": "#e5e7eb",
        "grid.alpha": 0.9,
        "text.color": "#374151",
        "lines.linewidth": 2.2,
        "font.family": "DejaVu Sans",
        "figure.dpi": 110,
    }
)

# Vivid accent palette — pops on white backgrounds
ACCENT = "#6366f1"   # indigo
TEAL   = "#0ea5e9"   # sky
GREEN  = "#10b981"   # emerald
ORANGE = "#f97316"   # orange
PINK   = "#ec4899"   # pink

# ─────────────────────────────────────────────────────────────────────────────
# Shared demo datasets
# ─────────────────────────────────────────────────────────────────────────────
SIMPLE_DATA = [(1, 3), (3, 4), (4, 7), (5, 9), (6, 10)]
MULTI_DATA  = [([1, 2], 5), ([2, 1], 6), ([3, 3], 10), ([4, 2], 11), ([5, 3], 14)]

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-size:1.3rem;font-weight:700;color:#6366f1;">📐 LinReg from Scratch</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠 Home",
            "📈 Simple Linear Regression",
            "📊 Multiple Linear Regression",
            "🔵 Ridge Regression (L2)",
            "🟠 Lasso Regression (L1)",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Built from scratch — pure Python.\nNo scikit-learn. No magic.")


# helper: draw a clean legend (neutral colours so it works light/dark)
def _legend(ax):
    ax.legend(fontsize=9, framealpha=0.85, edgecolor="#d1d5db")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 0 — HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<h1 class="hero-title">Linear Regression from Scratch</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-sub">A complete interactive guide — built with pure Python, no libraries, no shortcuts.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        st.markdown(
            """
            <div class="info-card">
            <h3>🎯 What is Linear Regression?</h3>
            <p style="line-height:1.8;">
            Imagine you're a <strong>detective</strong>. You have clues (features like house size, number of rooms)
            and you want to predict a number (house price). Linear regression is your magnifying glass —
            it finds the best straight-line relationship between clues and the answer.
            </p>
            <p style="line-height:1.8;opacity:0.75;">
            The "learning" part? Your model starts clueless (all parameters = 0), measures how wrong it is
            (MSE), then nudges itself in the direction that reduces the error — one epoch at a time.
            That's <strong>gradient descent</strong>, the engine behind all of machine learning.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="formula-box">
            Core idea: &nbsp;&nbsp; ŷ = b₀ + w₁x₁ + w₂x₂ + ··· + wₘxₘ<br>
            Minimize: &nbsp;&nbsp;&nbsp; MSE = (1/n) Σ (ŷᵢ − yᵢ)²
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="info-card">
            <h3>🗺️ What's in this app?</h3>
            <table style="width:100%;border-collapse:collapse;">
            <tr><td style="padding:8px 6px;">📈 <strong>Simple LR</strong></td>
                <td style="padding:8px 6px;opacity:0.7;">One feature. Understand the core gradient descent loop.</td></tr>
            <tr><td style="padding:8px 6px;">📊 <strong>Multiple LR</strong></td>
                <td style="padding:8px 6px;opacity:0.7;">Many features. See weights evolve independently.</td></tr>
            <tr><td style="padding:8px 6px;">🔵 <strong>Ridge (L2)</strong></td>
                <td style="padding:8px 6px;opacity:0.7;">Add L2 penalty. Prevent overfitting. Shrink weights.</td></tr>
            <tr><td style="padding:8px 6px;">🟠 <strong>Lasso (L1)</strong></td>
                <td style="padding:8px 6px;opacity:0.7;">Add L1 penalty. Drive weak weights to exactly zero.</td></tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # Gradient descent illustration
        fig, ax = plt.subplots(figsize=(5, 4))
        x_curve = np.linspace(-3, 3, 300)
        y_curve = x_curve ** 2 + 0.5
        ax.plot(x_curve, y_curve, color=TEAL, lw=2.5, label="Loss surface")

        steps_x = [-2.5, -1.8, -1.1, -0.5, -0.1, 0.05]
        steps_y = [xi ** 2 + 0.5 for xi in steps_x]
        ax.scatter(steps_x, steps_y, color=ACCENT, s=80, zorder=5)
        for i in range(len(steps_x) - 1):
            ax.annotate(
                "",
                xy=(steps_x[i + 1], steps_y[i + 1]),
                xytext=(steps_x[i], steps_y[i]),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8),
            )
        ax.set_title("Gradient Descent", fontsize=13, pad=12)
        ax.set_xlabel("Parameter value (w)")
        ax.set_ylabel("Loss (MSE)")
        _legend(ax)
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(
            """
            <div class="insight-box">
            💡 <strong>Feynman Analogy</strong><br>
            Gradient descent is like rolling a ball down a foggy hill.
            You can't see the bottom — but you can always feel which way is downhill.
            Take a small step there. Repeat. Eventually you reach the valley (minimum loss).
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <span class="metric-chip">🐍 Pure Python</span>
        <span class="metric-chip">📐 Gradient Descent</span>
        <span class="metric-chip">🚫 No scikit-learn</span>
        <span class="metric-chip">📖 Feynman Explanations</span>
        <span class="metric-chip">🎬 Step-by-Step Explorer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SIMPLE LINEAR REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Simple Linear Regression":
    st.markdown("## 📈 Simple Linear Regression")
    st.markdown("One feature → one target. The simplest, purest form of machine learning.")

    with st.expander("📖 Theory & Intuition", expanded=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(
                """
                <div class="info-card">
                <h4>🎯 The Analogy</h4>
                <p style="line-height:1.8;">
                You run an ice cream shop. You notice hotter days → more sales.
                Simple linear regression finds <em>exactly</em> how much each degree of temperature
                adds to your revenue — automatically, from your past data.
                </p>
                <p style="line-height:1.8;opacity:0.75;">
                The model has two knobs:<br>
                • <strong>b₀</strong> (intercept): baseline sales when temp = 0<br>
                • <strong>b₁</strong> (slope): extra sales per degree
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="formula-box">
                Model: &nbsp; ŷ = b₀ + b₁·x<br><br>
                MSE = (1/n) Σ (ŷᵢ − yᵢ)²<br><br>
                ∂MSE/∂b₀ = (1/n) Σ (ŷᵢ − yᵢ)<br>
                ∂MSE/∂b₁ = (1/n) Σ (ŷᵢ − yᵢ)·xᵢ<br><br>
                b₀ ← b₀ − α · ∂MSE/∂b₀<br>
                b₁ ← b₁ − α · ∂MSE/∂b₁
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div class="note-box">
            ⚡ <strong>What is a gradient?</strong> It's the slope of the error curve at your current position.
            A positive gradient means "moving right increases error" — so step left. Always step
            <em>opposite</em> to the gradient.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Controls ─────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Hyperparameters")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        slr_alpha = st.slider("Learning Rate (α)", 0.001, 0.1, 0.01, 0.001, key="slr_alpha",
                              help="Step size for gradient descent. Too large → overshoots. Too small → slow.")
    with cc2:
        slr_epochs = st.slider("Max Epochs", 50, 2000, 500, 50, key="slr_epochs")
    with cc3:
        slr_tol = st.select_slider("Tolerance", [1e-8, 1e-6, 1e-4, 1e-2], value=1e-6, key="slr_tol")

    # ── Train ─────────────────────────────────────────────────────────────────
    X_s = [x for x, _ in SIMPLE_DATA]
    y_s = [y for _, y in SIMPLE_DATA]

    model_s = SimpleLinearRegression(alpha=slr_alpha, max_epochs=slr_epochs, tolerance=slr_tol)
    model_s.fit(X_s, y_s)
    hist_s = model_s.history

    # ── Step explorer ─────────────────────────────────────────────────────────
    st.markdown("### 🎬 Step-by-Step Training Explorer")
    epoch_idx = st.slider(
        "Epoch", 0, len(hist_s) - 1, len(hist_s) - 1, key="slr_epoch_idx",
        help="Drag to replay gradient descent step by step",
    )
    snap = hist_s[epoch_idx]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Epoch", snap["epoch"])
    m2.metric("MSE", f"{snap['mse']:.5f}")
    m3.metric("b₀ (intercept)", f"{snap['b0']:.4f}")
    m4.metric("b₁ (slope)", f"{snap['b1']:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    x_range = np.linspace(min(X_s) - 0.5, max(X_s) + 0.5, 200)
    y_line  = snap["b0"] + snap["b1"] * x_range
    ax1.scatter(X_s, y_s, color=TEAL, s=110, zorder=5, label="Training data", edgecolors="white", linewidths=0.8)
    ax1.plot(x_range, y_line, color=ACCENT, lw=2.2, label=f"ŷ = {snap['b0']:.3f} + {snap['b1']:.3f}·x")
    for xi, yi in zip(X_s, y_s):
        yi_hat = snap["b0"] + snap["b1"] * xi
        ax1.vlines(xi, min(yi, yi_hat), max(yi, yi_hat), color=ORANGE, lw=1.8, alpha=0.7, linestyle="--")
    ax1.set_title(f"Regression Line — Epoch {snap['epoch']}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    _legend(ax1)
    ax1.grid(True)

    epochs_all = [h["epoch"] for h in hist_s]
    mse_all    = [h["mse"]   for h in hist_s]
    ax2.plot(epochs_all, mse_all, color=GREEN, lw=2, label="MSE")
    ax2.axvline(snap["epoch"], color=ORANGE, lw=1.8, linestyle="--", alpha=0.9, label=f"Epoch {snap['epoch']}")
    ax2.scatter([snap["epoch"]], [snap["mse"]], color=ORANGE, s=80, zorder=5)
    ax2.set_title("Loss Curve (MSE)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    _legend(ax2)
    ax2.grid(True)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Weight trajectory
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 3))
    b0_vals = [h["b0"] for h in hist_s]
    b1_vals = [h["b1"] for h in hist_s]
    axes2[0].plot(epochs_all, b0_vals, color=PINK, lw=2)
    axes2[0].axvline(snap["epoch"], color=ORANGE, lw=1.5, linestyle="--", alpha=0.8)
    axes2[0].set_title("b₀ (intercept) over Epochs", fontweight="bold")
    axes2[0].set_xlabel("Epoch")
    axes2[0].grid(True)
    axes2[1].plot(epochs_all, b1_vals, color=ACCENT, lw=2)
    axes2[1].axvline(snap["epoch"], color=ORANGE, lw=1.5, linestyle="--", alpha=0.8)
    axes2[1].set_title("b₁ (slope) over Epochs", fontweight="bold")
    axes2[1].set_xlabel("Epoch")
    axes2[1].grid(True)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    st.markdown(
        f"""
        <div class="insight-box">
        ✅ <strong>Final Model:</strong> &nbsp; <code>{model_s.equation()}</code>
        &nbsp;&nbsp;|&nbsp;&nbsp; Epochs run: {len(hist_s)}
        &nbsp;&nbsp;|&nbsp;&nbsp; Final MSE: {hist_s[-1]['mse']:.6f}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MULTIPLE LINEAR REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Multiple Linear Regression":
    st.markdown("## 📊 Multiple Linear Regression")
    st.markdown("Multiple features — each with its own learned weight.")

    with st.expander("📖 Theory & Intuition", expanded=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(
                """
                <div class="info-card">
                <h4>🎯 The Analogy</h4>
                <p style="line-height:1.8;">
                Now you're a real estate agent. House price depends on <em>size</em> AND
                <em>number of rooms</em>. Multiple LR learns a separate weight for each feature
                simultaneously — each one answers:
                </p>
                <p style="line-height:1.8;opacity:0.75;">
                <em>"Holding everything else constant, how much does adding one more unit of
                feature xⱼ change the price?"</em> — the classic <strong>ceteris paribus</strong>
                interpretation of each weight.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="formula-box">
                Model: &nbsp; ŷ = b₀ + w₁x₁ + w₂x₂ + ··· + wₘxₘ<br><br>
                ∂MSE/∂b₀ = (1/n) Σ (ŷᵢ − yᵢ)<br>
                ∂MSE/∂wⱼ = (1/n) Σ (ŷᵢ − yᵢ)·xᵢⱼ<br><br>
                b₀ ← b₀ − α · ∂MSE/∂b₀<br>
                wⱼ ← wⱼ − α · ∂MSE/∂wⱼ
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### ⚙️ Hyperparameters")
    mc1, mc2 = st.columns(2)
    with mc1:
        mlr_alpha = st.slider("Learning Rate (α)", 0.001, 0.05, 0.01, 0.001, key="mlr_alpha")
    with mc2:
        mlr_epochs = st.slider("Max Epochs", 50, 2000, 500, 50, key="mlr_epochs")

    X_m = [x for x, _ in MULTI_DATA]
    y_m = [y for _, y in MULTI_DATA]

    model_m = MultipleLinearRegression(alpha=mlr_alpha, max_epochs=mlr_epochs, tolerance=1e-6)
    model_m.fit(X_m, y_m)
    hist_m = model_m.history

    st.markdown("### 🎬 Step-by-Step Explorer")
    epoch_m  = st.slider("Epoch", 0, len(hist_m) - 1, len(hist_m) - 1, key="mlr_epoch_idx")
    snap_m   = hist_m[epoch_m]

    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.metric("Epoch", snap_m["epoch"])
    mm2.metric("MSE",   f"{snap_m['mse']:.5f}")
    mm3.metric("w₁",    f"{snap_m['weights'][0]:.4f}")
    mm4.metric("w₂",    f"{snap_m['weights'][1]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Predictions vs actuals
    preds_m = [
        snap_m["b0"] + sum(w * xij for w, xij in zip(snap_m["weights"], xi))
        for xi in X_m
    ]
    x_pos = range(len(y_m))
    axes[0].bar([p - 0.2 for p in x_pos], y_m,     width=0.38, color=TEAL,   alpha=0.85, label="Actual")
    axes[0].bar([p + 0.2 for p in x_pos], preds_m,  width=0.38, color=ACCENT, alpha=0.85, label="Predicted")
    axes[0].set_title(f"Predictions vs Actual — Epoch {snap_m['epoch']}", fontweight="bold")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("y")
    _legend(axes[0])
    axes[0].grid(True, axis="y")

    # MSE loss curve
    epochs_m_all = [h["epoch"] for h in hist_m]
    mse_m_all    = [h["mse"]   for h in hist_m]
    axes[1].plot(epochs_m_all, mse_m_all, color=GREEN, lw=2)
    axes[1].axvline(snap_m["epoch"], color=ORANGE, lw=1.8, linestyle="--", alpha=0.9)
    axes[1].scatter([snap_m["epoch"]], [snap_m["mse"]], color=ORANGE, s=80, zorder=5)
    axes[1].set_title("MSE Loss Curve", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].grid(True)

    # Weight evolution
    w1_vals = [h["weights"][0] for h in hist_m]
    w2_vals = [h["weights"][1] for h in hist_m]
    axes[2].plot(epochs_m_all, w1_vals, color=PINK,  lw=2, label="w₁ (feature 1)")
    axes[2].plot(epochs_m_all, w2_vals, color=ACCENT, lw=2, label="w₂ (feature 2)")
    axes[2].axvline(snap_m["epoch"], color=ORANGE, lw=1.5, linestyle="--", alpha=0.8)
    axes[2].set_title("Weight Evolution", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Weight value")
    _legend(axes[2])
    axes[2].grid(True)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        f"""
        <div class="insight-box">
        ✅ <strong>Final Model:</strong> &nbsp; <code>{model_m.equation()}</code>
        &nbsp;&nbsp;|&nbsp;&nbsp; Epochs run: {len(hist_m)}
        &nbsp;&nbsp;|&nbsp;&nbsp; Final MSE: {hist_m[-1]['mse']:.6f}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RIDGE REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Ridge Regression (L2)":
    st.markdown("## 🔵 Ridge Regression (L2 Regularization)")
    st.markdown("Same as Multiple LR, but we penalize large weights to prevent overfitting.")

    with st.expander("📖 Theory & Intuition", expanded=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(
                """
                <div class="info-card">
                <h4>🎯 The Analogy</h4>
                <p style="line-height:1.8;">
                Imagine training a dog. Without constraints, it might memorise every quirk of your house
                and fail with new owners (overfitting). Ridge is like a leash —  the dog can still move
                freely, but it can't go too far in any one direction.
                </p>
                <p style="line-height:1.8;opacity:0.75;">
                The L2 penalty adds <strong>λ · Σwⱼ²</strong> to the loss, gently pushing all weights
                toward zero. No weight ever reaches <em>exactly</em> zero — every feature keeps some influence.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="formula-box">
                Loss = MSE + λ · Σwⱼ²<br><br>
                ∂Loss/∂wⱼ = (1/n)Σ(ŷᵢ−yᵢ)xᵢⱼ + 2λwⱼ<br><br>
                wⱼ ← wⱼ − α · (∂MSE/∂wⱼ + 2λwⱼ)<br><br>
                <span style="font-size:0.85em;opacity:0.6;">
                b₀ is NOT regularized — standard practice
                </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div class="note-box">
            💡 <strong>Two metrics to watch:</strong>
            <strong>MSE</strong> = pure prediction error (how accurate your predictions are).
            <strong>Loss</strong> = MSE + λ‖w‖² — the full objective that gradient descent minimises.
            As λ grows, Loss rises above MSE because the penalty term gets heavier.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ⚙️ Hyperparameters")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        r_alpha  = st.slider("Learning Rate (α)", 0.001, 0.05, 0.01, 0.001, key="r_alpha")
    with rc2:
        r_lambda = st.slider("Regularization λ", 0.0, 2.0, 0.1, 0.05, key="r_lambda")
    with rc3:
        r_epochs = st.slider("Max Epochs", 50, 2000, 500, 50, key="r_epochs")

    X_r = [x for x, _ in MULTI_DATA]
    y_r = [y for _, y in MULTI_DATA]

    model_r = RidgeRegression(alpha=r_alpha, lambda_=r_lambda, max_epochs=r_epochs, tolerance=1e-6)
    model_r.fit(X_r, y_r)
    hist_r = model_r.history

    st.markdown("### 🎬 Step-by-Step Explorer")
    epoch_r = st.slider("Epoch", 0, len(hist_r) - 1, len(hist_r) - 1, key="r_epoch_idx")
    snap_r  = hist_r[epoch_r]

    rm1, rm2, rm3, rm4, rm5 = st.columns(5)
    rm1.metric("Epoch",  snap_r["epoch"])
    rm2.metric("MSE",    f"{snap_r['mse']:.5f}")
    rm3.metric("Loss",   f"{snap_r['loss']:.5f}")
    rm4.metric("w₁",     f"{snap_r['weights'][0]:.4f}")
    rm5.metric("w₂",     f"{snap_r['weights'][1]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    epochs_r_all = [h["epoch"] for h in hist_r]
    mse_r_all    = [h["mse"]   for h in hist_r]
    loss_r_all   = [h["loss"]  for h in hist_r]

    # Loss vs MSE
    axes[0].plot(epochs_r_all, loss_r_all, color=ACCENT, lw=2, label="Regularized Loss")
    axes[0].plot(epochs_r_all, mse_r_all,  color=GREEN,  lw=2, linestyle="--", label="Pure MSE")
    axes[0].axvline(snap_r["epoch"], color=ORANGE, lw=1.5, linestyle=":", alpha=0.9)
    axes[0].set_title("Loss vs Pure MSE", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    _legend(axes[0])
    axes[0].grid(True)

    # Weight evolution
    w1_r = [h["weights"][0] for h in hist_r]
    w2_r = [h["weights"][1] for h in hist_r]
    axes[1].plot(epochs_r_all, w1_r, color=PINK, lw=2, label="w₁")
    axes[1].plot(epochs_r_all, w2_r, color=TEAL, lw=2, label="w₂")
    axes[1].axvline(snap_r["epoch"], color=ORANGE, lw=1.5, linestyle="--", alpha=0.8)
    axes[1].set_title("Weight Shrinkage over Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weight value")
    _legend(axes[1])
    axes[1].grid(True)

    # Lambda sweep: coefficient path
    lambdas  = [i * 0.1 for i in range(21)]
    final_w1 = []
    final_w2 = []
    for lam in lambdas:
        tmp = RidgeRegression(alpha=r_alpha, lambda_=lam, max_epochs=500, tolerance=1e-6)
        tmp.fit(X_r, y_r)
        final_w1.append(tmp.weights[0])
        final_w2.append(tmp.weights[1])
    axes[2].plot(lambdas, final_w1, color=PINK, lw=2, marker="o", markersize=4, label="w₁")
    axes[2].plot(lambdas, final_w2, color=TEAL, lw=2, marker="s", markersize=4, label="w₂")
    axes[2].axvline(r_lambda, color=ORANGE, lw=1.8, linestyle="--", alpha=0.9, label=f"Current λ = {r_lambda}")
    axes[2].set_title("Coefficient Path vs λ  (Ridge never hits 0)", fontweight="bold")
    axes[2].set_xlabel("λ (regularization strength)")
    axes[2].set_ylabel("Final weight value")
    _legend(axes[2])
    axes[2].grid(True)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        f"""
        <div class="insight-box">
        ✅ <strong>Final Model:</strong> &nbsp; <code>{model_r.equation()}</code>
        &nbsp;&nbsp;|&nbsp;&nbsp; Epochs run: {len(hist_r)}
        &nbsp;&nbsp;|&nbsp;&nbsp; MSE: {hist_r[-1]['mse']:.6f}
        &nbsp;&nbsp;|&nbsp;&nbsp; Loss: {hist_r[-1]['loss']:.6f}
        &nbsp;&nbsp;|&nbsp;&nbsp; λ = {r_lambda}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LASSO REGRESSION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🟠 Lasso Regression (L1)":
    st.markdown("## 🟠 Lasso Regression (L1 Regularization)")
    st.markdown("L1 penalty can shrink weights all the way to **zero** — automatic feature selection!")

    with st.expander("📖 Theory & Intuition", expanded=True):
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(
                """
                <div class="info-card">
                <h4>🎯 The Analogy</h4>
                <p style="line-height:1.8;">
                Ridge is a leash — Lasso is a knife. While Ridge <em>shrinks</em> weights toward zero,
                Lasso can cut weak weights completely to <em>exactly</em> zero, eliminating useless
                features from the model entirely.
                </p>
                <p style="line-height:1.8;opacity:0.75;">
                Why? The L1 penalty has a sharp corner at zero (unlike the smooth L2 quadratic).
                The gradient near zero is a constant <strong>±λ</strong>, which can overwhelm a tiny
                weight gradient and pin it at zero — built-in <strong>feature selection</strong>.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="formula-box">
                Loss = MSE + λ · Σ|wⱼ|<br><br>
                ∂Loss/∂wⱼ = (1/n)Σ(ŷᵢ−yᵢ)xᵢⱼ + λ·sign(wⱼ)<br><br>
                sign(w) = +1 if w&gt;0, −1 if w&lt;0, 0 if w=0<br><br>
                wⱼ ← wⱼ − α · (∂MSE/∂wⱼ + λ·sign(wⱼ))
                </div>
                """,
                unsafe_allow_html=True,
            )

        # L1 vs L2 penalty shape comparison — always shown
        fig_compare, axes_c = plt.subplots(1, 2, figsize=(12, 3.5))
        w_vals = np.linspace(-2, 2, 300)

        axes_c[0].plot(w_vals, w_vals ** 2,     color=TEAL,   lw=2.5, label="L2 = w²  (Ridge)")
        axes_c[0].plot(w_vals, np.abs(w_vals),  color=ORANGE, lw=2.5, linestyle="--", label="L1 = |w|  (Lasso)")
        axes_c[0].set_title("Penalty Shape Comparison", fontweight="bold")
        axes_c[0].set_xlabel("w")
        axes_c[0].set_ylabel("Penalty value")
        _legend(axes_c[0])
        axes_c[0].grid(True)

        axes_c[1].plot(w_vals, 2 * w_vals,      color=TEAL,   lw=2.5, label="∂L2/∂w = 2w  (shrinks to 0)")
        axes_c[1].plot(w_vals, np.sign(w_vals), color=ORANGE, lw=2.5, linestyle="--", label="∂L1/∂w = sign(w)  (constant push)")
        axes_c[1].axhline(0, color="#9ca3af", lw=1, linestyle=":")
        axes_c[1].set_title("Gradient of Penalty  →  why Lasso hits exactly 0", fontweight="bold")
        axes_c[1].set_xlabel("w")
        axes_c[1].set_ylabel("Gradient")
        _legend(axes_c[1])
        axes_c[1].grid(True)
        st.pyplot(fig_compare, use_container_width=True)
        plt.close(fig_compare)

        st.markdown(
            """
            <div class="note-box">
            🔑 <strong>Key insight:</strong> L2's gradient (2w) weakens as w→0, so it never actually
            reaches zero. L1's gradient (sign(w)) stays at a constant ±λ and can overpower a tiny
            weight gradient, pushing it all the way to zero — that's the sparsity superpower.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Controls
    st.markdown("### ⚙️ Hyperparameters")
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        l_alpha  = st.slider("Learning Rate (α)", 0.001, 0.05, 0.01, 0.001, key="l_alpha")
    with lc2:
        l_lambda = st.slider("Regularization λ", 0.0, 2.0, 0.1, 0.05, key="l_lambda")
    with lc3:
        l_epochs = st.slider("Max Epochs", 50, 2000, 500, 50, key="l_epochs")

    X_l = [x for x, _ in MULTI_DATA]
    y_l = [y for _, y in MULTI_DATA]

    model_l = LassoRegression(alpha=l_alpha, lambda_=l_lambda, max_epochs=l_epochs, tolerance=1e-6)
    model_l.fit(X_l, y_l)
    hist_l = model_l.history

    st.markdown("### 🎬 Step-by-Step Explorer")
    epoch_l = st.slider("Epoch", 0, len(hist_l) - 1, len(hist_l) - 1, key="l_epoch_idx")
    snap_l  = hist_l[epoch_l]

    lm1, lm2, lm3, lm4, lm5 = st.columns(5)
    lm1.metric("Epoch", snap_l["epoch"])
    lm2.metric("MSE",   f"{snap_l['mse']:.5f}")
    lm3.metric("Loss",  f"{snap_l['loss']:.5f}")
    lm4.metric("w₁",    f"{snap_l['weights'][0]:.4f}")
    lm5.metric("w₂",    f"{snap_l['weights'][1]:.4f}")

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4))

    epochs_l_all = [h["epoch"] for h in hist_l]
    mse_l_all    = [h["mse"]   for h in hist_l]
    loss_l_all   = [h["loss"]  for h in hist_l]

    # Loss vs MSE
    axes3[0].plot(epochs_l_all, loss_l_all, color=ORANGE, lw=2, label="Regularized Loss")
    axes3[0].plot(epochs_l_all, mse_l_all,  color=GREEN,  lw=2, linestyle="--", label="Pure MSE")
    axes3[0].axvline(snap_l["epoch"], color=ACCENT, lw=1.5, linestyle=":", alpha=0.9)
    axes3[0].set_title("Loss vs Pure MSE", fontweight="bold")
    axes3[0].set_xlabel("Epoch")
    axes3[0].set_ylabel("Value")
    _legend(axes3[0])
    axes3[0].grid(True)

    # Weight evolution
    w1_l = [h["weights"][0] for h in hist_l]
    w2_l = [h["weights"][1] for h in hist_l]
    axes3[1].plot(epochs_l_all, w1_l, color=PINK,  lw=2, label="w₁")
    axes3[1].plot(epochs_l_all, w2_l, color=TEAL,  lw=2, label="w₂")
    axes3[1].axhline(0, color="#9ca3af", lw=1.2, linestyle="--")
    axes3[1].axvline(snap_l["epoch"], color=ORANGE, lw=1.5, linestyle="--", alpha=0.8)
    axes3[1].set_title("Weight Evolution  (Lasso can hit exactly 0!)", fontweight="bold")
    axes3[1].set_xlabel("Epoch")
    axes3[1].set_ylabel("Weight value")
    _legend(axes3[1])
    axes3[1].grid(True)

    # Lambda sweep: sparsity / coefficient path
    lambdas_l = [i * 0.1 for i in range(21)]
    fw1_l, fw2_l = [], []
    for lam in lambdas_l:
        tmp = LassoRegression(alpha=l_alpha, lambda_=lam, max_epochs=500, tolerance=1e-8)
        tmp.fit(X_l, y_l)
        fw1_l.append(tmp.weights[0])
        fw2_l.append(tmp.weights[1])
    axes3[2].plot(lambdas_l, fw1_l, color=PINK,   lw=2, marker="o", markersize=4, label="w₁")
    axes3[2].plot(lambdas_l, fw2_l, color=TEAL,   lw=2, marker="s", markersize=4, label="w₂")
    axes3[2].axhline(0, color="#9ca3af", lw=1.2, linestyle="--")
    axes3[2].axvline(l_lambda, color=ORANGE, lw=1.8, linestyle="--", alpha=0.9, label=f"Current λ = {l_lambda}")
    axes3[2].set_title("Coefficient Path vs λ  (Lasso hits 0!)", fontweight="bold")
    axes3[2].set_xlabel("λ (regularization strength)")
    axes3[2].set_ylabel("Final weight value")
    _legend(axes3[2])
    axes3[2].grid(True)

    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Ridge vs Lasso side-by-side
    st.markdown("### 🆚 Ridge vs Lasso — Side-by-Side Comparison")
    compare_lambdas = [i * 0.1 for i in range(21)]
    ridge_w1, ridge_w2, lasso_w1, lasso_w2 = [], [], [], []
    for lam in compare_lambdas:
        r = RidgeRegression(alpha=0.01, lambda_=lam, max_epochs=500, tolerance=1e-8)
        r.fit(X_l, y_l)
        ridge_w1.append(r.weights[0])
        ridge_w2.append(r.weights[1])
        la = LassoRegression(alpha=0.01, lambda_=lam, max_epochs=500, tolerance=1e-8)
        la.fit(X_l, y_l)
        lasso_w1.append(la.weights[0])
        lasso_w2.append(la.weights[1])

    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4))
    axes4[0].plot(compare_lambdas, ridge_w1, color=TEAL,   lw=2, label="Ridge w₁")
    axes4[0].plot(compare_lambdas, ridge_w2, color=TEAL,   lw=2, linestyle="--", alpha=0.6, label="Ridge w₂")
    axes4[0].plot(compare_lambdas, lasso_w1, color=ORANGE, lw=2, label="Lasso w₁")
    axes4[0].plot(compare_lambdas, lasso_w2, color=ORANGE, lw=2, linestyle="--", alpha=0.6, label="Lasso w₂")
    axes4[0].axhline(0, color="#9ca3af", lw=1, linestyle=":")
    axes4[0].set_title("Ridge vs Lasso — Coefficient Paths", fontweight="bold")
    axes4[0].set_xlabel("λ")
    axes4[0].set_ylabel("Final weight")
    _legend(axes4[0])
    axes4[0].grid(True)

    sparse_lasso = [
        1 if abs(w1) < 0.01 or abs(w2) < 0.01 else 0
        for w1, w2 in zip(lasso_w1, lasso_w2)
    ]
    sparse_ridge = [
        1 if abs(w1) < 0.01 or abs(w2) < 0.01 else 0
        for w1, w2 in zip(ridge_w1, ridge_w2)
    ]
    axes4[1].fill_between(compare_lambdas, sparse_lasso, alpha=0.45, color=ORANGE, label="Lasso: weight ≈ 0")
    axes4[1].fill_between(compare_lambdas, sparse_ridge, alpha=0.30, color=TEAL,   label="Ridge:  weight ≈ 0")
    axes4[1].set_title("Sparsity — When does any weight reach ≈ 0?", fontweight="bold")
    axes4[1].set_xlabel("λ")
    axes4[1].set_ylabel("Sparsity triggered (1 = yes)")
    _legend(axes4[1])
    axes4[1].grid(True)
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

    st.markdown(
        f"""
        <div class="insight-box">
        ✅ <strong>Final Model:</strong> &nbsp; <code>{model_l.equation()}</code>
        &nbsp;&nbsp;|&nbsp;&nbsp; Epochs run: {len(hist_l)}
        &nbsp;&nbsp;|&nbsp;&nbsp; MSE: {hist_l[-1]['mse']:.6f}
        &nbsp;&nbsp;|&nbsp;&nbsp; Loss: {hist_l[-1]['loss']:.6f}
        &nbsp;&nbsp;|&nbsp;&nbsp; λ = {l_lambda}
        </div>
        """,
        unsafe_allow_html=True,
    )
