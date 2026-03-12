"""
AIstats_lab.py

Student starter file for the Regularization & Overfitting lab.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# =========================
# Helper Functions
# =========================

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# =========================
# Q1 Lasso Regression
# =========================

def lasso_regression_diabetes(lambda_reg=0.1, lr=0.01, epochs=2000):
    """
    Implement Lasso regression using gradient descent with L1 regularization.

    L(θ) = MSE + λ‖θ‖₁
    Subgradient of L1: ∂|θ| = sign(θ)

    Returns: train_mse, test_mse, train_r2, test_r2, theta
    """

    # 1. Load diabetes dataset
    X, y = load_diabetes(return_X_y=True)

    # 2. Train/test split (80/20, fixed seed for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Standardize features (fit on train, transform both)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # 4. Add bias column (intercept term)
    X_train_b = add_bias(X_train)  # shape: (n_train, n_features + 1)
    X_test_b  = add_bias(X_test)   # shape: (n_test,  n_features + 1)

    n_samples, n_features = X_train_b.shape

    # 5. Initialize theta to zeros
    theta = np.zeros(n_features)

    # 6. Gradient descent with L1 regularization
    #    Gradient of MSE  : (2/n) * Xᵀ(Xθ − y)
    #    L1 subgradient   : λ * sign(θ)
    #    Note: we do NOT regularize the bias term (index 0)
    for _ in range(epochs):
        y_pred = X_train_b @ theta                        # (n,)
        residuals = y_pred - y_train                      # (n,)

        grad_mse = (2 / n_samples) * (X_train_b.T @ residuals)  # (p,)

        # L1 subgradient — skip bias term
        l1_grad = lambda_reg * np.sign(theta)
        l1_grad[0] = 0.0                                  # no penalty on bias

        theta -= lr * (grad_mse + l1_grad)

    # 7. Compute predictions
    y_train_pred = X_train_b @ theta
    y_test_pred  = X_test_b  @ theta

    # 8. Compute metrics
    train_mse = mse(y_train, y_train_pred)
    test_mse  = mse(y_test,  y_test_pred)
    train_r2  = r2_score(y_train, y_train_pred)
    test_r2   = r2_score(y_test,  y_test_pred)

    return train_mse, test_mse, train_r2, test_r2, theta


# =========================
# Q2 Polynomial Overfitting
# =========================

def polynomial_overfitting_experiment(max_degree=10):
    """
    Study overfitting using polynomial regression on the BMI feature.

    Fits models of increasing degree using the normal equation:
        θ = (XᵀX)⁻¹ Xᵀy

    Returns:
        {
            "degrees":   list of int,
            "train_mse": list of float,
            "test_mse":  list of float,
        }
    """

    # 1. Load dataset
    X, y = load_diabetes(return_X_y=True)

    # 2. Select BMI feature only (index 2), keep as 2-D column
    X_bmi = X[:, 2].reshape(-1, 1)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bmi, y, test_size=0.2, random_state=42
    )

    degrees     = []
    train_errors = []
    test_errors  = []

    # 4. Loop through polynomial degrees 1 → max_degree
    for degree in range(1, max_degree + 1):

        # 5. Create polynomial features (includes bias term via include_bias=True)
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        X_train_poly = poly.fit_transform(X_train)   # fit on train only
        X_test_poly  = poly.transform(X_test)

        # 6. Fit using normal equation: θ = (XᵀX)⁻¹ Xᵀy
        #    Use lstsq for numerical stability (handles near-singular matrices)
        theta, _, _, _ = np.linalg.lstsq(X_train_poly, y_train, rcond=None)

        # 7. Compute predictions and MSE
        y_train_pred = X_train_poly @ theta
        y_test_pred  = X_test_poly  @ theta

        degrees.append(degree)
        train_errors.append(mse(y_train, y_train_pred))
        test_errors.append(mse(y_test,  y_test_pred))

    return {
        "degrees":   degrees,
        "train_mse": train_errors,
        "test_mse":  test_errors,
    }
