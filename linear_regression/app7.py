# Linear Regression and Ridge Regression comparison
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# =========================
# 1. Create complex dataset
# =========================
np.random.seed(0)

X = np.linspace(-3, 3, 50).reshape(-1, 1)

# nonlinear function + noise
y = X**2 + np.random.randn(50, 1) * 2
y = y.ravel()

# =========================
# 2. Train / test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 3. Make model TOO powerful (overfitting setup)
# =========================
# High-degree polynomial → very flexible model
linear_overfit = make_pipeline(
    PolynomialFeatures(degree=15),
    LinearRegression()
)

ridge_model = make_pipeline(
    PolynomialFeatures(degree=15),
    Ridge(alpha=1.0)
)

# =========================
# 4. Train both models
# =========================
linear_overfit.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# =========================
# 5. Evaluate
# =========================
print("=== Linear (Overfitting prone) ===")
print("Train score:", linear_overfit.score(X_train, y_train))
print("Test score:", linear_overfit.score(X_test, y_test))

print("\n=== Ridge (Regularized) ===")
print("Train score:", ridge_model.score(X_train, y_train))
print("Test score:", ridge_model.score(X_test, y_test))

# =========================
# 6. Model parameters
# =========================
print("\n=== Model Parameters ===")
print("Linear Regression Coefficients:", linear_overfit.named_steps['linearregression'].coef_)
print("Linear Regression Intercept:", linear_overfit.named_steps['linearregression'].intercept_)
print("Ridge Regression Coefficients:", ridge_model.named_steps['ridge'].coef_)
print("Ridge Regression Intercept:", ridge_model.named_steps['ridge'].intercept_)