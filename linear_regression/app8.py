# Linear Regression and Lasso Regression comparison
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
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
linear_overfit = make_pipeline(
    PolynomialFeatures(degree=15),
    LinearRegression()
)

lasso_model = make_pipeline(
    PolynomialFeatures(degree=15),
    Lasso(alpha=0.1, max_iter=10000)
)

# =========================
# 4. Train both models
# =========================
linear_overfit.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# =========================
# 5. Evaluate
# =========================
print("=== Linear (Overfitting prone) ===")
print("Train score:", linear_overfit.score(X_train, y_train))
print("Test score:", linear_overfit.score(X_test, y_test))

print("\n=== Lasso (Regularized L1) ===")
print("Train score:", lasso_model.score(X_train, y_train))
print("Test score:", lasso_model.score(X_test, y_test))

# =========================
# 6. Model parameters
# =========================
print("\n=== Model Parameters ===")
print("Linear Coefficients:", linear_overfit.named_steps['linearregression'].coef_)
print("Linear Intercept:", linear_overfit.named_steps['linearregression'].intercept_)

print("Lasso Coefficients:", lasso_model.named_steps['lasso'].coef_)
print("Lasso Intercept:", lasso_model.named_steps['lasso'].intercept_)