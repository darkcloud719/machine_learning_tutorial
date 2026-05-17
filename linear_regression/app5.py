# Overfitting example with polynomial regression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# =========================
# 1. Create noisy NON-linear dataset
# =========================
np.random.seed(0)

X = np.linspace(-3, 3, 50).reshape(-1, 1)

# true function + noise
y = X**2 + np.random.randn(50, 1) * 2
y = y.ravel()

# =========================
# 2. Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 3. OVERFITTING MODEL
# =========================
# We intentionally make the model too complex:
# Polynomial degree = 15 (very high)
model = make_pipeline(
    PolynomialFeatures(degree=15),
    LinearRegression()
)

model.fit(X_train, y_train)

# =========================
# 4. Evaluate
# =========================
print("Train score (R^2):", model.score(X_train, y_train))
print("Test score (R^2):", model.score(X_test, y_test))

print("Coefficients:", model.named_steps['linearregression'].coef_)
print("Intercept:", model.named_steps['linearregression'].intercept_)