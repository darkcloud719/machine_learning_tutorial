# Underfitting example with linear regression
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# =========================
# 1. Create NON-linear dataset
# =========================
# True function: y = x^2 + noise
# This is a NON-linear relationship
np.random.seed(0)

X = np.linspace(-3, 3, 50).reshape(-1, 1)

# Add quadratic pattern + noise
y = X**2 + np.random.randn(50, 1) * 2

# Convert y to 1D array (required by sklearn)
y = y.ravel()

# =========================
# 2. Split dataset into train/test
# =========================
# Train set: used for learning
# Test set: used for evaluation (unseen data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 3. Build Linear Regression model
# =========================
# Model assumption:
# y = w * x + b
# BUT real data is x^2 (non-linear)
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 4. Evaluate model performance
# =========================
print("Train score (R^2):", model.score(X_train, y_train))
print("Test score (R^2):", model.score(X_test, y_test))

# =========================
# 5. Model parameters
# =========================
# coef_ = weight (slope of line)
# intercept_ = bias (y-intercept)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
