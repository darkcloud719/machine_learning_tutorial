# Ridge Regression example
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split

# 1. Create noisy dataset
np.random.seed(0)

X = np.linspace(-3, 3, 50).reshape(-1,1)

# non-linear true function + noise
y = X**2 + np.random.randn(50, 1) * 2
y = y.ravel()

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Linear Regression (without regularization , baseline)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print("Linear Regression Train score (R^2):", linear_model.score(X_train, y_train))
print("Linear Regression Test score (R^2):", linear_model.score(X_test, y_test))
# 4. Ridge Regression (with L2 regularization)
ridge_model = Ridge(alpha=1.0) # alpha = regularization strength and controls penalty strength
ridge_model.fit(X_train, y_train)
print("Ridge Regression Train score (R^2):", ridge_model.score(X_train, y_train))
print("Ridge Regression Test score (R^2):", ridge_model.score(X_test, y_test))

# 5. Compare coefficients
print("Linear Regression Coefficients:", linear_model.coef_)
print("Linear Regression Intercept:", linear_model.intercept_)
print("Ridge Regression Coefficients:", ridge_model.coef_)
print("Ridge Regression Intercept:", ridge_model.intercept_)