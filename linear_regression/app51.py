import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# 1. Create dataset (binary classification problem)
# ============================================================
np.random.seed(0)

# Feature: 2D data
X = np.random.randn(200, 2)

# True rule (simple linear boundary):
# if x0 + x1 > 0 → class 1 else class 0
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# ============================================================
# 2. Split dataset
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================================
# 3. Build Logistic Regression model
# ============================================================
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# ============================================================
# 4. Predict
# ============================================================
y_pred = model.predict(X_test)

# ============================================================
# 5. Evaluate
# ============================================================
print("Accuracy:", accuracy_score(y_test, y_pred))

# ============================================================
# 6. Model parameters
# ============================================================
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)