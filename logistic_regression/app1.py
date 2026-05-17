import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================
# 1. Create dataset (binary classification problem)
# ============================================================
np.random.seed(0)

# Generate 200 samples, each with 2 features
# Each feature follows a standard normal distribution (mean=0, std=1)
X = np.random.randn(200, 2)

# Create binary labels using a simple linear rule:
# If (x1 + x2 > 0) → class 1
# Else → class 0
y = (X[:,0] + X[:,1] > 0).astype(int)

# ============================================================
# 2. Split dataset into training and testing sets
# ============================================================
# 70% training data, 30% test data
# random_state ensures reproducibility (same split every run)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================================
# 3. Build Logistic Regression model
# ============================================================
# Logistic Regression is a linear classification model
# It learns weights (coef_) and bias (intercept_)
model = LogisticRegression()

# Train the model using training data
model.fit(X_train, y_train)

# ============================================================
# 4. Evaluate (commented out)
# ============================================================
# Option 1: model.score() → returns accuracy for classification
print("Accuracy:", model.score(X_test, y_test))

# Option 2: manual prediction + accuracy calculation
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# ============================================================
# 5. Inspect learned parameters
# ============================================================
# coef_ → weights learned for each feature
# intercept_ → bias term
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)