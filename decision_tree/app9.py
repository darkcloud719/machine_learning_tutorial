import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================
# 1. Create synthetic dataset
# =====================================================

np.random.seed(42)

X = np.linspace(0, 10, 200).reshape(-1, 1)

# sin wave + noise
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# =====================================================
# 2. Train / test split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# 3. Build models
# =====================================================

tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# =====================================================
# 4. Train models
# =====================================================

tree_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# =====================================================
# 5. Prediction
# =====================================================

# Decision Tree
y_tree_train_pred = tree_model.predict(X_train)
y_tree_test_pred = tree_model.predict(X_test)

# Random Forest
y_rf_train_pred = rf_model.predict(X_train)
y_rf_test_pred = rf_model.predict(X_test)

# =====================================================
# 6. Evaluation Report
# =====================================================

print("=" * 50)
print("Decision Tree Report")
print("=" * 50)

print("Train MSE:", mean_squared_error(y_train, y_tree_train_pred))
print("Test MSE :", mean_squared_error(y_test, y_tree_test_pred))

print("Train R2 :", r2_score(y_train, y_tree_train_pred))
print("Test R2  :", r2_score(y_test, y_tree_test_pred))

print()

print("=" * 50)
print("Random Forest Report")
print("=" * 50)

print("Train MSE:", mean_squared_error(y_train, y_rf_train_pred))
print("Test MSE :", mean_squared_error(y_test, y_rf_test_pred))

print("Train R2 :", r2_score(y_train, y_rf_train_pred))
print("Test R2  :", r2_score(y_test, y_rf_test_pred))

# =====================================================
# 7. Plot prediction curves
# =====================================================

X_range = np.linspace(0, 10, 500).reshape(-1, 1)

y_tree_curve = tree_model.predict(X_range)
y_rf_curve = rf_model.predict(X_range)

plt.figure(figsize=(10, 6))

# original data
plt.scatter(X, y, s=20, alpha=0.5, label="Data")

# Decision Tree
plt.plot(
    X_range,
    y_tree_curve,
    linewidth=2,
    label="Decision Tree"
)

# Random Forest
plt.plot(
    X_range,
    y_rf_curve,
    linewidth=2,
    label="Random Forest"
)

plt.title("Decision Tree vs Random Forest")
plt.xlabel("X")
plt.ylabel("y")

plt.legend()
plt.show()