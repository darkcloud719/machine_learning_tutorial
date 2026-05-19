import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create synthetic dataset

np.random.seed(42)

X = np.linspace(0,10,200).reshape(-1,1)

# sin wave + noise
y = np.sin(X).ravel() + np.random.normal(0,0.2,X.shape[0])

# 2. Build models

# Single Decision Tree
tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

# Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

# 3. Train models

tree_model.fit(X, y)
rf_model.fit(X, y)

# 4. Prediction

X_range = np.linspace(0,10,500).reshape(-1,1)

y_tree_pred = tree_model.predict(X_range)
y_rf_pred = rf_model.predict(X_range)

# 5. Plot

plt.figure(figsize=(10,6))

# Original data
plt.scatter(X, y, s=20, alpha=0.5, label="Data")

# Decision Tree prediction
plt.plot(
    X_range,
    y_tree_pred,
    label="Decision Tree",
    linewidth=2
)

# Random Forest prediction
plt.plot(
    X_range,
    y_rf_pred,
    label="Random Forest",
    linewidth=2
)

plt.title("Decision Tree vs Random Forest")
plt.xlabel("X")
plt.ylabel("y")

plt.legend()
plt.show()