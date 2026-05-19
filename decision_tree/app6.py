import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Create synthetic regression dataset

np.random.seed(42)

X = np.linspace(0, 10, 200). reshape(-1, 1)

# nonlinear function + noise

y = np.sin(X).ravel() + 0.3 * np.random.randn(200)

# 2. Train / test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Build GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,     # number of trees
    learning_rate=0.1,    # shrink step size
    max_depth=3,          # tree depth
    random_state=42
)

# 4. Train model

model.fit(X_train, y_train)

# 5. Predict

y_pred = model.predict(X_test)

# 6. Evaluation

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)

# 7. Visualization

X_range = np.linspace(0,10,500).reshape(-1,1)
y_range_pred = model.predict(X_range)

plt.scatter(X, y, label="True Data", alpha=0.5)
plt.plot(X_range, y_range_pred, color="red", label="GBDT prediction")

plt.title("Gradient Boosting Regression")
plt.legend()
plt.show()