import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Create synthetic regression dataset

np.random.seed(42)

X = np.linspace(0,10,500).reshape(-1,1)
y = np.sin(X).ravel() + np.random.normal(0,0.2,X.shape[0])

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 2. Build GradientBoostingRegressor
model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# 3. Calculate deviance for each boosting iteration

train_deviance = []
test_deviance = []

for y_pred_train, y_pred_test in zip(
    model.staged_predict(X_train),
    model.staged_predict(X_test)
    # model.predict(X_train),
    # model.predict(X_test)
):
    train_deviance.append(mean_squared_error(y_train, y_pred_train))
    test_deviance.append(mean_squared_error(y_test, y_pred_test))

# # 4. Plot
plt.figure(figsize=(7,5))

plt.plot(train_deviance, color="blue", label="Training Set Deviance")
plt.plot(test_deviance, color="red", label="Test Set Deviance")

plt.title("Deviance")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")

plt.legend()
plt.show()