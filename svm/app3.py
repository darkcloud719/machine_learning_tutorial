import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# 1. Generate non-linear data

np.random.seed(42)

X = np.linspace(0,10,200).reshape(-1,1)

# function + noise
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# 2. Train / Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 3. Build SVR model
model = SVR(
    kernel="rbf",   # non-linear kernel
    C=10,           # error penalty
    epsilon=0.1     # ε-tube width
)

# 4. Train model

model.fit(X_train, y_train)

# 5. Predict

y_pred = model.predict(X_test)

# 6. Visualize SVR curve

X_grid = np.linspace(0,10,500).reshape(-1,1)
y_grid = model.predict(X_grid)

plt.figure(figsize=(8,6))

# Original data
plt.scatter(X, y, color="blue", label="Data")

# SVR prediction
plt.plot(X_grid, y_grid, color="red", linewidth=2, label="SVR Prediction")


plt.title("Support Vector Regression (SVR)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()