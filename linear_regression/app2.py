from sklearn.linear_model import LinearRegression

# =========================
# 1. Prepare the dataset
# =========================
# X = features (house size, number of rooms)
# Each row represents one sample
x = [
    [20, 1],
    [30, 2],
    [40, 2],
    [50, 3]
]

# y = target values (house prices)
# Each value corresponds to one row in X
y = [300, 450, 600, 800]

# =========================
# 2. Create the model
# =========================
# LinearRegression learns a linear relationship:
# price = w1 * size + w2 * rooms + b
model = LinearRegression()

# =========================
# 3. Train the model
# =========================
# fit() finds the optimal parameters (weights and bias)
# by minimizing the Mean Squared Error (MSE)
model.fit(x, y)

# =========================
# 4. Make a prediction
# =========================
# Predict the price of a new house:
# 35 square meters, 2 rooms
prediction = model.predict([[35, 2]])
print("Predicted price:", prediction)

# =========================
# 5. Model parameters
# =========================

# coef_ = weights for each feature
# It shows how much each feature contributes to the prediction
print("Coefficients:", model.coef_)

# intercept_ = bias term
# It represents the base value when all features are zero
print("Intercept:", model.intercept_)