from sklearn import linear_model

# =========================
# 1. Create Linear Regression Model
# =========================
# LinearRegression is used to learn the relationship:
# y = w1*x1 + w2*x2 + b
reg = linear_model.LinearRegression()

# =========================
# 2. Prepare Training Data
# =========================
# x = features (inputs)
# Each sample has 2 features (x1, x2)
x = [
    [0, 0],
    [1, 1],
    [2, 2]
]

# y = target values (labels)
# The model learns the mapping from x → y
y = [0, 1, 2]

# =========================
# 3. Train the Model (Learning Process)
# =========================
# fit = find the best parameters (coef_, intercept_)
# Objective: minimize Mean Squared Error (MSE)
reg.fit(x, y)

# =========================
# 4. Make Predictions
# =========================
# predict = use the learned model to predict new data
# [3,3] is a new input sample
print(reg.predict([[3, 3]]))

# =========================
# 5. Model Parameters
# =========================

# coef_ = weights for each feature (importance)
# Represents w1 and w2 in the equation:
# y = w1*x1 + w2*x2 + b
print(reg.coef_)

# intercept_ = bias term
# Represents the offset (b) in the equation:
# y = w1*x1 + w2*x2 + b
print(reg.intercept_)