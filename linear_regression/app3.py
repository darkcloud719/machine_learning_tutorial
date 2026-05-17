from sklearn.linear_model import LinearRegression

# =========================
# 1. Prepare the dataset
# =========================
# X = features (years of experience, education level, skill score)
# Each row represents one sample (one person)
X = [
    [1, 1, 50],
    [3, 2, 60],
    [5, 2, 70],
    [7, 3, 85],
    [10, 3, 95]
]

# y = target values (salary in thousand dollars)
# Each value corresponds to one row in X
y = [40, 60, 80, 110, 150]

# =========================
# 2. Create the model
# =========================
# LinearRegression learns a linear relationship:
# salary = w1 * experience + w2 * education + w3 * skill + b
model = LinearRegression()

# =========================
# 3. Train the model
# =========================
# fit() finds the best parameters (coefficients and intercept)
# by minimizing the Mean Squared Error (MSE)
model.fit(X, y)

# =========================
# 4. Make a prediction
# =========================
# Predict salary for a new employee:
# 6 years of experience, education level 2, skill score 80
prediction = model.predict([[6, 2, 80]])
print("Predicted salary:", prediction)

# =========================
# 5. Model parameters
# =========================

# coef_ = weights for each feature
# It shows how much each feature affects the prediction
print("Coefficients:", model.coef_)

# intercept_ = bias term
# It represents the base salary when all features are zero
print("Intercept:", model.intercept_)

print("The score of the model (R^2):", model.score(X, y))