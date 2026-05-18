import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. True values (ground truth)
y_true = np.array([100,200,300,400])

# 2. Predicted values (model output)
y_pred = np.array([110,190,310,380])

# 3. MSE(Mean Squared Error) 均方誤差
mse = mean_squared_error(y_true, y_pred)

# 4. RMSE(Root Mean Squared Error) 均方根誤差
rmse = np.sqrt(mse)

# 5. MAE(Mean Absolute Error) 平均絕對誤差
mae = mean_absolute_error(y_true, y_pred)

# 6. MAPE(Mean Absolute Percentage Error) 平均絕對百分比誤差
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 7. R² Score (Coefficient of Determination) 決定係數
r2 = r2_score(y_true, y_pred)

# 8. Print results
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape, "%")
print("R² Score:", r2)