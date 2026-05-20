import numpy as np

# 1. 原本 2D 圖片（2x2）
X = np.array([
    [10, 20],
    [30, 40]
])

print("原本 shape:", X.shape)
print(X)

X_reshaped = X.reshape(2, 2, 1)

print("\nreshape 後 shape:", X_reshaped.shape)
print(X_reshaped)