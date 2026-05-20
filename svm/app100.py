import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# =====================================================
# 1. 產生非線性資料
# =====================================================

np.random.seed(42)

X = np.linspace(0, 10, 200).reshape(-1, 1)

# 真實 function + noise
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# =====================================================
# 2. 切分資料
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# 3. 建立 SVR 模型
# =====================================================

model = SVR(
    kernel="rbf",   # 非線性核心
    C=10,           # 控制錯誤懲罰
    epsilon=0.1     # ε-tube 容忍區
)

# =====================================================
# 4. 訓練模型
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# 5. 預測
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# 6. 視覺化 SVR 曲線
# =====================================================

X_grid = np.linspace(0, 10, 500).reshape(-1, 1)
y_grid = model.predict(X_grid)

plt.figure(figsize=(8, 6))

# 原始資料
plt.scatter(X, y, color="blue", label="Data")

# SVR 預測曲線
plt.plot(X_grid, y_grid, color="red", linewidth=2, label="SVR prediction")

plt.title("Support Vector Regression (SVR)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()