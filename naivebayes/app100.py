import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 載入資料
iris = load_iris()
X = iris.data
y = iris.target

# 2. 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. 建立 Naive Bayes 模型
model = GaussianNB()

# 4. 訓練模型
model.fit(X_train, y_train)

# 5. 預測
y_pred = model.predict(X_test)

# 6. 評估
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))