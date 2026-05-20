import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# dataset

X = np.array([
    [2,3,],
    [1,1,],
    [2,1,],
    [3,2,],
    [6,5],
    [7,7]
])

y = np.array([0,0,0,0,1,1])

# split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# model
model = Perceptron(max_iter=1000, eta0=0.1)

model.fit(X_train, y_train)

# predict

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Predictions:", y_pred)