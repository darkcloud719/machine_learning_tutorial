# k-fold cross validation for logistic regression
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
])

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # model
    model = LogisticRegression()

    # training
    model.fit(X_train, y_train)

    print("\nCoefficients:", model.coef_)
    print("\nIntercept:", model.intercept_)    

    # prediction
    y_pred = model.predict(X_test)

    # evaluation
    acc = accuracy_score(y_test, y_pred)

    scores.append(acc)

print("K-Fold Accuracies:", scores)
print("Mean Accuracy:", np.mean(scores))