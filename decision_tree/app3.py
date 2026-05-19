import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. load dataset

iris = load_iris()

X = iris.data
y = iris.target

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# 2. split data

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create model
model = GradientBoostingClassifier(
    n_estimators=100, # number of trees
    learning_rate=0.1, # step size
    max_depth=3, # depth of each tree
    random_state=42
)

# 4. Train model
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluation
acc = accuracy_score(y_test, y_pred)

print("Results")
print("Accuracy:", acc)
print("Classification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# 7. Feature importance
print("Feature Imprtance")
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(name, ":", importance)
    
sample = np.array([[5.1,3.5,1.4,0.2]])

pred_class = model.predict(sample)
pred_proba = model.predict_proba(sample)

print("New Sample Prediction")
print("Predicted class:", iris.target_names[pred_class][0])
print("Probabilities:", pred_proba)