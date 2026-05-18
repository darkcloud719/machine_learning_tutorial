import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset

iris = load_iris()

X = iris.data
y = iris.target

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("Data shape:", X.shape)

# 2. Split data

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Train Decision Tree
model = DecisionTreeClassifier(
    max_depth=3, # Control tree depth to avoid overfitting
    random_state=42
)

model.fit(X_train, y_train)

# 4. Prediction
y_pred = model.predict(X_test)

# 5. Evaluation
acc = accuracy_score(y_test, y_pred)

print("\n================ Results ================\n")
print("Accuracy:", acc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Feature Importance
print("\nFeature Importance")
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(name, ":", importance)

# 7. Predict new sample
sample = np.array([[5.1,3.5,1.4,0.2]])

pred_class = model.predict(sample)
print("\nPredicted class index:", pred_class[0])
pred_proba = model.predict_proba(sample)

print("\n================ New Sample Prediction ===============\n")
print("Predicted class:", iris.target_names[pred_class][0])
print("Probabilities:", pred_proba)


# print(iris.feature_names)
# print(model.feature_importances_)

# a = zip(iris.feature_names, model.feature_importances_)

# for n, i in a:
#     print(n, ":", i)