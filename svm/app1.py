import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. load dataset

iris = load_iris()
X = iris.data
y = iris.target

# 2. train / test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Build SVM model

model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    random_state=42
)

# 4. Train model

model.fit(X_train, y_train)

# 5. Predict

y_pred = model.predict(X_test)

# 6. Evaluation

print("==== TEst Set Evaluation ====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Cross Validation

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\n=== Cross Validation ===")
print("scores:", cv_scores)
print("Mean Accuracy:", np.mean(cv_scores))
print("Std:", np.std(cv_scores))