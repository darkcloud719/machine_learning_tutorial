import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset

iris = load_iris()

X = iris.data
y = iris.target

# Model 

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

# 3. Stratified K-Fold

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 4. Cross validation accuracy

scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("==== Cross Validation Scores ====")
print("Accuracies:", scores)
print("Mean Accuracy:", np.mean(scores))
print("std:", np.std(scores))

# 5. Train final model (optional for report)

rf_model.fit(X,y)
y_pred = rf_model.predict(X)

print("\n==== Full dataset report (for reference only) ====")
print("Accuracy:", accuracy_score(y, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))