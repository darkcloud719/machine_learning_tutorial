import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lightgbm import LGBMClassifier

# =========================
# 1. Load dataset
# =========================

iris = load_iris()
X = iris.data
y = iris.target

# =========================
# 2. Train / Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3. Build LightGBM model
# =========================

model = LGBMClassifier(
    n_estimators=100,        # number of trees
    learning_rate=0.1,       # step size
    max_depth=3,             # tree depth
    subsample=0.8,           # row sampling
    colsample_bytree=0.8,    # feature sampling
    random_state=42
)

# =========================
# 4. Train model
# =========================

model.fit(X_train, y_train)

# =========================
# 5. Predict
# =========================

y_pred = model.predict(X_test)

# =========================
# 6. Evaluation
# =========================

print("=== Test Set Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================
# 7. Cross Validation
# =========================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\n=== Cross Validation ===")
print("Scores:", cv_scores)
print("Mean:", np.mean(cv_scores))
print("Std:", np.std(cv_scores))