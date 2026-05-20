import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# 3. Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Mean:")
print(np.mean(X_train, axis=0))

print("\nStandard Deviation:")
print(np.std(X_train, axis=0))

# 4. Train LDA classifier
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# 5. Predict
y_pred = lda.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))