import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# data
X = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
])

y = np.array([0,0,0,0,0,1,1,1,1,1])

# =========================================================
# 1️⃣ WITHOUT K-FOLD (single train/test split)
# =========================================================

print("\n================ No K-FOLD ================\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc_single = accuracy_score(y_test, y_pred)

print("Single Split Accuracy:", acc_single)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# =========================================================
# 2️⃣ MANUAL K-FOLD
# =========================================================

print("\n================ MANUAL K-FOLD ================\n")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores_manual = []

fold = 1

for train_index, test_index in kf.split(X_train):

    X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    model = LogisticRegression()
    model.fit(X_train_fold, y_train_fold)

    y_pred = model.predict(X_test_fold)   # ✔ 修正這裡

    acc = accuracy_score(y_test_fold, y_pred)
    scores_manual.append(acc)

    print(f"Fold {fold} Accuracy: {acc}")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
    print("-" * 40)

    fold += 1


# =========================================================
# 3️⃣ AUTO K-FOLD (cross_val_score)
# =========================================================

# 這是stratified K-Fold，會保持類別比例
print("\n================ AUTO K-FOLD (cross_val_score) ================\n")

print("len(X_train):", len(X_train))

model = LogisticRegression()

scores_auto = cross_val_score(
    model,
    X_train,
    y_train,
    cv=4,
    scoring='accuracy'
)

print("Auto CV Scores:", scores_auto)
print("Mean Auto CV:", np.mean(scores_auto))


# =========================================================
# 4️⃣ FINAL COMPARISON
# =========================================================

print("\n================ FINAL COMPARISON ================\n")

print("Single Split Accuracy:", acc_single)
print("Manual K-Fold Mean:", np.mean(scores_manual))
print("Auto K-Fold Mean:", np.mean(scores_auto))