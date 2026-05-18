import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. Dataset (spam vs ham)
# ============================================================
texts = [
    "Win money now",                # spam
    "Limited offer click now",      # spam
    "Congratulations you won prize",# spam
    "Cheap loan available",         # spam
    "Hi how are you",               # ham
    "Let's have lunch tomorrow",    # ham
    "Are you coming to meeting",    # ham
    "See you later",                # ham
]

# labels: 1 = spam, 0 = ham
labels = [1, 1, 1, 1, 0, 0, 0, 0]

# ============================================================
# 2. Text → TF-IDF features
# ============================================================
vectorizer = TfidfVectorizer()

# convert text into numeric matrix
X = vectorizer.fit_transform(texts)

# show vocabulary size and words
print("Number of features:", len(vectorizer.get_feature_names_out()))
print("Feature names:", vectorizer.get_feature_names_out())

# convert sparse matrix to dense (for viewing)
print("\nTF-IDF matrix:\n", X.toarray())

# ============================================================
# 3. Train / Test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)

# ============================================================
# 4. Train Logistic Regression model
# ============================================================
model = LogisticRegression()
model.fit(X_train, y_train)

# ============================================================
# 5. Predict on test set
# ============================================================
y_pred = model.predict(X_test)

# ============================================================
# 6. Evaluation
# ============================================================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# 7. Predict new email
# ============================================================
new_text = ["Congratulations your won money now"]

# transform using same TF-IDF model
new_X = vectorizer.transform(new_text)

# predicted class (0 or 1)
prediction = model.predict(new_X)

# predicted probability [ham, spam]
prediction_proba = model.predict_proba(new_X)

print("\nPrediction probabilities:", prediction_proba)

# final label output
print("\nPrediction:", "SPAM" if prediction[0] == 1 else "HAM")