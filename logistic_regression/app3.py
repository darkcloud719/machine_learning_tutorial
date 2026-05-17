import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# 1. Simple dataset (spam vs ham)
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

labels = [
    1, 1, 1, 1,   # spam
    0, 0, 0, 0    # ham
]

# ============================================================
# 2. Convert text → numerical features (TF-IDF)
# ============================================================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# ============================================================
# 3. Train / test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)

# ============================================================
# 4. Build Logistic Regression model
# ============================================================
model = LogisticRegression()
model.fit(X_train, y_train)

# ============================================================
# 5. Predict
# ============================================================
y_pred = model.predict(X_test)

# ============================================================
# 6. Evaluation
# ============================================================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# 7. Test on new email
# ============================================================
new_text = ["Congratulations you won money now"]
new_X = vectorizer.transform(new_text)

prediction = model.predict(new_X)

print("\nNew email prediction:", "SPAM" if prediction[0] == 1 else "HAM")