from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# ============================================================
# 1. Sample text data (documents)
# ============================================================
texts = [
    "I love machine learning",
    "I love AI",
    "machine learning is fun"
]

# ============================================================
# 2. Create TF-IDF vectorizer
# ============================================================
vectorizer = TfidfVectorizer()

# Fit vocabulary and transform text into numerical features
# This converts text → TF-IDF matrix
X = vectorizer.fit_transform(texts)

# ============================================================
# 3. Print raw TF-IDF sparse matrix
# ============================================================
# This is a sparse matrix representation:
# (row_index, column_index) -> TF-IDF value
print(X)

# ============================================================
# 4. Get feature names (words)
# ============================================================
# This shows which word corresponds to each column index
# Example: column 0 = "ai", column 1 = "fun", etc.
print(vectorizer.get_feature_names_out())

# ============================================================
# 5. Convert sparse matrix to dense DataFrame
# ============================================================
# X.toarray() converts sparse matrix → full numeric matrix
# columns=feature names makes it human-readable
df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

# ============================================================
# 6. Print readable table
# ============================================================
# Each row = one sentence
# Each column = TF-IDF score of a word
print(df)

# ============================================================
# 7. Print matrix shape
# ============================================================
# (number of documents, number of unique words)
# rows = sentences, columns = vocabulary size
print(X.shape)