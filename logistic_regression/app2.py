import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

texts = [
    "Win money now",
    "Limited offer click now",
    "Congratulations you won prize",
    "Cheap loan available",
    "Hi how are you",
    "Let's have lunch tomorrow",
    "Are you coming to meeting",
    "See you later"
]

labels = [
    1, 1, 1, 1,
    0, 0, 0, 0
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

print(X)

print(vectorizer.get_feature_names_out()[10])