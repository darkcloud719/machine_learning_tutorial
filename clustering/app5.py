# LDA (Linear Discriminant Analysis) 降維+視覺化
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 1. Load dataset

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Standardization (Recommended for LDA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply LDA (reduce to 2D)
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# 4. Explained variance (optional insight)
print("Explained variance ratio:", lda.explained_variance_ratio_)
print("Total explained variance:", sum(lda.explained_variance_ratio_))

# 5. Visualization
plt.figure(figsize=(8,6))

colors = ["red", "green", "blue"]

for i, color, name in zip([0,1,2], colors, target_names):
    plt.scatter(
        X_lda[y==i,0],
        X_lda[y==i,1],
        color=color,
        label=name,
        edgecolors='k'
    )

plt.xlabel("LD1 (Linear Discriminant 1)")
plt.ylabel("LD2 (Linear Discriminant 2)")
plt.title("LDA on Iris Dataset")
plt.legend()
plt.grid()
plt.show()