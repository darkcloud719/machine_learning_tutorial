# PCA Principal Component Analysis

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Standardization (VERY important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA (reduce to 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Print explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", sum(pca.explained_variance_ratio_))

# 5. Visualization
plt.figure(figsize=(8,6))

colors = ["red", "green", "blue"]

for i, color, name in zip([0,1,2], colors, target_names):
    plt.scatter(
        X_pca[y==i,0],
        X_pca[y==i,1],
        color=color,
        label=name,
        edgecolors='k'
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.legend()
plt.grid()
plt.show()