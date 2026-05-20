# DBSCAN

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# 1. Create dataset (non-linear shape)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# 2. Build DBSCAN model

model = DBSCAN(
    eps=0.15,        # Neighborhood radius
    min_samples=5   # Minimum number of points
)

# 3. Train + Predict
labels = model.fit_predict(X)

# 4. Count clusters (including noise)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Number of clusters:", n_clusters)
print("Noise points:", np.sum(labels == -1))

# 5. Visualization
plt.figure(figsize=(8,6))

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', edgecolors='k')

plt.title(f"DBSCAN Clustering (clusters={n_clusters})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()