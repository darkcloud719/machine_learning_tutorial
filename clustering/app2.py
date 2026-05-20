import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# 1. Create dataset

X, y_true = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

# 2. Build model (without specifying K)
model = AgglomerativeClustering(
    n_clusters=None,        # Without specifying number of clusters,
    distance_threshold=4.0  # Use distance to determine when to stop
)

# 3. Train + Predict
labels = model.fit_predict(X)

# 4. Calculate number of clusters
n_clusters = len(np.unique(labels))

print("=== Clustering Result ===")
print("Number of clusters:", n_clusters)
print("Cluster labels:", np.unique(labels))

# 5. Visualization
plt.figure(figsize=(8,6))

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', edgecolor='k')

plt.title(f"Agglomerative Clustering (clusters = {n_clusters})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.show()