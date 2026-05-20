# 指定k
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

# 1. create dataset

X, y_true = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

# 2. build model

model = AgglomerativeClustering(n_clusters=3)

labels = model.fit_predict(X)

# 3. plot result
plt.figure(figsize=(8,6))

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', edgecolor='k')

plt.title("Agglomerative Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
