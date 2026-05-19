import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. create data
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# 2. build model
kmeans = KMeans(n_clusters=3, random_state=42)

# 3. train
kmeans.fit(X)

# 4. predict cluster
labels = kmeans.labels_

# 5. cluster centers
centers = kmeans.cluster_centers_

# 6. plot
plt.figure(figsize=(8,6))

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(centers[:,0], centers[:,1], s=200, c='red', marker='X')

plt.title("K-Means Clustering")
plt.show()