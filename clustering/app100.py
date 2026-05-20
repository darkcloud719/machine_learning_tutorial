import umap
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# data
iris = load_iris()
X = iris.data
y = iris.target

# standardize
X_scaled = StandardScaler().fit_transform(X)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# plot
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
plt.title("UMAP Visualization")
plt.show()