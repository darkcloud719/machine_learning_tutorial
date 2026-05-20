import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Load data
iris = load_iris()
X = iris.data
y = iris.target

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis')
plt.title("t-SNE Visualization")
plt.show()