import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.svm import SVC

# =========================
# 1. Create 2D dataset
# =========================

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# =========================
# 2. Train SVM model
# =========================

model = SVC(
    kernel="rbf",   # nonlinear boundary
    C=1.0,
    gamma=1.0
)

model.fit(X, y)

# =========================
# 3. Create mesh grid (for visualization)
# =========================

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# =========================
# 4. Predict on grid
# =========================

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# =========================
# 5. Plot decision boundary
# =========================

plt.figure(figsize=(8, 6))

# decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")

# support vectors
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    facecolors="none",
    edgecolors="black",
    label="Support Vectors"
)

plt.title("SVM Decision Boundary (RBF Kernel)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()