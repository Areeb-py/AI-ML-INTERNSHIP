import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap import UMAP

# Generate dataset with 10 features
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    random_state=42
)

# Scale data
X_scaled = StandardScaler().fit_transform(X)

# Apply t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled)

# Apply UMAP
umap = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

X_umap = umap.fit_transform(X_scaled)

# Plot t-SNE
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis")
plt.title("t-SNE Visualization")

# Plot UMAP
plt.subplot(1, 2, 2)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="viridis")
plt.title("UMAP Visualization")

plt.show()