import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Generate dataset
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# Scale data
X_scaled = StandardScaler().fit_transform(X)

# Apply K-Means
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(X_scaled)

# Calculate Silhouette Score
score = silhouette_score(X_scaled, labels)

print("Silhouette Score:", score)

# Visualize clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis"
)

plt.title("K-Means Clustering")
plt.show()