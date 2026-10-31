import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate sample data with three natural groups
X, _ = make_blobs(
    n_samples=100,
    centers=3,
    random_state=42
)

# Scale the features
X_scaled = StandardScaler().fit_transform(X)

# Perform hierarchical clustering
model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

labels = model.fit_predict(X_scaled)

# Plot the discovered clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels
)
plt.title("Agglomerative Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Build and plot the dendrogram
linkage_matrix = linkage(X_scaled, method="ward")

plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()