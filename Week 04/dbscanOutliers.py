import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Generate normal clustered data
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.6,
    random_state=42
)

# Add some artificial outliers
outliers = np.array([
    [8, 8],
    [-8, -8],
    [10, -5],
    [-10, 6],
    [0, 10],
    [12, 12]
])

# Combine normal data and outliers
X = np.vstack([X, outliers])

# Scale the data
X_scaled = StandardScaler().fit_transform(X)

# Apply DBSCAN
model = DBSCAN(eps=0.3, min_samples=5)
labels = model.fit_predict(X_scaled)

# Separate clusters and noise
noise_mask = labels == -1

print("Detected outliers:", noise_mask.sum())
print("Outlier coordinates:")
print(X[noise_mask])

# Plot results
plt.figure(figsize=(9, 6))

# Plot clustered points
plt.scatter(
    X_scaled[~noise_mask, 0],
    X_scaled[~noise_mask, 1],
    c=labels[~noise_mask],
    cmap="viridis",
    label="Clusters"
)

# Plot detected outliers in red
plt.scatter(
    X_scaled[noise_mask, 0],
    X_scaled[noise_mask, 1],
    color="red",
    marker="X",
    s=150,
    label="Detected Outliers"
)

plt.title("DBSCAN: Clusters and Detected Outliers")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()