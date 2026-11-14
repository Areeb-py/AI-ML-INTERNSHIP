# Import required libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data containing three natural groups
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the K-Means model
model = KMeans(
    n_clusters=3,  # Number of clusters
    random_state=42,
    n_init=10
)

# Train the model and assign cluster labels
labels = model.fit_predict(X_scaled)

# Display the clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels
)

# Display cluster centers
centers = model.cluster_centers_

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()