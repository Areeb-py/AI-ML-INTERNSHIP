import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Generate sample data
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

# Scale data
X_scaled = StandardScaler().fit_transform(X)

# Create GMM model
model = GaussianMixture(
    n_components=3,
    random_state=42
)

# Train and predict clusters
model.fit(X_scaled)
labels = model.predict(X_scaled)

# Get cluster probabilities
probabilities = model.predict_proba(X_scaled)

# Display results
print("First 5 cluster predictions:")
print(labels[:5])

print("\nFirst 5 probability predictions:")
print(probabilities[:5])

# Visualize clusters
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis"
)

plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()