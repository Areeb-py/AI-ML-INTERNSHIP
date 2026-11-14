import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Generate dataset with 10 features
X, y = make_classification(
    n_samples=300,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    random_state=42
)

print("Original shape:", X.shape)

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("Reduced shape:", X_pca.shape)

# Explained variance
print("Explained variance:", pca.explained_variance_ratio_)

# Visualize
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")
plt.show()