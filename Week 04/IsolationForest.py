import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import IsolationForest

# Generate normal data
X = np.random.normal(
    loc=0,
    scale=1,
    size=(300, 2)
)

# Add artificial outliers
outliers = np.array([
    [5, 5],
    [-5, -5],
    [6, -4],
    [-6, 5],
    [0, 7]
])

X = np.vstack([X, outliers])

# Create Isolation Forest model
model = IsolationForest(
    contamination=0.02,
    random_state=42
)

# Detect anomalies
predictions = model.fit_predict(X)

# Separate normal points and anomalies
normal = X[predictions == 1]
anomalies = X[predictions == -1]

print("Normal points:", len(normal))
print("Detected anomalies:", len(anomalies))

# Visualize results
plt.scatter(
    normal[:, 0],
    normal[:, 1],
    label="Normal"
)

plt.scatter(
    anomalies[:, 0],
    anomalies[:, 1],
    color="red",
    marker="X",
    s=100,
    label="Anomalies"
)

plt.title("Isolation Forest Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()