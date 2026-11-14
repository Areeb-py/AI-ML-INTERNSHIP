
import numpy as np

# Input features: [hours studied, attendance]
X = np.array([
    [2, 50],
    [3, 60],
    [5, 80],
    [6, 90]
])

# Expected outputs: 0 = Fail, 1 = Pass
y = np.array([0, 0, 1, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0

learning_rate = 0.1
epochs = 10


# Step activation function
def activation(z):
    return 1 if z >= 0 else 0


# Training loop
for epoch in range(epochs):

    for i in range(len(X)):

        # Forward propagation
        z = np.dot(X[i], weights) + bias
        prediction = activation(z)

        # Calculate prediction error
        error = y[i] - prediction

        # Update weights and bias
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

    print(f"Epoch {epoch + 1}")
    print("Weights:", weights)
    print("Bias:", bias)


# Test predictions
print("\nFinal predictions:")

for sample in X:
    z = np.dot(sample, weights) + bias
    prediction = activation(z)
    print(sample, "=>", prediction)
