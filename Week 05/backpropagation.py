import numpy as np

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Reproducible random weights
np.random.seed(42)

# Initialize weights and biases
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 10000

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training
for epoch in range(epochs):

    # Forward propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    predictions = sigmoid(final_input)

    # Calculate error
    error = y - predictions

    # Backpropagation
    output_gradient = error * sigmoid_derivative(predictions)

    hidden_error = np.dot(output_gradient, W2.T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights and biases
    W2 += learning_rate * np.dot(hidden_output.T, output_gradient)
    b2 += learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

    W1 += learning_rate * np.dot(X.T, hidden_gradient)
    b1 += learning_rate * np.sum(hidden_gradient, axis=0, keepdims=True)

    if epoch % 1000 == 0:
        loss = np.mean(error ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Final predictions
print("\nFinal predictions:")
print(predictions.round(3))