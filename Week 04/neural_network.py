import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(

            # Input: 4 features, Output: 16 neurons
            nn.Linear(4, 16),

            # Activation function
            nn.ReLU(),

            # Hidden layer
            nn.Linear(16, 8),
            nn.ReLU(),

            # Output layer
            nn.Linear(8, 1),

            # Binary classification probability
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# Create model
model = NeuralNetwork()

# Example input: 5 samples, each with 4 features
X = torch.rand(5, 4)

# Forward propagation
predictions = model(X)

print("Model:")
print(model)

print("\nPredictions:")
print(predictions)

print("\nOutput shape:")
print(predictions.shape)