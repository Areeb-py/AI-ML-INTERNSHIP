import torch
import torch.nn as nn


# Training dataset
# Features: study hours, attendance percentage
X = torch.tensor([
    [1.0, 40.0],
    [2.0, 50.0],
    [3.0, 60.0],
    [4.0, 70.0],
    [5.0, 80.0],
    [6.0, 90.0],
    [7.0, 95.0],
    [8.0, 98.0]
])

# Labels: 0 = Fail, 1 = Pass
y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0]
])


# Neural network
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


# Initialize model
model = NeuralNetwork()


# Loss function
criterion = nn.BCELoss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# Training loop
epochs = 1000

for epoch in range(epochs):

    # Forward pass
    predictions = model(X)

    # Calculate loss
    loss = criterion(predictions, y)

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )


# Evaluate model
model.eval()

with torch.no_grad():

    predictions = model(X)

    predicted_classes = (predictions >= 0.5).float()

    print("\nPredictions:")
    print(predicted_classes)

    print("\nActual labels:")
    print(y)
