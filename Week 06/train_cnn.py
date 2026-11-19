import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# Handwritten digits dataset: 1797 grayscale images, 8x8 pixels, 10 classes
digits = load_digits()

# Scale pixel values from 0-16 to 0-1
images = digits.images / 16.0
labels = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Conv2d expects (batch, channels, height, width), so add a channel dimension
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)


# Convolutional neural network
class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 1x8x8 -> 16x8x8 -> 16x4x4
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16x4x4 -> 32x4x4 -> 32x2x2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 64),
            nn.ReLU(),

            # 10 outputs, one score (logit) per digit
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = DigitCNN()

# CrossEntropyLoss applies softmax internally, so the model outputs raw logits
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def accuracy(model, X, y):

    model.eval()

    with torch.no_grad():
        predictions = model(X).argmax(dim=1)

    return (predictions == y).float().mean().item()


# Training loop
epochs = 20

for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    for batch_images, batch_labels in train_loader:

        # Forward pass
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)

        # Backpropagation and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Loss: {total_loss / len(train_loader):.4f}, "
            f"Test accuracy: {accuracy(model, X_test, y_test):.4f}"
        )


print("\nFinal train accuracy:", round(accuracy(model, X_train, y_train), 4))
print("Final test accuracy: ", round(accuracy(model, X_test, y_test), 4))

# Show a few predictions next to the true labels
model.eval()

with torch.no_grad():
    sample_predictions = model(X_test[:10]).argmax(dim=1)

print("\nPredicted:", sample_predictions.tolist())
print("Actual:   ", y_test[:10].tolist())
