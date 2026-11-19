import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

torch.manual_seed(42)

# ---------------------------------------------------------------
# 1. Load a ResNet-18 that was already trained on ImageNet
#    (the weights are downloaded the first time this runs)
# ---------------------------------------------------------------
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# 2. Freeze every pretrained layer so its weights are not updated
for parameter in model.parameters():
    parameter.requires_grad = False

# 3. Replace the final layer: 1000 ImageNet classes -> 10 digits
#    A new layer is trainable by default
model.fc = nn.Linear(model.fc.in_features, 10)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total parameters:    ", total)
print("Trainable parameters:", trainable)


# ---------------------------------------------------------------
# Prepare the data the way the pretrained model expects it
# ---------------------------------------------------------------
digits = load_digits()

images = torch.tensor(digits.images / 16.0, dtype=torch.float32)
labels = torch.tensor(digits.target, dtype=torch.long)

# 8x8 grayscale -> 64x64, then repeat the channel to get 3 "RGB" channels
images = F.interpolate(images[:, None], size=64, mode="bilinear")
images = images.repeat(1, 3, 1, 1)

# Normalize with the ImageNet mean and standard deviation
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
images = (images - mean) / std

X_train, X_test, y_train, y_test = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)


def evaluate(model, loader):

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_images, batch_labels in loader:
            predictions = model(batch_images).argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)

    return correct / total


# ---------------------------------------------------------------
# Train only the new final layer (feature extraction)
# ---------------------------------------------------------------
criterion = nn.CrossEntropyLoss()

# Only the parameters of the new layer are given to the optimizer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)

# eval() keeps the frozen BatchNorm layers using their pretrained statistics
model.eval()

print("\nTest accuracy before training:", round(evaluate(model, test_loader), 4))

epochs = 5

for epoch in range(epochs):

    total_loss = 0.0

    for batch_images, batch_labels in train_loader:

        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {total_loss / len(train_loader):.4f}, "
        f"Test accuracy: {evaluate(model, test_loader):.4f}"
    )

# Fine-tuning would be the next step: unfreeze some pretrained layers
# and continue training with a much smaller learning rate, for example:
#
# for parameter in model.layer4.parameters():
#     parameter.requires_grad = True
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
