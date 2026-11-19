import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

# Grad-CAM shows WHERE in the image a CNN looked to make its decision.
# Steps:
#   1. Take the feature maps of the last convolution layer.
#   2. Take the gradient of the predicted class score w.r.t. those maps.
#   3. Average each gradient map -> one importance weight per feature map.
#   4. Weighted sum of the feature maps, then ReLU -> heatmap.

digits = load_digits()

images = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)
labels = torch.tensor(digits.target, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    images, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Global average pooling: one value per feature map
        x = x.mean(dim=(2, 3))

        return self.fc(x)


model = DigitCNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Quick full-batch training so the model has something to explain
for epoch in range(300):

    loss = criterion(model(X_train), y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

with torch.no_grad():
    test_accuracy = (model(X_test).argmax(dim=1) == y_test).float().mean()

print("Test accuracy:", round(test_accuracy.item(), 4))


# Hooks capture what flows through the last convolution layer
saved = {}


def save_activation(module, inputs, output):
    saved["activation"] = output


def save_gradient(module, grad_input, grad_output):
    saved["gradient"] = grad_output[0]


model.conv2.register_forward_hook(save_activation)
model.conv2.register_full_backward_hook(save_gradient)


def grad_cam(model, image):

    # Forward pass on a single image
    scores = model(image.unsqueeze(0))
    predicted_class = scores.argmax(dim=1).item()

    # Backward pass from the predicted class score only
    model.zero_grad()
    scores[0, predicted_class].backward()

    activation = saved["activation"][0]    # (32, 8, 8)
    gradient = saved["gradient"][0]        # (32, 8, 8)

    # One weight per feature map: the average of its gradients
    weights = gradient.mean(dim=(1, 2))

    # Weighted sum of feature maps; ReLU keeps only positive evidence
    heatmap = F.relu((weights[:, None, None] * activation).sum(dim=0))

    # Scale to 0-1 for display
    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap.detach(), predicted_class


# Explain the first six test images
figure, axes = plt.subplots(2, 6, figsize=(12, 4.5))

for index in range(6):

    image = X_test[index]
    heatmap, predicted_class = grad_cam(model, image)

    axes[0, index].imshow(image[0], cmap="gray")
    axes[0, index].set_title(f"Actual: {y_test[index].item()}")
    axes[0, index].axis("off")

    # Heatmap drawn on top of the digit
    axes[1, index].imshow(image[0], cmap="gray")
    axes[1, index].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[1, index].set_title(f"Predicted: {predicted_class}")
    axes[1, index].axis("off")

    print(
        f"Image {index}: actual={y_test[index].item()}, "
        f"predicted={predicted_class}, "
        f"heatmap shape={tuple(heatmap.shape)}"
    )

plt.tight_layout()
plt.savefig("grad_cam_examples.png", dpi=120)

print("Saved: grad_cam_examples.png")
