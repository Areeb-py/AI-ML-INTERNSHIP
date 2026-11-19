import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from torchvision.transforms import v2

torch.manual_seed(42)

# Take one handwritten digit (8x8) and enlarge it to 64x64
digits = load_digits()
image = torch.tensor(digits.images[7] / 16.0, dtype=torch.float32)

# Shape (1, 1, 8, 8) -> (1, 64, 64): one channel, height, width
image = F.interpolate(image[None, None], size=64, mode="bilinear")[0]

print("Image shape:", image.shape)

# Each transform creates a slightly different version of the same image.
# The label stays the same, so the model sees more variety for free.
augmentations = {
    "Original": v2.Identity(),
    "Rotation": v2.RandomRotation(degrees=20),
    "Shift": v2.RandomAffine(degrees=0, translate=(0.15, 0.15)),
    "Zoom": v2.RandomAffine(degrees=0, scale=(0.7, 1.3)),
    "Crop": v2.RandomResizedCrop(size=64, scale=(0.6, 1.0)),
    "Blur": v2.GaussianBlur(kernel_size=7, sigma=2.0),
    "Erasing": v2.RandomErasing(p=1.0, scale=(0.05, 0.15))
}

# A typical training pipeline chains several random transforms together
train_transform = v2.Compose([
    v2.RandomRotation(degrees=15),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.RandomErasing(p=0.5, scale=(0.02, 0.1))
])

augmentations["Combined"] = train_transform

# Plot every augmentation side by side
figure, axes = plt.subplots(2, 4, figsize=(10, 5))

for axis, (name, transform) in zip(axes.flat, augmentations.items()):
    augmented = transform(image)
    axis.imshow(augmented[0], cmap="gray")
    axis.set_title(name)
    axis.axis("off")

plt.tight_layout()
plt.savefig("augmentation_examples.png", dpi=120)

print("Saved: augmentation_examples.png")

# The same pipeline gives a different result every time it is called
first = train_transform(image)
second = train_transform(image)

print("Two augmented copies identical:", torch.equal(first, second))
