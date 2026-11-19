import torch
import torch.nn as nn


# LeNet-5 (1998): the classic CNN for handwritten digits
class LeNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extractor: convolution + pooling
        self.features = nn.Sequential(

            # 1x32x32 -> 6x28x28
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),

            # 6x28x28 -> 6x14x14
            nn.AvgPool2d(kernel_size=2),

            # 6x14x14 -> 16x10x10
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),

            # 16x10x10 -> 16x5x5
            nn.AvgPool2d(kernel_size=2)
        )

        # Classifier: fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),

            nn.Linear(120, 84),
            nn.Tanh(),

            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = LeNet()

# 8 grayscale images, 32x32 pixels
images = torch.rand(8, 1, 32, 32)

outputs = model(images)

print(model)

print("\nInput shape: ", images.shape)
print("Output shape:", outputs.shape)

total_parameters = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_parameters)
