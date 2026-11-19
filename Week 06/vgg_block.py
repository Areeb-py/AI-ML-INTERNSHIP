import torch
import torch.nn as nn


# VGG idea: stack several small 3x3 convolutions,
# then halve the image size with max pooling
def vgg_block(in_channels, out_channels, num_convs):

    layers = []

    for _ in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        layers.append(nn.ReLU())
        in_channels = out_channels

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)


# A small VGG-style network: channels grow while the image shrinks
class MiniVGG(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            vgg_block(3, 32, num_convs=2),     # 32x32 -> 16x16
            vgg_block(32, 64, num_convs=2),    # 16x16 -> 8x8
            vgg_block(64, 128, num_convs=3)    # 8x8   -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = MiniVGG()

images = torch.rand(4, 3, 32, 32)

# Follow the shape through each block
x = images
print("Input:        ", x.shape)

for index, block in enumerate(model.features):
    x = block(x)
    print(f"After block {index + 1}:", x.shape)

outputs = model(images)
print("Output:       ", outputs.shape)

total_parameters = sum(p.numel() for p in model.parameters())
print("\nTotal parameters:", total_parameters)
