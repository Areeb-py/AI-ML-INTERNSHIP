import torch
import torch.nn as nn


# ResNet idea: add the input back to the output (skip connection)
# so gradients can flow through very deep networks
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # If the shape changes, the shortcut needs a 1x1 convolution
        # so the input can still be added to the output
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection: output = F(x) + x
        out = out + identity

        return self.relu(out)


images = torch.rand(4, 64, 32, 32)

# Same shape in and out: plain identity shortcut
same_block = ResidualBlock(64, 64)
print("Input shape:          ", images.shape)
print("Same-size block:      ", same_block(images).shape)

# More channels and half the size: 1x1 convolution shortcut
down_block = ResidualBlock(64, 128, stride=2)
print("Downsampling block:   ", down_block(images).shape)

print("\nShortcut of same-size block:   ", same_block.shortcut)
print("Shortcut of downsampling block:", down_block.shortcut)
