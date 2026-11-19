import torch
import torch.nn as nn

# A batch of 4 RGB images, each 32x32 pixels
# Shape: (batch size, channels, height, width)
images = torch.rand(4, 3, 32, 32)

# Convolution layer: 3 input channels -> 16 feature maps
conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1
)

# Max pooling halves the height and width
pool = nn.MaxPool2d(kernel_size=2, stride=2)

after_conv = conv(images)
after_pool = pool(after_conv)

print("Input shape:     ", images.shape)
print("After Conv2d:    ", after_conv.shape)
print("After MaxPool2d: ", after_pool.shape)

# Each filter has 3 x 3 x 3 weights plus 1 bias
print("\nConv weight shape:", conv.weight.shape)
print("Conv bias shape:  ", conv.bias.shape)

total_parameters = sum(p.numel() for p in conv.parameters())
print("Conv parameters:  ", total_parameters)

# Without padding the output shrinks
conv_no_padding = nn.Conv2d(3, 16, kernel_size=5)
print("\n5x5 kernel, no padding:", conv_no_padding(images).shape)

# A stride of 2 downsamples without a pooling layer
conv_stride = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
print("3x3 kernel, stride 2:  ", conv_stride(images).shape)
