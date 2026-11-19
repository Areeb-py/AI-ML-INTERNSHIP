import numpy as np

# A small 6x6 grayscale "image": dark on the left, bright on the right
image = np.array([
    [0, 0, 0, 10, 10, 10],
    [0, 0, 0, 10, 10, 10],
    [0, 0, 0, 10, 10, 10],
    [0, 0, 0, 10, 10, 10],
    [0, 0, 0, 10, 10, 10],
    [0, 0, 0, 10, 10, 10]
])

# 3x3 kernel (filter) that detects vertical edges
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])


def convolve2d(image, kernel, stride=1, padding=0):

    # Add zeros around the image border
    if padding > 0:
        image = np.pad(image, padding)

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Output size formula: (input - kernel) / stride + 1
    output_height = (image_height - kernel_height) // stride + 1
    output_width = (image_width - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width))

    # Slide the kernel over the image
    for row in range(output_height):
        for col in range(output_width):

            # Region of the image currently under the kernel
            top = row * stride
            left = col * stride
            region = image[top:top + kernel_height, left:left + kernel_width]

            # Multiply element-wise and add everything up
            output[row, col] = np.sum(region * kernel)

    return output


feature_map = convolve2d(image, kernel)

print("Image shape:", image.shape)
print("Kernel shape:", kernel.shape)

print("\nFeature map (no padding):")
print(feature_map)
print("Shape:", feature_map.shape)

# Padding of 1 keeps the output the same size as the input
same_size = convolve2d(image, kernel, padding=1)
print("\nShape with padding=1:", same_size.shape)

# Stride of 2 moves the kernel two pixels at a time
strided = convolve2d(image, kernel, stride=2)
print("Shape with stride=2:", strided.shape)
