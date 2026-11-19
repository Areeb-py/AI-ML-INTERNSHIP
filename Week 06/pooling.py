import numpy as np

# 4x4 feature map
feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [7, 2, 9, 1],
    [3, 4, 5, 8]
])


def pool2d(feature_map, size=2, stride=2, mode="max"):

    height, width = feature_map.shape

    output_height = (height - size) // stride + 1
    output_width = (width - size) // stride + 1

    output = np.zeros((output_height, output_width))

    for row in range(output_height):
        for col in range(output_width):

            top = row * stride
            left = col * stride
            window = feature_map[top:top + size, left:left + size]

            # Max pooling keeps the strongest value,
            # average pooling keeps the mean of the window
            if mode == "max":
                output[row, col] = np.max(window)
            else:
                output[row, col] = np.mean(window)

    return output


print("Feature map:")
print(feature_map)

print("\nMax pooling (2x2):")
print(pool2d(feature_map, mode="max"))

print("\nAverage pooling (2x2):")
print(pool2d(feature_map, mode="average"))
