import torch

# Scalar
scalar = torch.tensor(10)

# Vector
vector = torch.tensor([1, 2, 3])

# Matrix
matrix = torch.tensor([
    [1, 2],
    [3, 4]
])

# Random tensor
random_tensor = torch.rand(2, 3)

print("Scalar:\n", scalar)
print("Vector:\n", vector)
print("Matrix:\n", matrix)
print("Random Tensor:\n", random_tensor)

print("Matrix shape:", matrix.shape)
print("Tensor data type:", matrix.dtype)