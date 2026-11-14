import torch

# Create a tensor that tracks gradients
x = torch.tensor(2.0, requires_grad=True)

# Mathematical operation
y = x ** 2 + 3 * x + 1

# Calculate gradients
y.backward()

print("Output:", y.item())
print("Gradient:", x.grad.item())