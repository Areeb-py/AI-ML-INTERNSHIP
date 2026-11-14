import numpy as np

values = np.array([-3, -2, -1, 0, 1, 2, 3])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

print("Input:  ", values)
print("Sigmoid:", sigmoid(values))
print("ReLU:   ", relu(values))
print("Tanh:   ", tanh(values))