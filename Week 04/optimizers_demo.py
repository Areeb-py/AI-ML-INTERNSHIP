import numpy as np

# Example gradient and initial weight
gradient = 0.8
weight = 5.0
learning_rate = 0.1

# SGD
sgd_weight = weight - learning_rate * gradient

# Momentum
momentum = 0.9
previous_velocity = 0.0

velocity = momentum * previous_velocity + gradient
momentum_weight = weight - learning_rate * velocity

# Adam (first update, simplified)
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

m = (1 - beta1) * gradient
v = (1 - beta2) * gradient**2

m_hat = m / (1 - beta1)
v_hat = v / (1 - beta2)

adam_weight = weight - learning_rate * (
    m_hat / (np.sqrt(v_hat) + epsilon)
)

print("Original weight:", weight)
print("SGD weight:", sgd_weight)
print("Momentum weight:", momentum_weight)
print("Adam weight:", adam_weight)