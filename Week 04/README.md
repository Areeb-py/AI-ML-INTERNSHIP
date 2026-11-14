# Week 04 – Neural Networks and PyTorch Fundamentals

## Overview
This week focused on the foundations of deep learning: how a single neuron learns, how networks of neurons are trained with backpropagation, and how to build and train neural networks using PyTorch.

The main objective was to understand what happens inside a neural network by implementing the core ideas from scratch in NumPy first, and then repeating them with PyTorch.

## Topics Covered
- The Perceptron and the step activation function.
- Activation functions: Sigmoid, ReLU and Tanh.
- Forward propagation.
- Backpropagation and gradient descent.
- PyTorch tensors.
- Automatic differentiation with Autograd.
- Building models with `nn.Module` and `nn.Sequential`.
- Loss functions (Binary Cross-Entropy).
- Optimizers: SGD, Momentum and Adam.
- Regularization: Dropout and Batch Normalization.
- The PyTorch training loop.

## Practice Files

| File | What it covers |
|---|---|
| `perceptron.py` | A single perceptron trained from scratch in NumPy to predict pass/fail from study hours and attendance. |
| `activation_functions.py` | Sigmoid, ReLU and Tanh implemented in NumPy and compared on the same inputs. |
| `backpropagation.py` | A two-layer network trained on the XOR problem using manually written forward and backward passes. |
| `tensors.py` | Creating scalars, vectors, matrices and random tensors, and inspecting shape and data type. |
| `autograd.py` | Computing a gradient automatically with `requires_grad` and `backward()`. |
| `neural_network.py` | A feed-forward binary classifier (4 → 16 → 8 → 1) defined with `nn.Module`. |
| `optimizers_demo.py` | One weight update calculated by hand for SGD, Momentum and Adam. |
| `pytorch.py` | Creating Dropout and Batch Normalization layers. |
| `regularization_demo.py` | A network that combines Batch Normalization and Dropout between its layers. |
| `train_model.py` | A complete training loop: model, BCE loss, Adam optimizer, 1000 epochs and evaluation. |

## Dataset
The Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is included in this folder for the neural network churn-prediction project.

Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Tools & Libraries
- Python
- NumPy
- PyTorch

## Key Learning Outcomes
- Understood how a perceptron updates its weights and bias from prediction errors.
- Learned why activation functions are needed and how Sigmoid, ReLU and Tanh differ.
- Understood why XOR cannot be solved by a single perceptron and needs a hidden layer.
- Implemented backpropagation manually to see how gradients flow backward through a network.
- Learned how PyTorch tensors and Autograd replace manual gradient calculations.
- Practiced defining neural networks with `nn.Module` and `nn.Sequential`.
- Understood how SGD, Momentum and Adam update weights differently.
- Learned how Dropout and Batch Normalization help reduce overfitting and stabilize training.
- Practiced the standard training loop: forward pass, loss, `zero_grad()`, `backward()` and `step()`.

## Weekly Summary
Built neural network fundamentals from the ground up, starting with a NumPy perceptron and manual backpropagation, then moving to PyTorch tensors, Autograd, model definition, regularization and a full training loop.
