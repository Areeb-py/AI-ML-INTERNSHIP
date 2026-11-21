# Week 06 – CNNs and Image Classification

## Overview

This week focused on convolutional neural networks (CNNs), the standard architecture for image data. The main objective was to understand how convolution and pooling extract features from images, how the classic architectures are built, and how a pretrained model can be reused for a new task.

Each idea was first implemented from scratch in NumPy or as a small PyTorch script, and then applied in an image-classification project.

## Topics Covered

- Convolution: kernels, feature maps, padding and stride.
- Max pooling and average pooling.
- Convolution and pooling layers in PyTorch.
- Classic architectures: LeNet, VGG and ResNet.
- Residual blocks and skip connections.
- Training a CNN for image classification.
- Data augmentation.
- Transfer learning: feature extraction and fine-tuning.
- Grad-CAM for explaining CNN predictions.

## Practice Files

| File | What it covers |
|---|---|
| `convolution.py` | 2D convolution written by hand in NumPy, with padding and stride, using a vertical-edge kernel. |
| `pooling.py` | Max pooling and average pooling written by hand in NumPy. |
| `conv_layers.py` | `nn.Conv2d` and `nn.MaxPool2d`: output shapes and parameter counts. |
| `lenet.py` | The LeNet-5 architecture. |
| `vgg_block.py` | VGG-style blocks of stacked 3x3 convolutions in a small VGG network. |
| `resnet_block.py` | A residual block with identity and 1x1-convolution shortcuts. |
| `train_cnn.py` | A CNN trained on the scikit-learn handwritten digits dataset (about 97% test accuracy). |
| `data_augmentation.py` | Rotation, shift, zoom, crop, blur and random erasing with `torchvision.transforms.v2`. |
| `transfer_learning.py` | A frozen pretrained ResNet-18 with a new final layer trained on digits. |
| `grad_cam.py` | Grad-CAM implemented with hooks on a small digit CNN. |

## Practical Project – CIFAR-10 Image Classification

Built an image classifier for CIFAR-10 (60,000 colour images, 32x32 pixels, 10 classes) in two ways and compared them on the 10,000 test images: a CNN trained from scratch and a fine-tuned pretrained ResNet-18. See `CIFAR10_Image_Classification/README.md` for setup and run order.

Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

### Project Workflow

1. Downloading CIFAR-10 and splitting it into 45,000 train, 5,000 validation and 10,000 test images.
2. Data augmentation (random shifts and flips) and normalisation.
3. Training a CNN from scratch on all 45,000 training images, and again on a 5,000-image subset.
4. Loading a pretrained ResNet-18 and training a new final layer on the same 5,000 images.
5. Fine-tuning the last residual stage with a smaller learning rate.
6. Comparing all models on the test set, including per-class accuracy and confusion matrices.
7. Explaining predictions with Grad-CAM.

### Model Comparison

| Model | Training images | Test accuracy | F1 (macro) |
|---|---:|---:|---:|
| CNN from scratch | 45,000 | 75.30% | 75.09% |
| CNN from scratch | 5,000 | 60.37% | 61.25% |
| ResNet-18 pretrained + fine-tuned | 5,000 | 84.12% | 84.10% |

*Measured on the 10,000 CIFAR-10 test images. With the same 5,000 training images, pretraining adds about 24 percentage points; it even beats the from-scratch CNN trained on 9 times more data.*

## Tools & Libraries

- Python
- NumPy
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib

## Key Learning Outcomes

- Understood how a kernel slides over an image to produce a feature map.
- Learned how padding, stride and pooling change the output size.
- Understood how LeNet, VGG and ResNet differ in design.
- Learned why skip connections make very deep networks trainable.
- Practiced training a CNN with a full PyTorch training loop.
- Learned how data augmentation adds variety without new data.
- Understood the difference between feature extraction and fine-tuning.
- Saw that transfer learning clearly outperforms training from scratch when labelled data is limited.
- Learned to read training curves, per-class accuracy and confusion matrices (cats and dogs are the hardest classes).
- Learned how Grad-CAM shows which image regions drive a prediction.

## Weekly Summary

Built convolutional neural networks from the ground up, from a hand-written convolution to the classic architectures, and applied them to CIFAR-10. A CNN trained from scratch reached 75.3% test accuracy, while a pretrained ResNet-18 fine-tuned on only 5,000 images reached 84.1%, and Grad-CAM was used to check what the model was looking at.
