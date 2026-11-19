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

## Practical Project – Ants vs Bees Image Classification

Built an image classifier on the Hymenoptera (ants vs bees) dataset in two ways and compared them. See `Ants_Bees_Classification/README.md` for setup and run order.

Dataset: https://download.pytorch.org/tutorial/hymenoptera_data.zip

### Project Workflow

1. Downloading and organising the dataset.
2. Data augmentation and preprocessing.
3. Building and training a CNN from scratch.
4. Loading a pretrained ResNet-18 and training a new final layer.
5. Fine-tuning the last residual stage with a smaller learning rate.
6. Comparing both models on the validation set.
7. Explaining predictions with Grad-CAM.

### Model Comparison

| Model | Accuracy | F1 (macro) |
|---|---:|---:|
| CNN from scratch | 72.55% | 70.43% |
| ResNet-18 fine-tuned | 94.77% | 94.72% |

*These results are from the completed Ants vs Bees project, measured on 153 validation photos.*

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
- Saw that transfer learning clearly outperforms training from scratch on a small dataset.
- Learned how Grad-CAM shows which image regions drive a prediction.

## Weekly Summary

Built convolutional neural networks from the ground up, from a hand-written convolution to the classic architectures, and applied them to a real image dataset. A fine-tuned pretrained ResNet-18 reached about 95% accuracy on ants vs bees where a CNN trained from scratch reached about 73%, and Grad-CAM was used to check what the model was looking at.
