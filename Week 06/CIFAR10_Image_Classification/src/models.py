import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def conv_block(in_channels, out_channels):

    # Convolution -> batch norm -> ReLU -> halve the image size
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


# CNN built and trained from scratch (random starting weights)
class ScratchCNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            conv_block(3, 32),      # 3x32x32  -> 32x16x16
            conv_block(32, 64),     # 32x16x16 -> 64x8x8
            conv_block(64, 128)     # 64x8x8   -> 128x4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_resnet18(num_classes=10, pretrained=True):

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Replace the 1000-class ImageNet layer with our own
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def freeze_backbone(model):

    # Freeze everything except the new final layer
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.")


def unfreeze_last_stage(model):

    # Fine-tuning: also train the last residual stage (layer4)
    for name, parameter in model.named_parameters():
        if name.startswith("layer4."):
            parameter.requires_grad = True


def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable
