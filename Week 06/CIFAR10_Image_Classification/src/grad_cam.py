import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import config
from data import get_test_dataset
from models import build_resnet18

# How many test images to explain (the first image of each class)
NUM_IMAGES = 8


def grad_cam(model, image):

    saved = {}

    # Capture the output of the last convolution stage (layer4)
    def save_activation(module, inputs, output):
        output.retain_grad()
        saved["activation"] = output

    hook = model.layer4.register_forward_hook(save_activation)

    # Forward pass on one image
    scores = model(image.unsqueeze(0))
    predicted_class = scores.argmax(dim=1).item()
    confidence = F.softmax(scores, dim=1)[0, predicted_class].item()

    # Backward pass from the predicted class score only
    model.zero_grad()
    scores[0, predicted_class].backward()

    hook.remove()

    activation = saved["activation"][0]           # (512, 4, 4)
    gradient = saved["activation"].grad[0]        # (512, 4, 4)

    # One weight per feature map: the average of its gradients
    weights = gradient.mean(dim=(1, 2))

    # Weighted sum of the feature maps; ReLU keeps only positive evidence
    heatmap = F.relu((weights[:, None, None] * activation).sum(dim=0))

    # Enlarge the small heatmap to the image size and scale it to 0-1
    heatmap = F.interpolate(
        heatmap[None, None],
        size=image.shape[1:],
        mode="bilinear",
        align_corners=False
    )[0, 0]

    heatmap = heatmap / (heatmap.max() + 1e-8)

    return heatmap.detach(), predicted_class, confidence


def denormalize(image):

    # Undo the ImageNet normalization so the picture can be displayed
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)

    return (image * std + mean).clamp(0, 1).permute(1, 2, 0)


def first_image_of_each_class(dataset, count):

    indices = []

    for class_index in range(count):
        indices.append(dataset.targets.index(class_index))

    return indices


def main():

    model = build_resnet18(len(config.CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(config.TRANSFER_MODEL_PATH))
    model.eval()

    test_dataset = get_test_dataset("transfer")
    sample_indices = first_image_of_each_class(test_dataset, NUM_IMAGES)

    figure, axes = plt.subplots(2, NUM_IMAGES, figsize=(2.3 * NUM_IMAGES, 5.2))

    for column, index in enumerate(sample_indices):

        image, label = test_dataset[index]
        heatmap, predicted_class, confidence = grad_cam(model, image)

        picture = denormalize(image)

        axes[0, column].imshow(picture)
        axes[0, column].set_title(config.CLASS_NAMES[label], fontsize=10)
        axes[0, column].axis("off")

        # Heatmap drawn on top of the picture
        axes[1, column].imshow(picture)
        axes[1, column].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[1, column].set_title(
            f"{config.CLASS_NAMES[predicted_class]} ({confidence:.0%})",
            fontsize=10
        )
        axes[1, column].axis("off")

        print(
            f"Image {index}: actual={config.CLASS_NAMES[label]}, "
            f"predicted={config.CLASS_NAMES[predicted_class]} "
            f"({confidence:.1%})"
        )

    figure.suptitle("Top: actual class   |   Bottom: Grad-CAM and prediction")

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "grad_cam.png", dpi=120)

    print("\nSaved:", config.OUTPUTS_DIR / "grad_cam.png")


if __name__ == "__main__":
    main()
