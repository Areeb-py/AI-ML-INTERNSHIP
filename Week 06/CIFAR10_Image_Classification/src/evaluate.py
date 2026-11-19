import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

import config
from data import get_test_loader
from models import ScratchCNN, build_resnet18


def predict(model, loader):

    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            predictions.extend(model(images).argmax(dim=1).tolist())
            labels.extend(batch_labels.tolist())

    return labels, predictions


def evaluate_model(name, model, kind):

    # The 10,000 test images were never seen during training
    labels, predictions = predict(model, get_test_loader(kind))

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )

    matrix = confusion_matrix(labels, predictions)

    # Accuracy for each class: correct predictions / images of that class
    per_class = matrix.diagonal() / matrix.sum(axis=1)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "per_class_accuracy": {
            class_name: float(value)
            for class_name, value in zip(config.CLASS_NAMES, per_class)
        }
    }

    # Confusion matrix image
    figure, axis = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay(matrix, display_labels=config.CLASS_NAMES)
    display.plot(cmap="Blues", colorbar=False, ax=axis, xticks_rotation=45)

    axis.set_title(name)
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / f"confusion_matrix_{name}.png", dpi=120)
    plt.close(figure)

    return metrics


def main():

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    num_classes = len(config.CLASS_NAMES)

    scratch_full = ScratchCNN(num_classes)
    scratch_full.load_state_dict(torch.load(config.SCRATCH_MODEL_PATH))

    scratch_subset = ScratchCNN(num_classes)
    scratch_subset.load_state_dict(
        torch.load(config.SCRATCH_SUBSET_MODEL_PATH)
    )

    resnet = build_resnet18(num_classes, pretrained=False)
    resnet.load_state_dict(torch.load(config.TRANSFER_MODEL_PATH))

    results = {
        "scratch_cnn_full": evaluate_model(
            "scratch_cnn_full", scratch_full, "scratch"
        ),
        "scratch_cnn_subset": evaluate_model(
            "scratch_cnn_subset", scratch_subset, "scratch"
        ),
        "resnet18_finetuned": evaluate_model(
            "resnet18_finetuned", resnet, "transfer"
        )
    }

    with open(config.OUTPUTS_DIR / "metrics.json", "w") as file:
        json.dump(results, file, indent=2)

    print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")

    for name, metrics in results.items():
        print(
            f"{name:<22}"
            f"{metrics['accuracy']:>10.4f}"
            f"{metrics['precision_macro']:>11.4f}"
            f"{metrics['recall_macro']:>9.4f}"
            f"{metrics['f1_macro']:>8.4f}"
        )

    print("\nPer-class accuracy:")
    print(f"{'Class':<12}" + "".join(f"{name[:18]:>20}" for name in results))

    for class_name in config.CLASS_NAMES:
        row = "".join(
            f"{metrics['per_class_accuracy'][class_name]:>20.4f}"
            for metrics in results.values()
        )
        print(f"{class_name:<12}{row}")

    print("\nSaved:", config.OUTPUTS_DIR / "metrics.json")


if __name__ == "__main__":
    main()
