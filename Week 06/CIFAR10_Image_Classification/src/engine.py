import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import config


def set_seed(seed=config.SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation and weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, criterion, optimizer, epochs,
        history, stage=""):

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"{stage}Epoch [{epoch + 1}/{epochs}] "
            f"train loss: {train_loss:.4f}, train acc: {train_accuracy:.4f}, "
            f"val loss: {val_loss:.4f}, val acc: {val_accuracy:.4f}"
        )

    return history


def new_history():

    return {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }


def save_history(history, name):

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.OUTPUTS_DIR / f"history_{name}.json", "w") as file:
        json.dump(history, file, indent=2)

    # Loss and accuracy curves
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))

    loss_axis.plot(epochs, history["train_loss"], label="Train")
    loss_axis.plot(epochs, history["val_loss"], label="Validation")
    loss_axis.set_title("Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.legend()

    accuracy_axis.plot(epochs, history["train_accuracy"], label="Train")
    accuracy_axis.plot(epochs, history["val_accuracy"], label="Validation")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.legend()

    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / f"training_curves_{name}.png", dpi=120)
    plt.close(figure)
