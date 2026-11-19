import torch
import torch.nn as nn

import config
from data import get_loaders
from engine import fit, new_history, save_history, set_seed
from models import ScratchCNN, count_parameters


def train(name, subset, epochs, model_path):

    set_seed()

    train_loader, val_loader = get_loaders("scratch", subset=subset)

    print(f"\n===== {name} =====")
    print("Training images:  ", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    model = ScratchCNN(num_classes=len(config.CLASS_NAMES))

    total, trainable = count_parameters(model)
    print(f"Parameters: {total} total, {trainable} trainable\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.SCRATCH_LR)

    history = fit(
        model, train_loader, val_loader,
        criterion, optimizer,
        epochs=epochs,
        history=new_history()
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    save_history(history, name)

    print("Saved model:", model_path)


def main():

    # 1. The CNN trained on all 45,000 training images
    train(
        "scratch_cnn_full",
        subset=False,
        epochs=config.SCRATCH_EPOCHS,
        model_path=config.SCRATCH_MODEL_PATH
    )

    # 2. The same CNN trained on only 5,000 images - the same images
    #    the pretrained ResNet-18 gets, so the two can be compared fairly
    train(
        "scratch_cnn_subset",
        subset=True,
        epochs=config.SCRATCH_SUBSET_EPOCHS,
        model_path=config.SCRATCH_SUBSET_MODEL_PATH
    )


if __name__ == "__main__":
    main()
