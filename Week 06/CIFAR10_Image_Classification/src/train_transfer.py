import torch
import torch.nn as nn

import config
from data import get_loaders
from engine import fit, new_history, save_history, set_seed
from models import (
    build_resnet18,
    count_parameters,
    freeze_backbone,
    unfreeze_last_stage
)


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def main():

    set_seed()

    # The pretrained model is trained on the 5,000-image subset
    train_loader, val_loader = get_loaders("transfer", subset=True)

    print("Training images:  ", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    # ResNet-18 pretrained on ImageNet, with a new 10-class final layer
    model = build_resnet18(num_classes=len(config.CLASS_NAMES))

    criterion = nn.CrossEntropyLoss()
    history = new_history()

    # -----------------------------------------------------------
    # Stage 1 - feature extraction: train only the new final layer
    # -----------------------------------------------------------
    freeze_backbone(model)

    total, trainable = count_parameters(model)
    print(f"\nStage 1 parameters: {total} total, {trainable} trainable")

    optimizer = torch.optim.Adam(
        trainable_parameters(model), lr=config.HEAD_LR
    )

    fit(
        model, train_loader, val_loader,
        criterion, optimizer,
        epochs=config.HEAD_EPOCHS,
        history=history,
        stage="[head] "
    )

    # -----------------------------------------------------------
    # Stage 2 - fine-tuning: unfreeze layer4, use a smaller learning rate
    # -----------------------------------------------------------
    unfreeze_last_stage(model)

    total, trainable = count_parameters(model)
    print(f"\nStage 2 parameters: {total} total, {trainable} trainable")

    optimizer = torch.optim.Adam(
        trainable_parameters(model), lr=config.FINETUNE_LR
    )

    fit(
        model, train_loader, val_loader,
        criterion, optimizer,
        epochs=config.FINETUNE_EPOCHS,
        history=history,
        stage="[fine-tune] "
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.TRANSFER_MODEL_PATH)

    save_history(history, "resnet18_finetuned")

    print("\nSaved model:", config.TRANSFER_MODEL_PATH)


if __name__ == "__main__":
    main()
