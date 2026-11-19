import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

import config


def get_transforms(kind, train):

    # kind = "scratch"  -> 32x32 images for the CNN built from scratch
    # kind = "transfer" -> enlarged images for the pretrained ResNet-18

    if kind == "scratch":
        mean, std = config.CIFAR_MEAN, config.CIFAR_STD
        resize = []
    else:
        mean, std = config.IMAGENET_MEAN, config.IMAGENET_STD
        resize = [v2.Resize(config.TRANSFER_IMAGE_SIZE)]

    to_tensor = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ]

    if train and kind == "scratch":
        # Data augmentation: random shift (pad then crop) and mirror
        augment = [v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip()]
    elif train:
        augment = [v2.RandomHorizontalFlip()]
    else:
        # Validation and test images are never augmented
        augment = []

    return v2.Compose([*resize, *augment, *to_tensor])


def split_indices():

    # Shuffle the 50,000 training images once, always in the same order,
    # then cut off the first 5,000 as the validation set
    generator = torch.Generator().manual_seed(config.SEED)
    order = torch.randperm(50000, generator=generator).tolist()

    val_indices = order[:config.VAL_SIZE]
    train_indices = order[config.VAL_SIZE:]

    return train_indices, val_indices


def get_loaders(kind, subset=False):

    # Two views of the same training images:
    # one with augmentation (for training), one without (for validation)
    augmented = CIFAR10(
        config.DATA_DIR, train=True,
        transform=get_transforms(kind, train=True)
    )
    plain = CIFAR10(
        config.DATA_DIR, train=True,
        transform=get_transforms(kind, train=False)
    )

    train_indices, val_indices = split_indices()

    if subset:
        train_indices = train_indices[:config.SUBSET_TRAIN_SIZE]
        val_indices = val_indices[:config.SUBSET_VAL_SIZE]

    train_loader = DataLoader(
        Subset(augmented, train_indices),
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        Subset(plain, val_indices),
        batch_size=256,
        shuffle=False
    )

    return train_loader, val_loader


def get_test_dataset(kind):

    return CIFAR10(
        config.DATA_DIR, train=False,
        transform=get_transforms(kind, train=False)
    )


def get_test_loader(kind):

    return DataLoader(get_test_dataset(kind), batch_size=256, shuffle=False)
