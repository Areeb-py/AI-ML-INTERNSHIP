from collections import Counter

from torchvision.datasets import CIFAR10

import config


def main():

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Downloads about 170 MB the first time, then reuses the local copy
    train_data = CIFAR10(config.DATA_DIR, train=True, download=True)
    test_data = CIFAR10(config.DATA_DIR, train=False, download=True)

    print("Training images:", len(train_data))
    print("Test images:    ", len(test_data))

    image, label = train_data[0]
    print("Image size:", image.size, "| mode:", image.mode)

    # Every class has the same number of images
    counts = Counter(train_data.targets)

    print("\nTraining images per class:")
    for index, name in enumerate(config.CLASS_NAMES):
        print(f"  {name:<11}{counts[index]}")


if __name__ == "__main__":
    main()
