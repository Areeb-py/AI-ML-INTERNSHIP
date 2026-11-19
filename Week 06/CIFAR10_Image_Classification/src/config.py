from pathlib import Path

# Folders (everything is relative to the project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# CIFAR-10: 60,000 colour images, 32x32 pixels, 10 classes
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

SEED = 42
BATCH_SIZE = 64

# The 50,000 training images are split into 45,000 train + 5,000 validation.
# The 10,000 test images are only used once, in evaluate.py.
VAL_SIZE = 5000

# Small subset, used so the pretrained model trains in minutes on a CPU
SUBSET_TRAIN_SIZE = 5000
SUBSET_VAL_SIZE = 1000

# CNN trained from scratch on the full training split
SCRATCH_EPOCHS = 8
SCRATCH_LR = 0.001

# The same CNN trained from scratch on the small subset only
SCRATCH_SUBSET_EPOCHS = 15

# Pretrained ResNet-18 (trained on the small subset)
# Images are enlarged from 32x32 because the model was pretrained on big photos
TRANSFER_IMAGE_SIZE = 112

# Stage 1: train only the new final layer
HEAD_EPOCHS = 3
HEAD_LR = 0.001

# Stage 2: unfreeze the last residual stage and fine-tune gently
FINETUNE_EPOCHS = 3
FINETUNE_LR = 0.0001

# Normalization statistics
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

# The pretrained model expects the ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SCRATCH_MODEL_PATH = MODELS_DIR / "scratch_cnn_full.pth"
SCRATCH_SUBSET_MODEL_PATH = MODELS_DIR / "scratch_cnn_subset.pth"
TRANSFER_MODEL_PATH = MODELS_DIR / "resnet18_finetuned.pth"
