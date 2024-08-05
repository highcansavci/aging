import torch
import torchvision.transforms as transforms
from sam_model.util.augmentations import AgeTransformer

EXPERIMENT_TYPE = 'ffhq_aging'

EXPERIMENT_DATA_ARGS = {
    "ffhq_aging": {
        "model_path": "../pretrained_models/sam_ffhq_aging.pt",
        "image_path": "notebooks/images/866.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]
IMAGE_SIZE = 256
TARGET_AGES = [10, 30, 50, 70]
AGE_TRANSFORMERS = [AgeTransformer(target_age=age) for age in TARGET_AGES]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
