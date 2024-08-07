import torch
import torchvision.transforms as transforms
from pathlib import Path
from adfd_model.util.augmentations import AgeTransformer

EXPERIMENT_TYPE = 'ffhq_aging'
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / '../../pretrained_models/sam_ffhq_aging.pt'
PREDICTOR_PATH = BASE_DIR / \
    '../../pretrained_models/shape_predictor_68_face_landmarks.dat'

EXPERIMENT_DATA_ARGS = {
    "ffhq_aging": {
        "model_path": MODEL_PATH,
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
