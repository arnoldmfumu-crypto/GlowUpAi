import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


# Normalisation

IMAGENET_MEAN = [0.4996, 0.4011, 0.3593]
IMAGENET_STD  = [0.3475, 0.3008, 0.2851]


# ── Pipelines ─────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Pipeline entraînement : augmentation + normalisation."""
    return transforms.Compose([
        transforms.Resize(640), 
        transforms.CenterCrop(640),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Pipeline validation/test : normalisation, sans augmentation."""
    return transforms.Compose([
        transforms.Resize((640,640)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Pipeline inférence (identique à val).
    Exposé séparément pour être importé proprement dans 04_predict.py et app.py
    sans dépendre du reste du module de training.
    """
    return get_val_transforms()