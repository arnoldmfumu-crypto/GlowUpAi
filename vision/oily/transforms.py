import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


# ── Prétraitement CLAHE ────────────────────────────────────────────────────────

def apply_clahe_lab(img: Image.Image) -> Image.Image:
    """
    Égalisation d'histogramme adaptative (CLAHE) appliquée sur le canal
    Luminance en espace LAB.
    Améliore le contraste localement — utile pour uniformiser des photos
    prises sous différentes conditions lumineuses.
    """
    img_array = np.array(img)
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


# ── Normalisation ImageNet (standard pour les backbones pré-entraînés) ─────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Pipelines ─────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Pipeline entraînement : CLAHE + augmentation + normalisation."""
    return transforms.Compose([
        transforms.Lambda(apply_clahe_lab),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15), #pas encore testé
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), #pas encore testé
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Pipeline validation/test : CLAHE + normalisation, sans augmentation."""
    return transforms.Compose([
        transforms.Lambda(apply_clahe_lab),
        transforms.Resize(224),
        transforms.CenterCrop(224),
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