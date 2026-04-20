import torch.nn as nn
from torchvision import models


CLASS_NAMES = ["acne", "normal"]
NUM_CLASSES = len(CLASS_NAMES)


def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> nn.Module:
    """
    Charge MobileNetV2 pré-entraîné sur ImageNet et adapte la tête de classification.

    Args:
        num_classes:      Nombre de classes en sortie (3 par défaut).
        freeze_backbone:  Si True, gèle toutes les couches sauf le classifier.
                          Mettre False pour le fine-tuning ou le chargement d'un checkpoint.

    Returns:
        Modèle PyTorch prêt à être envoyé sur device.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT
    model   = models.mobilenet_v2(weights=weights)

    in_features: int = int(model.classifier[1].in_features)  # type: ignore[arg-type]

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Remplacement de la tête avec Dropout pour réduire l'overfitting
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )

    return model