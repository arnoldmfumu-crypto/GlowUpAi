import torch.nn as nn
from torchvision import models


CLASS_NAMES = ["dry", "normal", "oily"]
NUM_CLASSES = len(CLASS_NAMES)

def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = True) -> nn.Module:
    model = models.mobilenet_v2(weights=None) 

    # On identifie la couche linéaire d'origine
    classifier_block = model.classifier[1]
    if isinstance(classifier_block, nn.Linear):
        num_features = classifier_block.in_features
    else:
        num_features = 1280

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes),
    )
    
    return model