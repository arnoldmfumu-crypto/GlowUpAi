import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

# --- GESTION DES CHEMINS (Correctif Docker) ---
# Dans Docker, le fichier est dans /app/predict.py
# parent = /app
CURRENT_DIR = Path(__file__).resolve().parent

# On définit ROOT avec sécurité pour Docker
try:
    if os.path.exists("/app"):
        ROOT = Path("/app")
    else:
        ROOT = CURRENT_DIR.parents[1] # Remonte de oily/ vers skinmatch/
except IndexError:
    ROOT = CURRENT_DIR

# Ajout au sys.path pour que Python trouve les modules 'oily'
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# On ajuste les imports pour Docker (si oily est le dossier courant)
try:
    from transforms import get_inference_transforms
    from mobilenetv2 import build_model, CLASS_NAMES, NUM_CLASSES
except ImportError:
    from acne.transforms import get_inference_transforms
    from acne.mobilenetv2 import build_model, CLASS_NAMES, NUM_CLASSES

# Chemin vers le modèle de détection de visage
# Note : Assure-toi que ce fichier .tflite est bien copié dans ton image Docker !
BLAZEFACE_MODEL = CURRENT_DIR / "models" / "blaze_face_short_range.tflite"

def load_model(model_path: str, device: Optional[str] = None) -> tuple[torch.nn.Module, dict]:
    resolved_device: str = device if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Ajout de weights_only=True par sécurité (bonne pratique PyTorch récente)
    checkpoint: dict = torch.load(model_path, map_location=resolved_device, weights_only=False)
    metadata: dict   = checkpoint.get("metadata", {})

    num_classes: int = int(metadata.get("num_classes", NUM_CLASSES))
    # On reconstruit le modèle
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(resolved_device)
    model.eval()

    print(f"Modèle chargé depuis : {model_path}")
    return model, metadata


def preprocess_image(image_input: Union[str, Path, Image.Image], use_face_detection: bool = True) -> Image.Image:
    # 1. Conversion en format OpenCV (BGR)
    if isinstance(image_input, (str, Path)):
        image_bgr = cv2.imread(str(image_input))
    else:
        # Conversion PIL -> OpenCV BGR
        image_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

    if image_bgr is None:
        raise ValueError("Impossible de charger l'image.")

    h, w = image_bgr.shape[:2]

    # 2. Détection de visage (Fallback si le modèle .tflite manque)
    if use_face_detection and BLAZEFACE_MODEL.exists():
        try:
            base_options = python.BaseOptions(model_asset_path=str(BLAZEFACE_MODEL))
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            with mp_vision.FaceDetector.create_from_options(options) as detector:
                # Conversion pour Mediapipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                result = detector.detect(mp_image)

                if result.detections:
                    bbox = result.detections[0].bounding_box
                    margin = 0.2
                    x = max(0, int(bbox.origin_x - bbox.width * margin))
                    y = max(0, int(bbox.origin_y - bbox.height * margin))
                    w_crop = int(bbox.width * (1 + margin * 2))
                    h_crop = int(bbox.height * (1 + margin * 2))
                    
                    crop = image_bgr[y:y+h_crop, x:x+w_crop]
                    if crop.size > 0:
                        image_bgr = crop
        except Exception as e:
            print(f"Face detection failed: {e}. Using original image.")

    # 3. Resize final et retour en PIL RGB
    resized = cv2.resize(image_bgr, (640, 640))
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def predict_image(
    model: torch.nn.Module,
    image_input: Union[str, Path, Image.Image],
    device: Optional[str] = None,
    use_face_detection: bool = True,
    class_names: Optional[list[str]] = None,
) -> dict:
    resolved_device: str = device if device is not None else str(next(model.parameters()).device)
    resolved_classes: list[str] = class_names if class_names is not None else CLASS_NAMES

    # 1. Prétraitement
    pil_image = preprocess_image(image_input, use_face_detection)

    # 2. Transforms (Normalisation ImageNet)
    transform = get_inference_transforms()
    tensor = transform(pil_image).unsqueeze(0).to(resolved_device)

    # 3. Inférence
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # 4. Formatage
    predicted_idx = int(probs.argmax().item())
    confidence = float(probs[predicted_idx].item())
    
    scores = {
        cls: round(float(probs[i].item()), 4)
        for i, cls in enumerate(resolved_classes)
    }

    return {
        "class": resolved_classes[predicted_idx],
        "confidence": round(confidence, 4),
        "scores": scores,
    }

# --- Bloc main inchangé mais sécurisé ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    model, meta = load_model(args.model)
    names = meta.get("class_names", CLASS_NAMES)
    res = predict_image(model, args.image, class_names=names)
    print(res)