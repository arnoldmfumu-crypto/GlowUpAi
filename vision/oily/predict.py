import argparse
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]  # skinmatch/
OILY_ROOT = Path(__file__).resolve().parent  # skinmatch/vision/oily/

if str(OILY_ROOT) not in sys.path:
    sys.path.insert(0, str(OILY_ROOT))

from transforms import get_inference_transforms        # type: ignore[import] # 01_transforms.py  # noqa: E402
from mobilenetv2 import build_model, CLASS_NAMES, NUM_CLASSES  # type: ignore[import] # 02_mobilenetv2.py  # noqa: E402

BLAZEFACE_MODEL = ROOT / "blaze_face_short_range.tflite"


def load_model(model_path: str, device: Optional[str] = None) -> tuple[torch.nn.Module, dict]:
    """
    Charge le checkpoint et reconstruit le modèle.

    Args:
        model_path : Chemin vers le fichier .pt
                     Ex: vision/models/mobilenetv2_baseline_oily_dry_normal.pt
        device     : "cpu", "cuda" ou None (auto-détection)

    Returns:
        (model, metadata)
    """
    resolved_device: str = device if device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint: dict = torch.load(model_path, map_location=resolved_device, weights_only=False)
    metadata: dict   = checkpoint.get("metadata", {})

    num_classes: int = int(metadata.get("num_classes", NUM_CLASSES))
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(resolved_device)
    model.eval()

    print(f"Modèle chargé depuis : {model_path}")
    print(f"Architecture : {metadata.get('architecture', 'MobileNetV2')}")
    print(f"Val Accuracy : {metadata.get('best_val_acc', 'N/A')}")

    return model, metadata


def preprocess_image(image_path: Union[str, Path], use_face_detection: bool = True) -> Image.Image:
    """
    Applique le pipeline de filtrage sur une image brute :
      1. Détection de visage avec MediaPipe BlazeFace
      2. Crop avec 20% de marge autour du visage
      3. Resize à 640x640
      4. Retourne une image PIL RGB

    Si use_face_detection=False ou si BlazeFace n'est pas disponible,
    on fait juste un resize à 640x640 (fallback).
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    h, w = image_bgr.shape[:2]

    if use_face_detection and BLAZEFACE_MODEL.exists():
        try:
            base_options = python.BaseOptions(model_asset_path=str(BLAZEFACE_MODEL))
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=0.5,
            )
            detector = mp_vision.FaceDetector.create_from_options(options)
            mp_image = mp.Image.create_from_file(str(image_path))
            result   = detector.detect(mp_image)

            if result.detections:
                bbox   = result.detections[0].bounding_box
                margin = 0.2
                x      = max(0, int(bbox.origin_x - bbox.width * margin))
                y      = max(0, int(bbox.origin_y - bbox.height * margin))
                x_end  = min(w, int(x + bbox.width  * (1 + margin * 2)))
                y_end  = min(h, int(y + bbox.height * (1 + margin * 2)))
                crop   = image_bgr[y:y_end, x:x_end]

                if crop.size > 0:
                    image_bgr = crop
        except Exception as e:
            print(f"Détection de visage échouée ({e}), fallback sur image entière.")

    resized = cv2.resize(image_bgr, (640, 640))
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def predict_image(
    model: torch.nn.Module,
    image_input: Union[str, Path, Image.Image],
    device: Optional[str] = None,
    use_face_detection: bool = True,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    Prédit la classe d'une image (oily / dry / normal).

    Args:
        model             : Modèle PyTorch chargé via load_model()
        image_input       : Chemin (str/Path) ou image PIL déjà chargée
        device            : "cpu" / "cuda" / None (auto depuis le modèle)
        use_face_detection: Appliquer le pipeline MediaPipe
        class_names       : Liste des classes (défaut : CLASS_NAMES)

    Returns:
        {
          "class":      "oily",
          "confidence": 0.87,
          "scores":     {"dry": 0.05, "normal": 0.08, "oily": 0.87}
        }
    """
    # Résolution du device : priorité à l'argument, sinon on lit depuis le modèle
    resolved_device: str = device if device is not None else str(next(model.parameters()).device)
    resolved_classes: list[str] = class_names if class_names is not None else CLASS_NAMES

    # 1. Chargement + filtrage
    if isinstance(image_input, (str, Path)):
        pil_image = preprocess_image(image_input, use_face_detection)
    elif isinstance(image_input, Image.Image):
        pil_image = image_input
    else:
        raise TypeError(f"image_input doit être un chemin ou une PIL Image, reçu : {type(image_input)}")

    # 2. Transforms (CLAHE + resize + normalisation ImageNet)
    transform = get_inference_transforms()
    tensor    = transform(pil_image).unsqueeze(0).to(resolved_device)  # [1, 3, 224, 224]

    # 3. Inférence
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)  # [num_classes]

    # 4. Résultat — cast explicite en types Python natifs pour éviter les erreurs Pylance
    predicted_idx: int = int(probs.argmax().item())
    confidence: float  = float(probs[predicted_idx].item())
    scores: dict[str, float] = {
        cls: round(float(probs[i].item()), 4)
        for i, cls in enumerate(resolved_classes)
    }

    return {
        "class":      resolved_classes[predicted_idx],
        "confidence": round(confidence, 4),
        "scores":     scores,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inférence oily/dry/normal sur une image")
    parser.add_argument("--model",  type=str, required=True,
                        help="Chemin vers le fichier .pt (ex: vision/models/baseline.pt)")
    parser.add_argument("--image",  type=str, required=True,
                        help="Chemin vers l'image à analyser")
    parser.add_argument("--device", type=str, default=None,
                        help="'cpu' ou 'cuda' (défaut : auto-détection)")
    parser.add_argument("--no-face-detection", action="store_true",
                        help="Désactiver la détection de visage MediaPipe")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model, meta = load_model(args.model, device=args.device)
    names: list[str] = meta.get("class_names", CLASS_NAMES)

    result = predict_image(
        model,
        args.image,
        device=args.device,
        use_face_detection=not args.no_face_detection,
        class_names=names,
    )

    print(f"Prédiction: {result['class'].upper()}")
    print(f"Confiance: {result['confidence']*100:.1f}%")
    print("Scores:")
    for cls, score in result["scores"].items():
        bar = "█" * int(score * 20)
        print(f"    {cls:<8} {score*100:5.1f}%  {bar}")

# Exemple en CLI:

# python vision/oily/predict.py \
#   --model vision/models/mobilenetv2_baseline_oily_dry_normal.pt \
#   --image vision/oily/image_test_oily.jpeg \
#   --device cpu

# python vision/oily/predict.py \
#   --model vision/models/mobilenetv2_baseline_oily_dry_normal.pt \
#   --image vision/oily/image_test_dry.jpeg \
#   --device cpu