import argparse
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image

from transforms import get_inference_transforms
from mobilenetv2 import build_model, CLASS_NAMES, NUM_CLASSES


# =========================
# 🔧 CONFIG
# =========================
IMAGE_SIZE = (640, 640)


# =========================
# 🧠 MODEL
# =========================
def load_model(model_path: str, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint.get("metadata", {})

    num_classes = int(metadata.get("num_classes", NUM_CLASSES))

    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Modèle chargé : {model_path}")
    return model, metadata


# =========================
# 🖼️ PREPROCESS (FIX MAJEUR ICI)
# =========================
def letterbox(image_bgr, target_size=(640, 640)):
    h, w = image_bgr.shape[:2]

    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image_bgr, (new_w, new_h))

    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas


def preprocess_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    # 🔁 Chargement
    if isinstance(image_input, (str, Path)):
        image_bgr = cv2.imread(str(image_input))
    else:
        image_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

    if image_bgr is None:
        raise ValueError("❌ Impossible de charger l'image")

    # 🔥 FIX : letterbox (comme training)
    image_bgr = letterbox(image_bgr, IMAGE_SIZE)

    # 🔁 Retour en PIL
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


# =========================
# 🔮 PREDICTION
# =========================
def predict_image(
    model,
    image_input,
    device: Optional[str] = None,
    class_names: Optional[list[str]] = None,
):
    device = device or str(next(model.parameters()).device)
    class_names = class_names or CLASS_NAMES

    # 1. Preprocess
    pil_image = preprocess_image(image_input)

    # 2. Transform (doit contenir Normalize !)
    transform = get_inference_transforms()
    tensor = transform(pil_image).unsqueeze(0).to(device)

    # 🔍 Debug utile
    print("Tensor stats:", tensor.min().item(), tensor.max().item())

    # 3. Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # 4. Résultat
    predicted_idx = int(probs.argmax().item())
    confidence = float(probs[predicted_idx].item())

    scores = {
        class_names[i]: round(float(probs[i].item()), 4)
        for i in range(len(class_names))
    }

    return {
        "class": class_names[predicted_idx],
        "confidence": round(confidence, 4),
        "scores": scores,
    }


#--- Bloc main inchangé mais sécurisé ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model, meta = load_model(args.model)
    names = meta.get("class_names", CLASS_NAMES)

    result = predict_image(model, args.image, class_names=names)

    print("\n🎯 Résultat :")
    print(result)