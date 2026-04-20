import sys
from io import BytesIO
from pathlib import Path
from typing import Dict

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Garantit que les modules locaux (predict, transforms, mobilenetv2) sont trouvables
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import load_model, predict_image  # noqa: E402
from mobilenetv2 import CLASS_NAMES

app = FastAPI(title="Acne Prediction API")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "mobilenetv2_baseline_acne_normal.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, _metadata = load_model(str(MODEL_PATH), device=DEVICE)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
        result = predict_image(model, pil_image, device=DEVICE, use_face_detection=True, class_names=CLASS_NAMES)
        return {
            "model": "acne_classifier",
            "prediction": result["class"],
            "confidence": result["confidence"],
            "scores": result["scores"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)