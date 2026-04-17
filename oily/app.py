from io import BytesIO
from typing import Dict

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
import uvicorn

app = FastAPI(title="Oily Prediction API")

MODEL_PATH = "models/oily_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# Replace this with your actual model class if needed
# -------------------------------------------------------------------
class DummyOilyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(224 * 224 * 3, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

def load_model():
    model = DummyOilyModel()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict):
        try:
            model.load_state_dict(state)
        except Exception:
            model = state
    else:
        model = state
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

CLASS_NAMES = ["not_oily", "oily"]

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_idx].item())

        return {
            "model": "oily",
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)