from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Product Recommendation API")

CSV_PATH = "data/products.csv"

SENSITIVITY_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
}

class RecommendationInput(BaseModel):
    acne_prediction: str
    acne_confidence: float
    oily_prediction: str
    oily_confidence: float
    skin_sensitivity: Literal["low", "medium", "high"]
    preferred_format: Literal["gel", "cream", "serum", "lotion"]
    budget: float

df_products = pd.read_csv(CSV_PATH)

def compatible_sensitivity(product_max: str, user_sensitivity: str) -> bool:
    # Product must be suitable for at least the user's sensitivity level
    return SENSITIVITY_ORDER[product_max] >= SENSITIVITY_ORDER[user_sensitivity]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend(payload: RecommendationInput):
    df = df_products.copy()

    df = df[df["acne_target"] == payload.acne_prediction]
    df = df[df["oily_target"] == payload.oily_prediction]
    df = df[df["format"] == payload.preferred_format]
    df = df[df["price"] <= payload.budget]
    df = df[df["sensitivity_max"].apply(lambda x: compatible_sensitivity(str(x), payload.skin_sensitivity))]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No compatible product found with the current filters."
        )

    # Basic scoring: cheapest first, then could be enriched later
    df = df.sort_values(by=["price"], ascending=True)
    best = df.iloc[0]

    reason = (
        f"Compatible with acne='{payload.acne_prediction}', oily='{payload.oily_prediction}', "
        f"format='{payload.preferred_format}', sensitivity='{payload.skin_sensitivity}', "
        f"and budget <= {payload.budget}€."
    )

    return {
        "product_name": best["product_name"],
        "brand": best["brand"],
        "format": best["format"],
        "price": float(best["price"]),
        "reason": reason
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)