from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ingest import CHROMA_DIR, build_index
from retriever import recommend

VALID_FORMULATIONS = {"vegan", "clean", "bio"}
VALID_ORIGINS = {"french"}
VALID_PRODUCT_TYPES = {
    "moisturizer", "serum", "cleanser", "toner",
    "mask", "oil", "sunscreen", "exfoliant", "eye_care",
}
VALID_PRICE_BANDS = {"budget", "mid", "premium", "any"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        build_index()
    yield


app = FastAPI(title="Product Recommendation API", lifespan=lifespan)


class Preferences(BaseModel):
    formulation: List[str] = []   # ex: ["vegan", "clean"]
    origin: List[str] = []        # ex: ["french"]
    product_type: Optional[str] = None  # ex: "serum"
    price_band: Optional[str] = None    # "budget" | "mid" | "premium" | "any"


class RecommendationInput(BaseModel):
    skin_type: str  # "oily" | "dry" | "normal"
    acne: bool
    preferences: Preferences


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend_endpoint(payload: RecommendationInput):
    prefs = payload.preferences.model_dump()

    prefs["formulation"] = [f for f in prefs["formulation"] if f in VALID_FORMULATIONS]
    prefs["origin"] = [o for o in prefs["origin"] if o in VALID_ORIGINS]
    if prefs.get("product_type") not in VALID_PRODUCT_TYPES:
        prefs["product_type"] = None
    if prefs.get("price_band") not in VALID_PRICE_BANDS:
        prefs["price_band"] = None

    return recommend(
        skin_type=payload.skin_type,
        acne=payload.acne,
        preferences=prefs,
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
