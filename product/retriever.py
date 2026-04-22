import os
from pathlib import Path
from typing import Optional

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "skincare_products"

_model: Optional[SentenceTransformer] = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _build_query(skin_type: str, acne: bool, preferences: dict) -> str: #il manque le skin type ici !!!!
    parts = [f"skincare product for {skin_type} skin"]
    if acne:
        parts.append("acne-prone")
    if pt := preferences.get("product_type"):
        parts.append(pt)
    for f in preferences.get("formulation", []):
        parts.append(f)
    if "french" in preferences.get("origin", []):
        parts.append("French brand")
    return ", ".join(parts)


def _build_where(preferences: dict) -> Optional[dict]:
    conditions = []
    for flag, key in [("vegan", "is_vegan"), ("clean", "is_clean"), ("bio", "is_bio")]:
        if flag in preferences.get("formulation", []):
            conditions.append({key: {"$eq": True}})
    if "french" in preferences.get("origin", []):
        conditions.append({"is_french": {"$eq": True}})
    if (pb := preferences.get("price_band")) and pb not in ("any", None):
        conditions.append({"price_band": {"$eq": pb}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _query(query_text: str, where: Optional[dict]) -> dict:
    collection = _get_collection()
    try:
        kwargs: dict = {"query_texts": [query_text], "n_results": 1}
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)
    except Exception:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]]}


def _generate_explanation(product_doc: str, skin_type: str, acne: bool, preferences: dict) -> str:
    prefs = []
    for f in preferences.get("formulation", []):
        prefs.append(f)
    if "french" in preferences.get("origin", []):
        prefs.append("marque française")
    if pt := preferences.get("product_type"):
        prefs.append(f"type : {pt}")
    if pb := preferences.get("price_band"):
        prefs.append(f"budget : {pb}")

    profile = f"peau {skin_type}"
    if acne:
        profile += ", tendance acnéique"
    if prefs:
        profile += f", préférences : {', '.join(prefs)}"

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un expert en soins de la peau. "
                    "En 2 à 3 phrases courtes en français, explique pourquoi ce produit "
                    "est adapté au profil de l'utilisateur. Sois précis et bienveillant. "
                    "Ne répète pas le nom du produit en entier."
                ),
            },
            {
                "role": "user",
                "content": f"Profil : {profile}\n\nProduit : {product_doc}",
            },
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content


def recommend(skin_type: str, acne: bool, preferences: dict) -> dict:
    query_text = _build_query(skin_type, acne, preferences)
    where = _build_where(preferences)

    results = _query(query_text, where)

    # Fallback : relâcher les filtres si aucun résultat
    if not results["ids"][0] and where:
        results = _query(query_text, None)

    if not results["ids"][0]:
        return {"error": "Aucun produit trouvé"}

    meta = results["metadatas"][0][0]
    doc = results["documents"][0][0]

    explanation = _generate_explanation(doc, skin_type, acne, preferences)

    return {
        "product_name": meta["name"],
        "brand": meta["brand"],
        "product_type": meta["product_type"],
        "price_display": f"{meta['price_eur']}€" if meta["price_eur"] > 0 else "",
        "source": meta["source"],
        "is_french": meta["is_french"],
        "is_vegan": meta["is_vegan"],
        "is_clean": meta["is_clean"],
        "explanation": explanation,
    }
