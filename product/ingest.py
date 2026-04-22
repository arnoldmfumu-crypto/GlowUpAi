from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from french_brands import is_french

CLEAN_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "skincare_products"

USD_TO_EUR = 0.92
GBP_TO_EUR = 1.17

# Product type detection — ordered by specificity (most specific first)
PRODUCT_TYPE_KEYWORDS = [
    ("sunscreen",    ["spf", "sunscreen", "sun protection", "solaire", "solar"]),
    ("eye_care",     ["eye cream", "eye serum", "eye gel", "contour des yeux", "eye care"]),
    ("serum",        ["serum", "sérum", "essence", "ampoule", "concentrate"]),
    ("oil",          ["face oil", "dry oil", "huile visage", "facial oil"]),
    ("mask",         ["mask", "masque", "sheet mask"]),
    ("exfoliant",    ["exfoliant", "scrub", "peel", "exfoliating", " aha ", " bha ", "acid toner"]),
    ("cleanser",     ["cleanser", "cleansing", "face wash", "foaming", "micellar", "gel nettoyant", "mousse nettoyante"]),
    ("toner",        ["toner", "toning lotion", "lotion tonique", "mist", "facial mist"]),
    ("moisturizer",  ["moisturizer", "moisturiser", "cream", "crème", "lotion", "balm", "baume", "gel moistur"]),
]

NON_VEGAN = [
    "beeswax", "cera alba", "lanolin", "carmine", "collagen", "elastin",
    "gelatin", "gelatine", "honey", "mel ", "silk", "sericin",
    "squalene", "tallow", "lard", "keratin", "casein", "chitosan",
    "guanine", "shellac",
]

NON_CLEAN = [
    "paraben", "sodium lauryl sulfate", "sodium laureth sulfate",
    "dimethicone", "cyclopentasiloxane", "cyclohexasiloxane", "cyclomethicone",
    "phthalate", "formaldehyde", "petrolatum", "mineral oil",
    "polyethylene glycol", "peg-", "butylated hydroxytoluene", "bht",
    "butylated hydroxyanisole", "bha ",
]

BIO_KEYWORDS = [
    "organic", "bio ", "ecocert", "cosmos organic", "certifié bio",
    "certified organic", "usda organic",
]


def _to_eur(price, currency: str) -> float | None:
    try:
        price = float(price)
    except (TypeError, ValueError):
        return None
    if currency == "USD":
        return round(price * USD_TO_EUR, 2)
    if currency == "GBP":
        return round(price * GBP_TO_EUR, 2)
    return None


def _price_band(price_eur: float | None) -> str:
    if price_eur is None:
        return "unknown"
    if price_eur < 15:
        return "budget"
    if price_eur <= 50:
        return "mid"
    return "premium"


def _normalize_type(raw_type: str, title: str = "") -> str:
    text = f"{raw_type or ''} {title or ''}".lower()
    for product_type, keywords in PRODUCT_TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return product_type
    return "other"


def _detect_vegan(ingredients: str, description: str = "") -> bool:
    combined = f"{ingredients or ''} {description or ''}".lower()
    if "vegan" in combined:
        return True
    if not ingredients:
        return False
    return not any(nv in ingredients.lower() for nv in NON_VEGAN)


def _detect_clean(ingredients: str) -> bool:
    if not ingredients:
        return False
    ingr = ingredients.lower()
    return not any(nc in ingr for nc in NON_CLEAN)


def _detect_bio(ingredients: str, description: str = "", title: str = "") -> bool:
    text = f"{ingredients or ''} {description or ''} {title or ''}".lower()
    return any(bk in text for bk in BIO_KEYWORDS)


def _make_document(r: dict) -> str:
    parts = []
    type_label = r["product_type"].replace("_", " ")
    parts.append(f"{type_label.title()} par {r['brand'] or 'marque inconnue'} : {r['name']}.")
    if r["skin_type"]:
        parts.append(f"Convient pour : {r['skin_type'][:120]}.")
    if r["description"]:
        parts.append(r["description"][:300])
    if r["ingredients"]:
        parts.append(f"Ingrédients : {r['ingredients'][:300]}.")
    flags = []
    if r["is_vegan"]:
        flags.append("vegan")
    if r["is_clean"]:
        flags.append("clean")
    if r["is_bio"]:
        flags.append("bio")
    if r["is_french"]:
        flags.append("marque française")
    if flags:
        parts.append(f"Labels : {', '.join(flags)}.")
    if r["price_eur"] is not None:
        parts.append(f"Prix : {r['price_eur']}€.")
    return " ".join(parts)


def _load_amazon() -> list[dict]:
    df = pd.read_csv(CLEAN_DIR / "amazon_skincare_cleaned.csv")
    records = []
    for _, row in df.iterrows():
        records.append({
            "name": str(row["Title"]).strip(),
            "brand": str(row["Brand"]).strip(),
            "product_type": _normalize_type(str(row["Product"]), str(row["Title"])),
            "description": "",
            "ingredients": "",
            "skin_type": str(row.get("Skin_Type", "")).lower().strip(),
            "price_eur": None,
            "source": "amazon",
        })
    return records


def _load_dermstore() -> list[dict]:
    df = pd.read_csv(CLEAN_DIR / "dermstore_skincare_cleaned.csv")
    records = []
    for _, row in df.iterrows():
        price_eur = _to_eur(row.get("price"), str(row.get("currency", "USD")))
        records.append({
            "name": str(row["title"]).strip(),
            "brand": str(row["brand"]).strip(),
            "product_type": _normalize_type("", str(row["title"])),
            "description": str(row.get("description", ""))[:500],
            "ingredients": str(row.get("ingredients", "")),
            "skin_type": str(row.get("skin_type_and_concerns", "")).lower()[:200],
            "price_eur": price_eur,
            "source": "dermstore",
        })
    return records


def _load_lookfantastic() -> list[dict]:
    df = pd.read_csv(CLEAN_DIR / "lookfantastic_skincare_cleaned.csv")
    records = []
    for _, row in df.iterrows():
        price_eur = _to_eur(row.get("price"), "GBP")
        records.append({
            "name": str(row["product_name"]).strip(),
            "brand": "",
            "product_type": _normalize_type(str(row.get("product_type", "")), str(row["product_name"])),
            "description": "",
            "ingredients": str(row.get("ingredients", "")),
            "skin_type": "",
            "price_eur": price_eur,
            "source": "lookfantastic",
        })
    return records


def build_index():
    print("Chargement des produits...")
    records = _load_amazon() + _load_dermstore() + _load_lookfantastic()
    print(f"{len(records)} produits chargés")

    for r in records:
        r["is_vegan"] = _detect_vegan(r["ingredients"], r["description"])
        r["is_clean"] = _detect_clean(r["ingredients"])
        r["is_bio"] = _detect_bio(r["ingredients"], r["description"], r["name"])
        r["is_french"] = is_french(r["brand"])
        r["price_band"] = _price_band(r["price_eur"])

    documents = [_make_document(r) for r in records]
    ids = [f"product_{i}" for i in range(len(records))]

    metadatas = [
        {
            "name": r["name"],
            "brand": r["brand"] or "",
            "product_type": r["product_type"],
            "price_eur": float(r["price_eur"]) if r["price_eur"] is not None else -1.0,
            "price_band": r["price_band"],
            "source": r["source"],
            "is_vegan": r["is_vegan"],
            "is_clean": r["is_clean"],
            "is_bio": r["is_bio"],
            "is_french": r["is_french"],
        }
        for r in records
    ]

    print("Génération des embeddings...", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    print("Stockage dans ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    batch_size = 500
    for i in range(0, len(records), batch_size):
        collection.add(
            documents=documents[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )

    print(f"Index construit : {len(records)} produits indexés dans {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
