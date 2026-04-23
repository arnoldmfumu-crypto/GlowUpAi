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

ROUTINE_STEPS = {
    "matin": ["cleanser", "toner", "serum", "moisturizer"],
    "soir":  ["cleanser", "serum", "moisturizer"],
}
SKIN_RULES = {
    "oily": {
        "avoid": ["oil", "heavy", "comedogenic"],
        "prefer": ["oil-free", "non-comedogenic", "light", "gel"],
    },
    "dry": {
        "avoid": ["alcohol"],
        "prefer": ["hydrating", "ceramides", "rich"],
    }
}

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


def _build_query(skin_type: str, acne: bool, preferences: dict) -> str:
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


def _query(query_text: str, where: Optional[dict], n_results: int = 5) -> dict:
    collection = _get_collection()
    model      = _get_model()  # ← ton SentenceTransformer, enfin utilisé

    embedding = model.encode(query_text).tolist()  # même modèle qu'ingest.py

    try:
        kwargs: dict = {
            "query_embeddings": [embedding],  # ← vecteur, plus query_texts
            "n_results": n_results,
        }
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
                ),
            },
            {
                "role": "user",
                "content": f"Profil : {profile}\n\nProduit : {product_doc}",
            },
        ],
        max_tokens=350,
    )
    return response.choices[0].message.content

def routine_with_llm(routine: dict, skin_type: str, acne: bool) -> str:
    import os
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # 1. FIX: On boucle sur la 'routine' envoyée, pas sur le dictionnaire brut ROUTINE_STEPS
    routine_text = ""
    for moment, steps in routine.items():
        routine_text += f"\n{moment.upper()}:\n"
        for step in steps:
            if step["product_name"]:
                routine_text += f"- {step['etape']}: {step['product_name']} ({step['brand']})\n"
            else:
                routine_text += f"- {step['etape']}: Aucun produit\n"

    profile = f"peau {skin_type}"
    if acne:
        profile += ", tendance acnéique"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un influenceur beauté expert en skincare. "
                    "Propose ta routine skincare en suivant la routine skincare qui t'a été fournie. "
                    "1. Verifie que le produit recommandé est adapté et si c'est le cas propose le dans ta routine"
                    "2. Propose un produit adapte au resultat du type de peau. "
                    "3. Explique avec enthousiasme et brievement le rôle de chaque produit. "
                    "Réponds en français avec un ton chaleureux. "
                    "CRUCIAL : Tu dois ABSOLUMENT utiliser la syntaxe Markdown suivante pour structurer ta réponse.\n\n"
                    "Format OBLIGATOIRE:\n"
                    "Coucou ! Voici ta routine sur-mesure ✨ :\n\n"
                    "### 🌞 Matin\n"
                    "- **[product_type]** :[Nom du produit] [Ton explication...]\n\n"
                    "### 🌙 Soir\n"
                    "- **[product_type]** : [Nom du produit][Ton explication...]\n"
                    "J'espère que cette routine te conviendra 😎!"
                ),
            },
            {
                "role": "user",
                "content": f"Profil: {profile}\n\nRoutine exacte à présenter:\n{routine_text}",
            },
        ],
        max_tokens=550,
    )

    return response.choices[0].message.content


def is_compatible(meta: dict, skin_type: str) -> bool:
    """Vérifie que le produit ne contient pas d'ingrédients déconseillés."""
    rules = SKIN_RULES.get(skin_type, {})
    name  = meta.get("name", "").lower()
    return not any(word in name for word in rules.get("avoid", []))


# 2. FIX: Il FAUT garder cette fonction dé-commentée, c'est elle qui interroge ChromaDB !
def _build_routine_steps(
    skin_type: str,
    acne: bool,
    main_product: dict,
    preferences: dict
) -> dict:

    routine = {}

    for moment, steps in ROUTINE_STEPS.items():
        routine[moment] = []

        for step in steps:
            # Produit principal = on prend direct
            if main_product.get("product_type") == step:
                routine[moment].append({
                    "etape":         step,
                    "product_name":  main_product["product_name"],
                    "brand":         main_product["brand"],
                    "price_display": main_product["price_display"],
                    "is_vegan":      main_product["is_vegan"],
                    "is_clean":      main_product["is_clean"],
                    "is_main":       True,
                })
                continue

            # Sinon -> chercher un produit
            query = f"{step} pour peau {skin_type}"
            if acne and step in ("serum", "cleanser", "exfoliant"):
                query += " acnéique"

            if skin_type == "oily" and step == "serum":
                query += " oil-free serum gel"
            if skin_type == "oily" and step == "moisturizer":
                query += " lightweight oil-free gel moisturizer"
            if step == "toner":
                query += " toner astringent niacinamide"
            if step == "exfoliant":
                if acne:
                    query += " salicylic acid bha exfoliant" 

            where = {"product_type": {"$eq": step}}
            
            results = _query(query, where, n_results=5)

            if results["ids"][0]:
                candidates = results["metadatas"][0]
                chosen = None
                for m in candidates:
                    if is_compatible(m, skin_type):
                        chosen = m
                        break

                if not chosen:
                    chosen = candidates[0]
                    
                routine[moment].append({
                    "etape":         step,
                    "product_name":  chosen["name"],
                    "brand":         chosen["brand"],
                    "price_display": f"{chosen['price_eur']}€" if chosen["price_eur"] > 0 else "",
                    "is_vegan":      chosen["is_vegan"],
                    "is_clean":      chosen["is_clean"],
                    "is_main":       False,
                })
            else:
                routine[moment].append({
                    "etape":         step,
                    "product_name":  None,
                    "brand":         None,
                    "is_main":       False,
                })

    return routine


def recommend(skin_type: str, acne: bool, preferences: dict) -> dict:
    query_text = _build_query(skin_type, acne, preferences)
    where      = _build_where(preferences)

    results = _query(query_text, where, n_results=5)

    if not results["ids"][0] and where:
        results = _query(query_text, None, n_results=5)

    if not results["ids"][0]:
        results = _query(query_text, None, n_results=5)

    if not results["ids"][0]:
        return {"error": "Aucun produit trouvé"}

    meta = results["metadatas"][0][0]
    doc  = results["documents"][0][0]

    main_product_dict = {
        "product_name":  meta["name"],
        "brand":         meta["brand"],
        "product_type":  meta["product_type"],
        "price_display": f"{meta['price_eur']}€" if meta["price_eur"] > 0 else "",
        "is_vegan":      meta["is_vegan"],
        "is_clean":      meta["is_clean"],
    }

    # 3. FIX: On dé-commente la création de la routine !
    routine_complete = _build_routine_steps(skin_type, acne, main_product_dict, preferences)
    
    # 4. FIX: On passe bien `routine_complete` à l'influenceur LLM
    validated_routine = routine_with_llm(
        routine=routine_complete,
        skin_type=skin_type,
        acne=acne
    )

    explanation = _generate_explanation(
        doc,
        skin_type,
        acne,
        preferences
    )
    
    return {
        "product_name":  meta["name"],
        "brand":         meta["brand"],
        "product_type":  meta["product_type"],
        "price_display": f"{meta['price_eur']}€" if meta["price_eur"] > 0 else "",
        "source":        meta["source"],
        "is_french":     meta["is_french"],
        "is_vegan":      meta["is_vegan"],
        "is_clean":      meta["is_clean"],
        "explanation":   explanation,
        "routine":       routine_complete,
        "routine_validated": validated_routine
    }