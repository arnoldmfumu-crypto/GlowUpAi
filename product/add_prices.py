from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "skincare_products"
CLEAN_DIR = Path(__file__).parent / "data"


def add_prices():
    # --- Dermstore: join on title to recover price + currency ---
    clean = pd.read_csv(CLEAN_DIR / "dermstore_skincare_cleaned.csv")
    if "price" not in clean.columns:
        raw = pd.read_csv(RAW_DIR / "dermstore_data.csv", usecols=["title", "price", "currency"])
        raw["title"] = raw["title"].str.strip()
        raw = raw.drop_duplicates("title")
        clean["title"] = clean["title"].str.strip()
        clean = clean.merge(raw, on="title", how="left")
        clean.to_csv(CLEAN_DIR / "dermstore_skincare_cleaned.csv", index=False)
        print(f"Dermstore: {clean['price'].notna().sum()}/{len(clean)} produits avec prix")
    else:
        print("Dermstore: colonne price déjà présente, skip")

    # --- LookFantastic: price existe mais format '£5.20', normaliser ---
    clean = pd.read_csv(CLEAN_DIR / "lookfantastic_skincare_cleaned.csv")
    if "currency" not in clean.columns:
        clean["price"] = (
            clean["price"]
            .astype(str)
            .str.replace("£", "", regex=False)
            .str.strip()
        )
        clean["price"] = pd.to_numeric(clean["price"], errors="coerce")
        clean["currency"] = "GBP"
        clean.to_csv(CLEAN_DIR / "lookfantastic_skincare_cleaned.csv", index=False)
        print(f"LookFantastic: {clean['price'].notna().sum()}/{len(clean)} produits avec prix")
    else:
        print("LookFantastic: colonne currency déjà présente, skip")

    # --- Amazon: prix en INR, non fiables → null ---
    clean = pd.read_csv(CLEAN_DIR / "amazon_skincare_cleaned.csv")
    if "price" not in clean.columns:
        clean["price"] = None
        clean["currency"] = None
        clean.to_csv(CLEAN_DIR / "amazon_skincare_cleaned.csv", index=False)
        print(f"Amazon: {len(clean)} produits sans prix (INR source, non exploitable)")
    else:
        print("Amazon: colonne price déjà présente, skip")


if __name__ == "__main__":
    add_prices()
