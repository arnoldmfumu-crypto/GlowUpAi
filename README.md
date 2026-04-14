## Installation

### 1. Installer uv

**Mac / Linux**
curl -LsSf https://astral.sh/uv/install.sh | sh

**Windows (PowerShell)**
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

Relancer le terminal après installation.

### 2. Installer les dépendances et lancer

uv sync

## Données

Les datasets bruts sont inclus dans le repo (`data/raw/`) pour faciliter 
la reproductibilité dans le cadre du projet. Il s'agit de datasets publics 
disponibles sur Kaggle :
- [Oily, Dry and Normal Skin Types](https://www.kaggle.com/datasets/shakyadissanayake/oily-dry-and-normal-skin-types-dataset)
- [Acne Dataset](https://www.kaggle.com/datasets/nayanchaure/acne-dataset)
- [Skincare Products and their Ingredients](https://www.kaggle.com/datasets/eward96/skincare-products-and-their-ingredients)
- [Amazon Skincare Products](https://www.kaggle.com/datasets/namantrisoliya/amazon-skincare-products)
- [Dermstore Skincare Products](https://www.kaggle.com/datasets/crawlfeeds/dermstore-skincare-products-and-ingredients-dataset)