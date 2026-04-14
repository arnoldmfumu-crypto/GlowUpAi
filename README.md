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

# Docker

Docker garantit que l'application tourne de façon **identique sur toutes les machines** (Mac M1, Windows, Linux), sans problème de version Python, de librairie système ou de variable d'environnement.

> **Règle d'or** : Docker est réservé à l'**intégration** et à la **démo**.  

## Prérequis

Vérifier que Docker est bien lancé :

```bash
docker --version
docker-compose --version
```

---

## Structure des conteneurs

```
skinmatch/
├── docker-compose.yml     ← orchestre les 2 conteneurs
├── api/
│   └── Dockerfile         ← conteneur FastAPI (port 8000)
└── app/
    └── Dockerfile         ← conteneur Streamlit (port 8501)
```

Le modèle Vision et le vectorstore Chroma **ne sont pas dans les conteneurs**.
Ils sont montés en local via des volumes. 
Pas besoin de rebuilder l'image quand le modèle change.

---

## Lancement

### Premier lancement (build des images, ~5-10 min)

```bash
docker-compose up --build
```

### Lancements suivants (instantané)

```bash
docker-compose up
```

### Arrêt

```bash
# Arrêt propre (Ctrl+C puis)
docker-compose down

# Arrêt + suppression des volumes (reset complet)
docker-compose down -v
```

---

## Accès aux services

| Service | URL | Description |
|---------|-----|-------------|
| Streamlit (frontend) | http://localhost:8501 | Interface utilisateur |
| FastAPI (backend) | http://localhost:8000 | API REST |
| Documentation API | http://localhost:8000/docs | Swagger auto-généré |
| Health check | http://localhost:8000/health | Vérification API |

---

## Variables d'environnement

Créer un fichier `.env` à la racine à partir du template :

```bash
cp .env.example .env
```

Puis remplir les valeurs dans `.env` :

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
HF_TOKEN=your_huggingface_token_here
API_URL=http://localhost:8000
```

> ⚠️ `.env` est dans le `.gitignore` pour ne jamais le commiter.  
> `.env.example` est commité pour montrer la structure sans les valeurs.

---

## Volumes — comment ça marche

Les volumes connectent des dossiers en local aux dossiers dans les conteneurs :

```
En local                    Conteneur API
─────────────────────────────────────────────
./data/              ←→   /app/data/
./vision/models/     ←→   /app/vision/models/
./rag/vectorstore/   ←→   /app/rag/vectorstore/
```

**Conséquence pratique** : quand vous entraînez un nouveau modèle en local avec `uv run python vision/train.py`, le checkpoint `.pt` sauvegardé dans `vision/models/` est **immédiatement disponible** dans le conteneur API sans rebuild.

---

### Pour tester l'intégration complète (J7-J8)

```bash
docker-compose up --build
# Ouvrir localhost:8501 et tester le pipeline complet
```

### Pour la démo Demoday

```bash
docker-compose up
# L'application est prête sur localhost:8501
```

---