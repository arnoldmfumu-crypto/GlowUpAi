# On part d'une image Python officielle — "slim" veut dire version allégée
# sans les outils de compilation inutiles. python:3.11-slim pèse ~150 MB
# vs python:3.11 qui pèse ~900 MB.
FROM python:3.11-slim

# On définit le dossier de travail à l'intérieur du conteneur.
# Toutes les commandes suivantes s'exécutent depuis /app.
# C'est l'équivalent d'un cd /app.
WORKDIR /app

# On installe des librairies SYSTÈME (pas Python) nécessaires à OpenCV
# et MediaPipe qui ont besoin de librairies C pour traiter les images.
# libgl1 et libglib2.0-0 sont des dépendances graphiques bas niveau.
# Sans elles, "import cv2" plante dans le conteneur même si opencv
# est dans requirements.txt.
# rm -rf /var/lib/apt/lists/* nettoie le cache apt pour réduire
# la taille de l'image finale.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# On copie UNIQUEMENT requirements.txt en premier — pas tout le code.
# Pourquoi ? Docker construit les images par couches (layers).
# Si requirements.txt n'a pas changé, Docker réutilise le layer
# pip install depuis son cache → rebuild 10x plus rapide.
# Si on copiait tout le code d'abord, le moindre changement
# dans train.py forcerait un pip install complet à chaque rebuild.
COPY requirements.txt .

# On installe les dépendances Python.
# --no-cache-dir évite de stocker le cache pip dans l'image
# (économise ~200 MB sur l'image finale).
RUN pip install --no-cache-dir -r requirements.txt

# Maintenant seulement on copie tout le code.
# Le "." source = tout le projet depuis la racine (context du build).
# Le "." destination = /app dans le conteneur.
COPY . .

# On documente que le conteneur écoute sur le port 8000.
# EXPOSE ne publie pas réellement le port — c'est de la documentation.
# La publication réelle se fait dans docker-compose.yml avec "ports:".
EXPOSE 8000

# La commande lancée quand le conteneur démarre.
# --host 0.0.0.0 est crucial : par défaut uvicorn écoute sur 127.0.0.1
# (localhost du conteneur) ce qui le rend inaccessible de l'extérieur.
# 0.0.0.0 signifie "écoute sur toutes les interfaces" → accessible
# depuis votre machine hôte et depuis le conteneur Streamlit.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]