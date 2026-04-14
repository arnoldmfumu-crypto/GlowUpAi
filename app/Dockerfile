# Même image de base que l'API — cohérence entre les deux conteneurs.
FROM python:3.11-slim

WORKDIR /app

# Pas de libgl1 / libglib2.0-0 ici — le frontend Streamlit ne fait
# pas de traitement d'image en local, il envoie juste les bytes à l'API.
# Moins de dépendances = image plus légère et build plus rapide.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port par défaut de Streamlit.
EXPOSE 8501

# --server.address=0.0.0.0 pour la même raison que l'API :
# accessible depuis l'extérieur du conteneur (votre navigateur).
# --server.port=8501 pour être explicite.
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]