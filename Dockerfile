# Image de base légère
FROM python:3.9-slim

# Répertoire de travail
WORKDIR /app

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source (backend)
COPY ./backend ./backend

# Port d'écoute de FastAPI
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]