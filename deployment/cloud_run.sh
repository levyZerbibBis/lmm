#!/bin/bash

# Variables d'environnement
export FLASK_APP=src/api/app.py
export FLASK_ENV=production
export PORT=${PORT:-5000}

# Installer les dépendances
pip install -r requirements.txt

# Démarrer l'application avec Gunicorn
exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 src.api.app:app 