# Variables
PYTHON = python
VENV = venv
PIP = $(VENV)/bin/pip
FLASK = $(VENV)/bin/flask
PORT = 5000

# Commandes principales
.PHONY: setup data train api clean mlflow test all

all: setup data train api

# Installation et configuration
setup: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

# Collecte et prétraitement des données
data: setup
	$(PYTHON) src/data/collect_data.py
	$(PYTHON) src/features/preprocess.py

# Entraînement des modèles
train: setup
	$(PYTHON) src/models/lda_model.py
	$(PYTHON) src/models/supervised_model.py

# Lancement de l'API
api: setup
	$(FLASK) run --host=0.0.0.0 --port=$(PORT)

# MLflow UI
mlflow: setup
	mlflow ui

# Tests
test: setup
	pytest tests/

# Nettoyage
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "mlruns" -exec rm -rf {} +

# Aide
help:
	@echo "Commandes disponibles:"
	@echo "  make setup   - Crée l'environnement virtuel et installe les dépendances"
	@echo "  make data    - Collecte et prétraite les données"
	@echo "  make train   - Entraîne les modèles LDA et supervisé"
	@echo "  make api     - Lance l'API Flask"
	@echo "  make mlflow  - Lance l'interface MLflow"
	@echo "  make test    - Lance les tests"
	@echo "  make clean   - Nettoie les fichiers temporaires"
	@echo "  make all     - Exécute setup, data, train et api" 