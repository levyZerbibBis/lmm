# Système de Suggestion Automatique de Tags Stack Overflow

Ce projet implémente un système de suggestion automatique de tags pour les questions Stack Overflow en utilisant des techniques de machine learning supervisé et non supervisé.

## 🌟 Fonctionnalités

- Collecte de données via Stack Exchange Data Explorer et API
- Prétraitement des données textuelles
- Modélisation non supervisée avec LDA
- Classification supervisée multi-label des tags
- API REST Flask pour les prédictions
- Suivi des expériences avec MLflow
- Pipeline MLOps avec Kedro
- Monitoring en production avec EvidentlyAI

## 🚀 Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd stackoverflow-tag-suggester
```

2. Créer un environnement virtuel :
```bash
python3 -m venv venv
```

```bash
source venv/bin/activate  # Linux/Mac
```

```bash
.\venv\Scripts\activate   # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 📂 Structure du Projet

```
stackoverflow-tag-suggester/
├── data/                      # Données brutes et prétraitées
├── notebooks/                 # Notebooks Jupyter
├── src/                      # Code source
│   ├── data/                 # Scripts de collecte de données
│   ├── features/             # Prétraitement et features
│   ├── models/               # Modèles ML
│   └── api/                  # API Flask
├── tests/                    # Tests unitaires
├── mlruns/                   # Logs MLflow
├── conf/                     # Configuration Kedro
└── deployment/               # Fichiers de déploiement
```

## 🔧 Utilisation

1. Collecte des données :
```bash
python3 src/data/collect_data.py
```

2. Entraînement des modèles :
```bash
python3 src/models/train_models.py
```

3. Lancement de l'API :
```bash
python3 src/api/app.py
```

## 📊 MLflow UI

Pour visualiser les expériences :
```bash
mlflow ui
```

## 🐳 Docker

Construction de l'image :
```bash
docker build -t tag-suggester -f deployment/Dockerfile .
```

## 📝 Documentation

- [Documentation technique](docs/technical.md)
- [Guide API](docs/api.md)
- [MLOps Pipeline](docs/mlops.md)

## 🤝 Contribution

Les contributions sont les bienvenues ! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour les directives.

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails. 