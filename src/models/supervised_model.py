import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow_hub as hub
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime

class TagClassifier:
    def __init__(self, embedding_type='use', min_tag_freq=5):
        self.embedding_type = embedding_type
        self.min_tag_freq = min_tag_freq
        self.mlb = MultiLabelBinarizer()
        self.model = None
        self.embedder = None
        
        if embedding_type == 'use':
            print("Chargement du modèle Universal Sentence Encoder...")
            self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    def prepare_data(self, df):
        """Prépare les données pour l'entraînement."""
        # Filtrer les tags peu fréquents
        tag_counts = {}
        for tags in df['tags_list']:
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
        valid_tags = {tag for tag, count in tag_counts.items() if count >= self.min_tag_freq}
        
        # Filtrer les tags
        df['filtered_tags'] = df['tags_list'].apply(lambda x: [tag for tag in x if tag in valid_tags])
        
        # Transformer les tags en format binaire
        y = self.mlb.fit_transform(df['filtered_tags'])
        
        # Créer les embeddings
        if self.embedding_type == 'use':
            X = self.embedder(df['text'].values.tolist()).numpy()
        else:
            raise ValueError(f"Type d'embedding non supporté : {self.embedding_type}")
            
        return X, y
    
    def train(self, X, y):
        """Entraîne le modèle de classification."""
        print("Entraînement du modèle de classification...")
        self.model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        self.model.fit(X, y)
        
    def evaluate(self, X, y):
        """Évalue le modèle sur les données de test."""
        y_pred = self.model.predict(X)
        
        metrics = {
            'precision_micro': precision_score(y, y_pred, average='micro'),
            'recall_micro': recall_score(y, y_pred, average='micro'),
            'f1_micro': f1_score(y, y_pred, average='micro'),
            'precision_macro': precision_score(y, y_pred, average='macro'),
            'recall_macro': recall_score(y, y_pred, average='macro'),
            'f1_macro': f1_score(y, y_pred, average='macro')
        }
        
        return metrics
    
    def predict(self, texts, k=5):
        """Prédit les k tags les plus probables pour les textes donnés."""
        if self.embedding_type == 'use':
            X = self.embedder(texts).numpy()
        
        # Obtenir les probabilités de prédiction
        y_pred_proba = self.model.predict_proba(X)
        
        # Obtenir les k tags les plus probables
        predictions = []
        for probs in y_pred_proba:
            top_k_indices = np.argsort(probs)[-k:][::-1]
            predictions.append([
                (self.mlb.classes_[idx], probs[idx])
                for idx in top_k_indices
            ])
            
        return predictions
    
    def save_model(self, output_dir):
        """Sauvegarde le modèle et ses composants."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(output_dir, f'tag_classifier_{timestamp}.joblib')
        
        model_components = {
            'model': self.model,
            'mlb': self.mlb,
            'embedding_type': self.embedding_type,
            'min_tag_freq': self.min_tag_freq
        }
        
        joblib.dump(model_components, model_path)
        print(f"Modèle sauvegardé dans {model_path}")
        
def main():
    # Configuration de MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "Tag_Classification"
    mlflow.set_experiment(experiment_name)
    
    # Charger les données prétraitées
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    processed_data = pd.read_csv(os.path.join(data_dir, 'processed', 'processed_questions.csv'))
    
    # Initialiser et entraîner le modèle
    with mlflow.start_run():
        print("Préparation du classificateur...")
        classifier = TagClassifier(embedding_type='use', min_tag_freq=5)
        
        # Log des paramètres
        mlflow.log_params({
            "embedding_type": classifier.embedding_type,
            "min_tag_freq": classifier.min_tag_freq
        })
        
        # Préparation des données
        X, y = classifier.prepare_data(processed_data)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entraînement
        classifier.train(X_train, y_train)
        
        # Évaluation
        metrics = classifier.evaluate(X_test, y_test)
        print("\nMétriques d'évaluation:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
            mlflow.log_metric(metric_name, value)
        
        # Sauvegarde du modèle
        models_dir = os.path.join(data_dir, 'models')
        classifier.save_model(models_dir)
        mlflow.sklearn.log_model(classifier.model, "tag_classifier")
        
        print("\nEntraînement terminé et modèle sauvegardé.")
        
        # Test de prédiction
        test_questions = [
            "How to read CSV file in Python pandas?",
            "JavaScript async/await best practices"
        ]
        
        print("\nTest de prédiction:")
        predictions = classifier.predict(test_questions)
        for question, preds in zip(test_questions, predictions):
            print(f"\nQuestion: {question}")
            print("Tags prédits:")
            for tag, prob in preds:
                print(f"  - {tag}: {prob:.4f}")

if __name__ == "__main__":
    main() 