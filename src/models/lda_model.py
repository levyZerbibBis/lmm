import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
import mlflow
import mlflow.sklearn
from datetime import datetime

class LDATopicModeling:
    def __init__(self, n_topics=10, max_features=10000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='batch'
        )
        
    def prepare_data(self, texts):
        """Prépare les données pour le modèle LDA."""
        return self.vectorizer.fit_transform(texts)
        
    def train(self, X):
        """Entraîne le modèle LDA."""
        return self.lda_model.fit(X)
    
    def get_topic_words(self, n_words=10):
        """Récupère les mots les plus importants pour chaque topic."""
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[:-n_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics[f"Topic {topic_idx+1}"] = top_words
            
        return topics
    
    def create_pyldavis_visualization(self, X):
        """Crée une visualisation interactive des topics avec pyLDAvis."""
        vis_data = pyLDAvis.sklearn.prepare(
            self.lda_model,
            X,
            self.vectorizer,
            mds='tsne'
        )
        return vis_data
    
    def save_visualization(self, vis_data, output_dir):
        """Sauvegarde la visualisation au format HTML."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'lda_visualization_{timestamp}.html')
        pyLDAvis.save_html(vis_data, output_path)
        print(f"Visualisation sauvegardée dans {output_path}")
        
def main():
    # Configuration de MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    experiment_name = "LDA_Topic_Modeling"
    mlflow.set_experiment(experiment_name)
    
    # Charger les données prétraitées
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    processed_data = pd.read_csv(os.path.join(data_dir, 'processed', 'processed_questions.csv'))
    
    # Initialiser et entraîner le modèle LDA
    with mlflow.start_run():
        print("Entraînement du modèle LDA...")
        lda = LDATopicModeling(n_topics=10, max_features=10000)
        
        # Log des paramètres
        mlflow.log_params({
            "n_topics": lda.n_topics,
            "max_features": lda.max_features
        })
        
        # Préparation des données
        X = lda.prepare_data(processed_data['text'])
        
        # Entraînement
        lda.train(X)
        
        # Récupération des topics
        topics = lda.get_topic_words()
        for topic_name, words in topics.items():
            print(f"\n{topic_name}:")
            print(", ".join(words))
            mlflow.log_text(", ".join(words), f"{topic_name}.txt")
        
        # Création et sauvegarde de la visualisation
        print("\nCréation de la visualisation...")
        vis_data = lda.create_pyldavis_visualization(X)
        vis_dir = os.path.join(data_dir, 'visualizations')
        lda.save_visualization(vis_data, vis_dir)
        
        # Sauvegarde du modèle
        mlflow.sklearn.log_model(lda.lda_model, "lda_model")
        
        print("\nEntraînement terminé et modèle sauvegardé.")

if __name__ == "__main__":
    main() 