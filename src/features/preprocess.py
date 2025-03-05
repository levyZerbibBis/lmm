import re
import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import os

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = CountVectorizer(max_features=10000)
        
    def clean_html(self, text):
        """Supprime les balises HTML du texte."""
        return BeautifulSoup(text, 'html.parser').get_text()
    
    def clean_text(self, text):
        """Nettoie le texte en supprimant la ponctuation et les caractères spéciaux."""
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Supprimer la ponctuation et les nombres
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize et lemmatise le texte."""
        # Tokenization
        tokens = word_tokenize(text)
        
        # Suppression des stop words et lemmatisation
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def process_tags(self, tags):
        """Convertit la chaîne de tags en liste."""
        # Supprimer les caractères <> et séparer les tags
        return tags.replace('<', ' ').replace('>', ' ').strip().split()
    
    def create_bow(self, texts):
        """Crée une représentation Bag of Words des textes."""
        return self.vectorizer.fit_transform(texts)

class DataPreprocessor:
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        
    def load_data(self, filename):
        """Charge les données depuis un fichier CSV."""
        return pd.read_csv(os.path.join(self.data_dir, 'raw', filename))
    
    def preprocess_data(self, df):
        """Prétraite les données."""
        # Prétraitement du titre et du corps
        df['clean_title'] = df['Title'].apply(self.text_preprocessor.clean_html)
        df['clean_title'] = df['clean_title'].apply(self.text_preprocessor.clean_text)
        df['clean_title'] = df['clean_title'].apply(self.text_preprocessor.tokenize_and_lemmatize)
        
        df['clean_body'] = df['Body'].apply(self.text_preprocessor.clean_html)
        df['clean_body'] = df['clean_body'].apply(self.text_preprocessor.clean_text)
        df['clean_body'] = df['clean_body'].apply(self.text_preprocessor.tokenize_and_lemmatize)
        
        # Combiner titre et corps
        df['text'] = df['clean_title'] + ' ' + df['clean_body']
        
        # Prétraitement des tags
        df['tags_list'] = df['Tags'].apply(self.text_preprocessor.process_tags)
        
        return df
    
    def save_processed_data(self, df, filename):
        """Sauvegarde les données prétraitées."""
        processed_dir = os.path.join(self.data_dir, 'processed')
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
            
        output_path = os.path.join(processed_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Données prétraitées sauvegardées dans {output_path}")

def main():
    # Initialiser le préprocesseur
    preprocessor = DataPreprocessor()
    
    # Charger les données
    print("Chargement des données...")
    # Vous devrez spécifier le nom correct du fichier CSV
    raw_data = preprocessor.load_data('stackoverflow_api_latest.csv')
    
    # Prétraiter les données
    print("Prétraitement des données...")
    processed_data = preprocessor.preprocess_data(raw_data)
    
    # Sauvegarder les données prétraitées
    preprocessor.save_processed_data(processed_data, 'processed_questions.csv')

if __name__ == "__main__":
    main() 