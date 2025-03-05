import os
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class StackOverflowDataCollector:
    def __init__(self):
        self.api_key = os.getenv('STACKOVERFLOW_API_KEY')
        self.base_url = 'https://api.stackexchange.com/2.3'
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw')
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def fetch_questions_api(self, tag='python', min_score=50, page_size=50):
        """Récupère les questions via l'API Stack Exchange."""
        endpoint = f'{self.base_url}/questions'
        
        params = {
            'order': 'desc',
            'sort': 'votes',
            'tagged': tag,
            'site': 'stackoverflow',
            'filter': '!9Z(-wwYGT',  # Filtre personnalisé pour inclure le corps et les tags
            'pagesize': page_size,
            'key': self.api_key
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            questions = response.json()['items']
            
            # Conversion en DataFrame
            df = pd.DataFrame(questions)
            
            # Sauvegarde des données brutes
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(self.data_dir, f'stackoverflow_api_{timestamp}.csv')
            df.to_csv(output_file, index=False)
            
            print(f"Données sauvegardées dans {output_file}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête API : {e}")
            return None

    def save_query_results(self, query_results, filename):
        """Sauvegarde les résultats de la requête SQL dans un fichier CSV."""
        output_file = os.path.join(self.data_dir, filename)
        query_results.to_csv(output_file, index=False)
        print(f"Données sauvegardées dans {output_file}")

def main():
    collector = StackOverflowDataCollector()
    
    # Récupération des données via l'API
    print("Récupération des données via l'API Stack Exchange...")
    questions_df = collector.fetch_questions_api()
    
    if questions_df is not None:
        print(f"Nombre de questions récupérées : {len(questions_df)}")
    else:
        print("Échec de la récupération des données via l'API")

if __name__ == "__main__":
    main() 