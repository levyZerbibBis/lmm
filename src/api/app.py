import os
import joblib
from flask import Flask, request, jsonify
from datetime import datetime
import tensorflow_hub as hub
import numpy as np

app = Flask(__name__)

# Charger le modèle
def load_model():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'models')
    model_files = [f for f in os.listdir(models_dir) if f.startswith('tag_classifier_')]
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    
    print(f"Chargement du modèle depuis {model_path}")
    model_components = joblib.load(model_path)
    
    return model_components

# Initialisation globale
print("Initialisation de l'API...")
model_components = load_model()
classifier = model_components['model']
mlb = model_components['mlb']
embedding_type = model_components['embedding_type']

if embedding_type == 'use':
    print("Chargement du modèle Universal Sentence Encoder...")
    embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

@app.route('/health', methods=['GET'])
def health_check():
    """Point de terminaison pour vérifier la santé de l'API."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_tags():
    """Point de terminaison pour prédire les tags d'une question."""
    try:
        # Récupérer les données de la requête
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Le champ "text" est requis dans le corps de la requête'
            }), 400
            
        question_text = data['text']
        k = data.get('k', 5)  # Nombre de tags à prédire (par défaut: 5)
        
        # Créer l'embedding
        if embedding_type == 'use':
            X = embedder([question_text]).numpy()
        
        # Obtenir les probabilités de prédiction
        y_pred_proba = classifier.predict_proba(X)
        
        # Obtenir les k tags les plus probables
        top_k_indices = np.argsort(y_pred_proba[0])[-k:][::-1]
        predictions = [
            {
                'tag': mlb.classes_[idx],
                'probability': float(y_pred_proba[0][idx])
            }
            for idx in top_k_indices
        ]
        
        return jsonify({
            'question': question_text,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Point de terminaison pour obtenir des informations sur le modèle."""
    return jsonify({
        'embedding_type': embedding_type,
        'min_tag_freq': model_components['min_tag_freq'],
        'number_of_tags': len(mlb.classes_),
        'available_tags': list(mlb.classes_),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 