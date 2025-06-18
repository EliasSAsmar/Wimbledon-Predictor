from flask import Flask, render_template, jsonify, request
from prediction import PredictionInterface
from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
import logging
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

app = Flask(__name__, 
            static_folder='../frontend/static',
            template_folder='../frontend/templates')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
data_loader = TennisDataLoader()
feature_engine = TennisFeatureEngine()
model_trainer = ModelTrainer()

# Model paths
MODEL_DIR = PROJECT_ROOT / 'models'
RF_MODEL_PATH = MODEL_DIR / 'corrected_wimbledon_rf_model.pkl'
XGB_MODEL_PATH = MODEL_DIR / 'corrected_wimbledon_xgb_model.pkl'

# Load models
try:
    # Check if model files exist
    if not RF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Random Forest model not found at {RF_MODEL_PATH}")
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found at {XGB_MODEL_PATH}")
    
    # Load models
    model_trainer.load_models(str(RF_MODEL_PATH), str(XGB_MODEL_PATH))
    predictor = PredictionInterface(feature_engine, model_trainer)
    
    # Load match data
    matches = data_loader.load_raw_data()
    if matches is None or len(matches) == 0:
        raise ValueError("No match data loaded")
        
    logger.info(f"Models and data loaded successfully. {len(matches)} matches available.")
    
except FileNotFoundError as e:
    logger.error(f"Model file error: {e}")
    predictor = None
    matches = None
except Exception as e:
    logger.error(f"Failed to initialize prediction system: {e}")
    predictor = None
    matches = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None or matches is None:
        return jsonify({'error': 'Prediction system not initialized'}), 500
    
    try:
        data = request.get_json()
        player1 = data.get('player1')
        player2 = data.get('player2')
        
        if not player1 or not player2:
            return jsonify({'error': 'Both players required'}), 400
            
        prediction = predictor.predict_match_outcome(player1, player2, matches)
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/players', methods=['GET'])
def get_players():
    if matches is None:
        return jsonify({'error': 'Match data not loaded'}), 500
        
    try:
        players = sorted(set(matches['winner_name'].unique()) | set(matches['loser_name'].unique()))
        return jsonify({'players': players})
        
    except Exception as e:
        logger.error(f"Error getting players: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 