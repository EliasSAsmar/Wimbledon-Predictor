import logging
import pickle
import pandas as pd
from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
from prediction import PredictionInterface
from ensemble_models import RF_SR_Ensemble # Import needed for unpickling
from utils.odds_api import fetch_wimbledon_odds

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def initialize_engine():
    """Initialize and return the prediction engine components and matches data"""
    data_loader = TennisDataLoader(data_dir="data/raw")
    feature_engine = TennisFeatureEngine()
    model_trainer = ModelTrainer()
    
    # Load and preprocess data
    matches = data_loader.load_raw_data()
    
    # Load updated ratings instead of rebuilding from scratch
    import glob
    import os
    
    # Find the most recent ratings file (should be the 2025 updated one)
    rating_files = glob.glob(f"{feature_engine.cache_dir}/ratings_*.pkl")
    if rating_files:
        # Sort by modification time to get the most recent
        rating_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_ratings_file = rating_files[0]
        
        print(f"Loading updated ratings from: {latest_ratings_file}")
        
        try:
            with open(latest_ratings_file, 'rb') as f:
                ratings_data = pickle.load(f)
            
            # Load all the rating systems
            feature_engine.player_elos.update(ratings_data['player_elos'])
            feature_engine.player_serve_ratings.update(ratings_data['player_serve_ratings'])
            feature_engine.player_return_ratings.update(ratings_data['player_return_ratings'])
            feature_engine.player_grass_elos.update(ratings_data['player_grass_elos'])
            feature_engine.player_grass_serve_ratings.update(ratings_data['player_grass_serve_ratings'])
            feature_engine.player_grass_return_ratings.update(ratings_data['player_grass_return_ratings'])
            feature_engine.player_last_10.update(ratings_data['player_last_10'])
            feature_engine.player_grass_last_10.update(ratings_data['player_grass_last_10'])
            
            print(f"✅ Loaded updated ratings for {len(feature_engine.player_elos)} players")
            print(f"✅ Loaded updated grass ratings for {len(feature_engine.player_grass_elos)} players")
            
        except Exception as e:
            print(f"Warning: Could not load updated ratings: {e}")
            print("Falling back to building features from scratch...")
            feature_engine.build_features(matches)
    else:
        print("No ratings files found, building features from scratch...")
        feature_engine.build_features(matches)
    
    # Load pre-trained models with correct paths (including RFSR ensemble)
    model_trainer.load_models(
        'models/grass_wimbledon_rf_model.pkl', 
        'models/grass_wimbledon_xgb_model.pkl',
        'models/RFSR_ensemble.pkl'
    )
    
    return PredictionInterface(feature_engine, model_trainer), matches

def convert_to_implied_probability(decimal_odds):
    """Convert decimal odds to implied probability."""
    if decimal_odds == 0:
        return 0
    return 1 / decimal_odds

def predict_match_with_ev(predictor, matches_df, player1, player2, odds_info, model='ensemble'):
    """
    Predict match outcome and calculate Expected Value (EV).
    """
    print(f"\n🎾 Predicting: {player1} vs {player2}")
    print("=" * 60)

    prediction = predictor.predict_match_outcome(player1, player2, matches_df, model=model)

    if not prediction:
        print("❌ Could not make prediction - check player names")
        return

    if 'rfsr_ensemble' not in prediction:
        print("❌ RFSR Ensemble model not found in prediction output.")
        return

    pred_data = prediction['rfsr_ensemble']
    p1_prob = pred_data['player1_win_prob']
    p2_prob = pred_data['player2_win_prob']

    # Get implied probabilities
    p1_implied = convert_to_implied_probability(odds_info['player1']['decimal'])
    p2_implied = convert_to_implied_probability(odds_info['player2']['decimal'])

    # Calculate EV
    ev1 = (p1_prob * (odds_info['player1']['decimal'] - 1)) - (1 - p1_prob)
    ev2 = (p2_prob * (odds_info['player2']['decimal'] - 1)) - (1 - p2_prob)

    print(f"  {player1}: {p1_prob:.1%} (My Odds) vs. {p1_implied:.1%} (Implied Odds)")
    print(f"  {player2}: {p2_prob:.1%} (My Odds) vs. {p2_implied:.1%} (Implied Odds)")

    if ev1 > 0.07:
        print(f"  🔥 Positive EV found for {player1}: {ev1:.2%}")
    if ev2 > 0.07:
        print(f"  🔥 Positive EV found for {player2}: {ev2:.2%}")

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize prediction engine
    logger.info("Initializing prediction engine...")
    predictor, matches = initialize_engine()
    
    print("\n" + "="*60)
    print("🎾 WIMBLEDON PREDICTION ENGINE")
    print("="*60)

    # Fetch odds
    odds_data = fetch_wimbledon_odds()

    if not odds_data:
        print("Could not fetch odds. Exiting.")
        return

    # Collect predictions for Excel export
    print("\n📊 Analyzing matches for positive EV...")
    for (p1, p2), odds in odds_data.items():
        predict_match_with_ev(predictor, matches, p1, p2, odds, model='ensemble')

    print("\n" + "="*60)
    print("🎾 EV ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()  

    