import logging
import pickle
import pandas as pd
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
from prediction import PredictionInterface
from ensemble_models import RF_SR_Ensemble  # Import needed for unpickling
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

def collect_prediction_data(predictor, matches_df, player1, player2, model='ensemble'):
    """
    Collect match prediction data for Excel export
    Returns a dictionary with prediction data or None if prediction fails
    """
    prediction = predictor.predict_match_outcome(player1, player2, matches_df, model=model)
    
    if not prediction:
        return None
    
    # Extract RFSR ensemble data (preferred model)
    if 'rfsr_ensemble' in prediction:
        pred_data = prediction['rfsr_ensemble']
        model_used = 'RFSR Ensemble'
    elif 'random_forest' in prediction:
        pred_data = prediction['random_forest']
        model_used = 'Random Forest'
    elif 'xgboost' in prediction:
        pred_data = prediction['xgboost']
        model_used = 'XGBoost'
    else:
        return None
    
    # Extract betting odds
    odds = pred_data['betting_odds']
    
    return {
        'Player_1': player1,
        'Player_1_Win_Probability': f"{pred_data['player1_win_prob']:.1%}",
        'Player_1_American_Odds': odds['player1']['american'],
        'Player_2': player2,
        'Player_2_Win_Probability': f"{pred_data['player2_win_prob']:.1%}",
        'Player_2_American_Odds': odds['player2']['american'],
        'Model_Used': model_used,
        'Confidence_Split': f"{pred_data['player1_win_prob']:.1%} - {pred_data['player2_win_prob']:.1%}",
        'Elo_Difference': f"{prediction['feature_differences']['elo_diff']:.1f}",
        'Grass_Elo_Difference': f"{prediction['feature_differences']['elo_grass_diff']:.1f}",
        'Serve_Rating_Difference': f"{prediction['feature_differences']['serve_rating_diff']:.1f}",
        'Return_Rating_Difference': f"{prediction['feature_differences']['return_rating_diff']:.1f}",
        'Recent_Form_Difference': f"{prediction['feature_differences']['last_10_diff']:.2f}"
    }

def predict_match_with_odds(predictor, matches_df, player1, player2, vegas_odds=None, model='ensemble'):
    """
    Predict match outcome between two players and display with Vegas odds
    Args:
        predictor: PredictionInterface instance
        matches_df: DataFrame containing match data
        player1: Name of first player
        player2: Name of second player
        vegas_odds: Dict containing Vegas odds for both players
        model: 'both', 'xgboost', 'random_forest', 'rfsr', or 'ensemble'
    """
    print(f"\n🎾 {player1} vs {player2}")
    print("-" * 50)
    
    prediction = predictor.predict_match_outcome(player1, player2, matches_df, model=model)
    
    if not prediction:
        print("❌ Could not make prediction - check player names")
        return
    
    # Display RFSR ensemble predictions (our preferred model)
    if 'rfsr_ensemble' in prediction:
        model_pred = prediction['rfsr_ensemble']
        print(f"🎯 Model Predictions:")
        print(f"  {player1}: {model_pred['player1_win_prob']:.1%}")
        print(f"  {player2}: {model_pred['player2_win_prob']:.1%}")
    
    # Display Vegas odds if available
    if vegas_odds:
        print(f"💰 Vegas Odds:")
        # Convert American odds to implied probability
        def american_to_implied_prob(american_odds):
            if american_odds > 0:
                return 100 / (american_odds + 100) * 100
            else:
                return abs(american_odds) / (abs(american_odds) + 100) * 100
        
        p1_odds = vegas_odds['player1']['decimal']
        p2_odds = vegas_odds['player2']['decimal']
        p1_implied = american_to_implied_prob(p1_odds)
        p2_implied = american_to_implied_prob(p2_odds)
        
        print(f"  {vegas_odds['player1']['name']}: {p1_implied:.1f}% ({p1_odds:+.0f})")
        print(f"  {vegas_odds['player2']['name']}: {p2_implied:.1f}% ({p2_odds:+.0f})")

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
    
    # Fetch current Wimbledon odds from API
    print("\n📡 Fetching current Wimbledon matches from odds API...")
    odds_data = fetch_wimbledon_odds()
    
    if not odds_data:
        print("❌ No odds data available. Check API connection.")
        return
    
    print(f"✅ Found {len(odds_data)} matches with odds")
    
    # Make predictions for all matches
    print("\n📊 Making predictions for all matches...")
    
    for (home_team, away_team), odds_info in odds_data.items():
        # Extract player names (odds API returns team names, but for tennis it's player names)
        player1 = odds_info['player1']['name']
        player2 = odds_info['player2']['name']
        
        predict_match_with_odds(predictor, matches, player1, player2, odds_info, model='ensemble')
    
    print("\n" + "="*60)
    print("🎾 ALL PREDICTIONS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()  

    