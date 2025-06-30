import logging
import pickle
import pandas as pd
from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
from prediction import PredictionInterface
from ensemble_models import RF_SR_Ensemble  # Import needed for unpickling
#from utils.odds_api import fetch_wimbledon_odds

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



def predict_match(predictor, matches_df, player1, player2, model='both'):
    """
    Predict match outcome between two players
    Args:
        predictor: PredictionInterface instance
        matches_df: DataFrame containing match data
        player1: Name of first player
        player2: Name of second player
        model: 'both', 'xgboost', 'random_forest', 'rfsr', or 'ensemble'
    """
    print(f"\n🎾 Predicting: {player1} vs {player2}")
    print("=" * 60)
    
    prediction = predictor.predict_match_outcome(player1, player2, matches_df, model=model)
    
    if not prediction:
        print("❌ Could not make prediction - check player names")
        return
    
    
    # Display RFSR ensemble predictions
    if 'rfsr_ensemble' in prediction:
        print(f"\n🎯 RFSR Ensemble Prediction:")
        print(f"  {player1}: {prediction['rfsr_ensemble']['player1_win_prob']:.1%}")
        print(f"  {player2}: {prediction['rfsr_ensemble']['player2_win_prob']:.1%}")
        print(f"  Weights: {prediction['rfsr_ensemble']['ensemble_weights']}")
        
        # Display betting odds
        # odds = prediction['rfsr_ensemble']['betting_odds']
        # print(f"\n💰 RFSR Ensemble Betting Odds:")
        # print(f"  {player1}: {odds['player1']['american']:+d}")
        # print(f"  {player2}: {odds['player2']['american']:+d}")
    
    # Display key features
    print(f"\n📈 Key Feature Differences ({player1} - {player2}):")
    features = prediction['feature_differences']
    print(f"  Elo rating: {features['elo_diff']:.1f}")
    print(f"  Grass Elo: {features['elo_grass_diff']:.1f}")
    print(f"  Serve rating: {features['serve_rating_diff']:.1f}")
    print(f"  Return rating: {features['return_rating_diff']:.1f}")
    print(f"  Recent form: {features['last_10_diff']:.2f}")

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
    
    # List of matches to predict
    match_predictions = [
        ("Adrian Mannarino", "Christopher Connell"),
        ("Leandro Riedi", "Oliver Tarvet"),
        ("Mattia Bellucci", "Oliver Crawford"),
        ("Stefanos Tsitsipas", "Valentin Royer"),
        ("Felix Auger Aliassime", "James Duckworth"),
        ("Filip Misolic", "Jan Lennard Struff"),
        ("Karen Khachanov", "Mackenzie McDonald"),
        ("Botic van de Zandschulp", "Matteo Arnaldi"),
        ("Aleksandar Vukic", "Chun-Hsin Tseng"),
        ("Alex de Minaur", "Roberto Carballes Baena"),
        ("Fabian Marozsan", "James McCabe"),
        ("Jack Pinnington Jones", "Tomas Martin Etcheverry"),
        ("Johannus Monday", "Tommy Paul")
    ]
    
    # Collect predictions for Excel export
    print("\n📊 Printing predictions for each match...")
    for player1, player2 in match_predictions:
        print(f"\n🔄 Processing: {player1} vs {player2}")
        predict_match(predictor, matches, player1, player2, model='ensemble')
    print("\n" + "="*60)
    print("🎾 PREDICTION PRINTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()  

    