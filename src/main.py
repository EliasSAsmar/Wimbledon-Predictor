import logging
import pickle
from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
from prediction import PredictionInterface

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def initialize_engine():
    """Initialize and return the prediction engine components and matches data"""
    data_loader = TennisDataLoader()
    feature_engine = TennisFeatureEngine()
    model_trainer = ModelTrainer()
    
    # Load and preprocess data
    matches = data_loader.load_raw_data()
    feature_engine.build_features(matches)
    
    # Load pre-trained models
    model_trainer.load_models('corrected_wimbledon_rf_model.pkl', 'corrected_wimbledon_xgb_model.pkl')
    
    return PredictionInterface(feature_engine, model_trainer), matches

def predict_match(predictor, matches_df, player1, player2, model='both'):
    """
    Predict match outcome between two players
    Args:
        predictor: PredictionInterface instance
        matches_df: DataFrame containing match data
        player1: Name of first player
        player2: Name of second player
        model: 'both', 'xgboost', or 'random_forest'
    """
    print(f"\n🎾 Predicting: {player1} vs {player2}")
    print("=" * 60)
    
    prediction = predictor.predict_match_outcome(player1, player2, matches_df, model=model)
    
    if not prediction:
        print("❌ Could not make prediction - check player names")
        return
    
    # Display predictions
    if 'xgboost' in prediction:
        print(f"\n🤖 XGBoost Prediction:")
        print(f"  {player1}: {prediction['xgboost']['player1_win_prob']:.1%}")
        print(f"  {player2}: {prediction['xgboost']['player2_win_prob']:.1%}")
        
        # Display betting odds
        odds = prediction['xgboost']['betting_odds']
        print(f"\n💰 XGBoost Betting Odds (7% vig):")
        print(f"  {player1}:")
        print(f"    American: {odds['player1']['american']:+d}")
        print(f"    Decimal:  {odds['player1']['decimal']}")
        print(f"    Fractional: {odds['player1']['fractional']}")
        print(f"  {player2}:")
        print(f"    American: {odds['player2']['american']:+d}")
        print(f"    Decimal:  {odds['player2']['decimal']}")
        print(f"    Fractional: {odds['player2']['fractional']}")
        print(f"  📊 Total Implied Probability: {odds['total_implied_prob']}")
    
    if 'random_forest' in prediction:
        print(f"\n🌲 RandomForest Prediction:")
        print(f"  {player1}: {prediction['random_forest']['player1_win_prob']:.1%}")
        print(f"  {player2}: {prediction['random_forest']['player2_win_prob']:.1%}")
        
        # Display betting odds
        odds = prediction['random_forest']['betting_odds']
        print(f"\n💰 RandomForest Betting Odds (7% vig):")
        print(f"  {player1}:")
        print(f"    American: {odds['player1']['american']:+d}")
        print(f"    Decimal:  {odds['player1']['decimal']}")
        print(f"    Fractional: {odds['player1']['fractional']}")
        print(f"  {player2}:")
        print(f"    American: {odds['player2']['american']:+d}")
        print(f"    Decimal:  {odds['player2']['decimal']}")
        print(f"    Fractional: {odds['player2']['fractional']}")
        print(f"  📊 Total Implied Probability: {odds['total_implied_prob']}")
    
    # Display key features
    print(f"\n📈 Key Feature Differences ({player1} - {player2}):")
    features = prediction['feature_differences']
    print(f"  Elo rating: {features['elo_diff']:.1f}")
    print(f"  Grass Elo: {features['elo_grass_diff']:.1f}")
    print(f"  Serve rating: {features['serve_rating_diff']:.1f}")
    print(f"  Return rating: {features['return_rating_diff']:.1f}")
    print(f"  Recent form: {features['last_10_diff']:.2f}")
    print(f"  Ranking: {-features['rank_diff']:.0f}")

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
    
    # Example predictions
    predict_match(predictor, matches, "Ben Shelton", "Jiri Lehecka")
    predict_match(predictor, matches, "Taylor Fritz", "Fucsovics")
    predict_match(predictor, matches, "Ben Shelton", "Alexander Zverev")
    predict_match(predictor, matches, "Taylor Fritz", "Felix Auger Aliassime")
    


if __name__ == "__main__":
    main() 