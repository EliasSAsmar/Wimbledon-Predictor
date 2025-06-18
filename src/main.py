import logging
import pickle
import pandas as pd
from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
from prediction import PredictionInterface
from ensemble_models import RF_SR_Ensemble  # Import needed for unpickling

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

def export_predictions_to_excel(predictions_list, filename="tennis_predictions.xlsx"):
    """
    Export predictions to Excel file
    """
    if not predictions_list:
        print("❌ No predictions to export")
        return
    
    # Create DataFrame
    df = pd.DataFrame(predictions_list)
    
    # Export to Excel
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Tennis Predictions', index=False)
        
        print(f"✅ Successfully exported {len(predictions_list)} predictions to {filename}")
        print(f"📊 File saved with columns: {', '.join(df.columns)}")
        
    except Exception as e:
        print(f"❌ Error exporting to Excel: {e}")

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
    
    # Display predictions
    if 'xgboost' in prediction:
        print(f"\n🤖 XGBoost Prediction:")
        print(f"  {player1}: {prediction['xgboost']['player1_win_prob']:.1%}")
        print(f"  {player2}: {prediction['xgboost']['player2_win_prob']:.1%}")
        
        # Display betting odds
        odds = prediction['xgboost']['betting_odds']
        print(f"\n💰 XGBoost Betting Odds:")
        print(f"  {player1}: {odds['player1']['american']:+d}")
        print(f"  {player2}: {odds['player2']['american']:+d}")
    
    if 'random_forest' in prediction:
        print(f"\n🌲 RandomForest Prediction:")
        print(f"  {player1}: {prediction['random_forest']['player1_win_prob']:.1%}")
        print(f"  {player2}: {prediction['random_forest']['player2_win_prob']:.1%}")
        
        # Display betting odds
        odds = prediction['random_forest']['betting_odds']
        print(f"\n💰 RandomForest Betting Odds:")
        print(f"  {player1}: {odds['player1']['american']:+d}")
        print(f"  {player2}: {odds['player2']['american']:+d}")
    
    # Display RFSR ensemble predictions
    if 'rfsr_ensemble' in prediction:
        print(f"\n🎯 RFSR Ensemble Prediction:")
        print(f"  {player1}: {prediction['rfsr_ensemble']['player1_win_prob']:.1%}")
        print(f"  {player2}: {prediction['rfsr_ensemble']['player2_win_prob']:.1%}")
        print(f"  Weights: {prediction['rfsr_ensemble']['ensemble_weights']}")
        
        # Display betting odds
        odds = prediction['rfsr_ensemble']['betting_odds']
        print(f"\n💰 RFSR Ensemble Betting Odds:")
        print(f"  {player1}: {odds['player1']['american']:+d}")
        print(f"  {player2}: {odds['player2']['american']:+d}")
    
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
        ("Tomas Machac", "Jesper de Jong"),
        ("Andrey Rublev", "Sebastian Ofner"),
        ("Tomas Martin Etcheverry", "Pedro Martinez"),
        ("Flavio Cobolli", "Joao Fonseca"),
        ("Alex Michelsen", "Francisco Cerundolo"),
        ("Denis Shapovalov", "Ugo Humbert"),
        ("Jannik Sinner", "Yannick Hanfmann"),
        ("Felix Auger-Aliassime", "Laslo Djere"),
        ("Lorenzo Sonego", "Jan-Lennard Struff"),
        ("Alexei Popyrin", "Aleksandar Vukic"),
        ("Brandon Nakashima", "Giovanni Mpetshi Perricard"),
        ("Jiri Lehecka", "Alex de Minaur"),
        ("Jack Draper", "Jenson Brooksby"),
        ("Arthur Rinderknech", "Ben Shelton"),
        ("Reilly Opelka", "Camilo Ugo Carabelli"),
        ("Carlos Alcaraz", "Adam Walton"),
        ("Jaume Munar", "Jordan Thompson"),
        ("Corentin Moutet", "Taylor Fritz"),
        ("Gabriel Diallo", "Billy Harris")
    ]
    
    # Collect predictions for Excel export
    print("\n📊 Collecting predictions for Excel export...")
    predictions_for_excel = []
    
    for player1, player2 in match_predictions:
        print(f"🔄 Processing: {player1} vs {player2}")
        
        # Collect data for Excel
        prediction_data = collect_prediction_data(predictor, matches, player1, player2, model='ensemble')
        if prediction_data:
            predictions_for_excel.append(prediction_data)
            print(f"  ✅ Data collected successfully")
        else:
            print(f"  ❌ Could not collect data - player names may not be found")
    
    # Export to Excel
    print(f"\n📋 Exporting {len(predictions_for_excel)} successful predictions to Excel...")
    export_predictions_to_excel(predictions_for_excel, "wimbledon_predictions_2024.xlsx")
    
    print("\n" + "="*60)
    print("🎾 PREDICTION EXPORT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main() 