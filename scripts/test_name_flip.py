import sys
sys.path.append('src')

from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
from prediction import PredictionInterface

def test_name_flipping():
    print("Testing name flipping with FIXED prediction system...")
    print("="*60)
    
    # Initialize components
    data_loader = TennisDataLoader()
    feature_engine = TennisFeatureEngine()
    model_trainer = ModelTrainer()
    
    # Load data and build features
    print("Loading data and building features...")
    matches = data_loader.load_raw_data()
    feature_engine.build_features(matches)
    
    # Load corrected models
    model_trainer.load_models('corrected_wimbledon_rf_model.pkl', 'corrected_wimbledon_xgb_model.pkl')
    
    # Initialize prediction interface
    predictor = PredictionInterface(feature_engine, model_trainer)
    
    # Test both orders
    print("\nTesting: Ben Shelton vs Jiri Lehecka")
    prediction1 = predictor.predict_match_outcome("Ben Shelton", "Jiri Lehecka", matches, model='xgb')
    
    print("Testing: Jiri Lehecka vs Ben Shelton")
    prediction2 = predictor.predict_match_outcome("Jiri Lehecka", "Ben Shelton", matches, model='xgb')
    
    if prediction1 and prediction2:
        print("\n" + "="*60)
        print("RESULTS COMPARISON:")
        print("="*60)
        
        ben_prob_1 = prediction1['xgboost']['player1_win_prob']
        jiri_prob_1 = prediction1['xgboost']['player2_win_prob']
        
        jiri_prob_2 = prediction2['xgboost']['player1_win_prob']
        ben_prob_2 = prediction2['xgboost']['player2_win_prob']
        
        print(f"Order 1 (Ben vs Jiri):")
        print(f"  Ben: {ben_prob_1:.1%}, Jiri: {jiri_prob_1:.1%}")
        
        print(f"Order 2 (Jiri vs Ben):")
        print(f"  Jiri: {jiri_prob_2:.1%}, Ben: {ben_prob_2:.1%}")
        
        print(f"\nConsistency Check:")
        print(f"Ben's win probability:")
        print(f"  Order 1: {ben_prob_1:.1%}")
        print(f"  Order 2: {ben_prob_2:.1%}")
        print(f"  Difference: {abs(ben_prob_1 - ben_prob_2):.6f}")
        
        print(f"Jiri's win probability:")
        print(f"  Order 1: {jiri_prob_1:.1%}")
        print(f"  Order 2: {jiri_prob_2:.1%}")
        print(f"  Difference: {abs(jiri_prob_1 - jiri_prob_2):.6f}")
        
        # Check if results are consistent
        ben_consistent = abs(ben_prob_1 - ben_prob_2) < 0.001
        jiri_consistent = abs(jiri_prob_1 - jiri_prob_2) < 0.001
        
        print(f"\n{'✅ FIXED!' if ben_consistent and jiri_consistent else '❌ STILL BIASED!'}")
        
        if ben_consistent and jiri_consistent:
            print("The prediction system is now order-independent!")
            print("No matter which name comes first, each player gets the same win probability.")
        else:
            print("There's still bias in the system!")

if __name__ == "__main__":
    test_name_flipping() 