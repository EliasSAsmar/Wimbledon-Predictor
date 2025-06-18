#!/usr/bin/env python3
"""
Quick bias test to verify the averaging trick eliminates naming bias
"""
import sys
sys.path.append('src')

from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
from prediction import PredictionInterface
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

def test_bias_elimination():
    """Test that player name order doesn't affect predictions"""
    print("🧪 Testing Model Averaging...")
    
    # Initialize system
    data_loader = TennisDataLoader()
    feature_engine = TennisFeatureEngine()
    model_trainer = ModelTrainer()
    
    matches = data_loader.load_raw_data()
    feature_engine.build_features(matches)
    model_trainer.load_models('corrected_wimbledon_rf_model.pkl', 'corrected_wimbledon_xgb_model.pkl')
    
    predictor = PredictionInterface(feature_engine, model_trainer)
    
    # Test cases
    test_pairs = [
        ("Ben Shelton", "Jiri Lehecka"),
        ("Taylor Fritz", "Alexander Zverev"),
        ("Ben Shelton", "Alexander Zverev")
    ]
    
    print("\n" + "="*80)
    print("BIAS TEST RESULTS")
    print("="*80)
    
    for player1, player2 in test_pairs:
        print(f"\n🎾 Testing: {player1} vs {player2}")
        
        # Predict both orientations
        pred1 = predictor.predict_match_outcome(player1, player2, matches, model='xgb')
        pred2 = predictor.predict_match_outcome(player2, player1, matches, model='xgb')
        
        if pred1 and pred2:
            # Check if probabilities are complementary
            prob1_wins_in_first = pred1['xgboost']['player1_win_prob']
            prob1_wins_in_second = pred2['xgboost']['player2_win_prob']
            
            bias = abs(prob1_wins_in_first - prob1_wins_in_second)
            
            print(f"  {player1} win prob (as player1): {prob1_wins_in_first:.3f}")
            print(f"  {player1} win prob (as player2): {prob1_wins_in_second:.3f}")
            print(f"  Bias: {bias:.6f}")
            
            if bias < 0.001:
                print("  ✅ PASS - No bias detected")
            else:
                print("  ❌ FAIL - Bias detected!")
        else:
            print("  ⚠️  Could not test - player not found")
    
    print("\n" + "="*80)
    print("✅ Bias test complete!")
    print("Bias should be < 0.001 for proper averaging trick implementation")
    print("="*80)

if __name__ == "__main__":
    test_bias_elimination() 