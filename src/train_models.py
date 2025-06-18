import logging
import pickle
from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from train_models import ModelTrainer

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize components
    data_loader = TennisDataLoader()
    feature_engine = TennisFeatureEngine()
    model_trainer = ModelTrainer()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    matches = data_loader.load_raw_data()
    grass_matches, non_grass_matches = data_loader.preprocess_data(matches)
    
    # Build features with corrected logic
    logger.info("Building features with corrected logic...")
    feature_engine.build_features(matches)
    
    # Save the corrected features
    logger.info("Saving corrected features...")
    feature_engine.save_features('corrected_train_features.pkl', 'corrected_test_features.pkl')
    
    # Prepare training data
    logger.info("Preparing training data...")
    X_train, y_train, X_test, y_test = model_trainer.prepare_training_data(
        feature_engine.features['train'], 
        feature_engine.features['test']
    )
    
    # Train models
    logger.info("Training models with corrected features...")
    model_trainer.train_models(X_train, y_train)
    
    # Evaluate models
    logger.info("Evaluating models...")
    metrics = model_trainer.evaluate_models(X_test, y_test)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION WITH CORRECTED FEATURES")
    print("="*50)
    
    print("\nRandomForest Results:")
    print(f"Accuracy: {metrics['random_forest']['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics['random_forest']['auc_roc']:.4f}")
    print(f"Log Loss: {metrics['random_forest']['log_loss']:.4f}")
    
    print("\nXGBoost Results:")
    print(f"Accuracy: {metrics['xgboost']['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics['xgboost']['auc_roc']:.4f}")
    print(f"Log Loss: {metrics['xgboost']['log_loss']:.4f}")
    
    # Show feature importance
    importance = model_trainer.get_feature_importance()
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)
    print("\nRandomForest Feature Importance:")
    print(importance['random_forest'])
    print("\nXGBoost Feature Importance:")
    print(importance['xgboost'])
    
    # Save corrected models
    logger.info("Saving corrected models...")
    model_trainer.save_models('corrected_wimbledon_rf_model.pkl', 'corrected_wimbledon_xgb_model.pkl')
    
    print("\n" + "="*50)
    print("CORRECTED MODELS SAVED!")
    print("="*50)
    print("✅ Models retrained with bias-free feature engineering")
    print("✅ Use these models for unbiased predictions")

if __name__ == "__main__":
    main() 