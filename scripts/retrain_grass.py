import pandas as pd
import logging
from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize components
    logger.info("Initializing components...")
    data_loader = TennisDataLoader(data_dir="data/raw")
    feature_engine = TennisFeatureEngine()
    model_trainer = ModelTrainer()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    matches = data_loader.load_raw_data()
    
    # Build features
    logger.info("Building features...")
    feature_engine.build_features(matches)
    
    # Prepare training data
    logger.info("Preparing training data...")
    X_train, y_train, X_test, y_test = model_trainer.prepare_training_data(
        feature_engine.features['train'],
        feature_engine.features['test']
    )
    
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Train models
    logger.info("Training models...")
    model_trainer.train_models(X_train, y_train)
    
    # Evaluate models
    logger.info("Evaluating models...")
    metrics = model_trainer.evaluate_models(X_test, y_test)
    
    # Print results
    logger.info("\nModel Performance:")
    logger.info("==================")
    for model_name, model_metrics in metrics.items():
        logger.info(f"\n{model_name.upper()}:")
        for metric_name, value in model_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
    
    # Save models
    logger.info("\nSaving models...")
    model_trainer.save_models(
        rf_path='models/grass_wimbledon_rf_model.pkl',
        xgb_path='models/grass_wimbledon_xgb_model.pkl'
    )
    
    # Get feature importance
    importance = model_trainer.get_feature_importance()
    logger.info("\nFeature Importance:")
    logger.info("==================")
    for model_name, importance_df in importance.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(importance_df.to_string())

if __name__ == "__main__":
    main() 