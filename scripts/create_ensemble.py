import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
from ensemble_models import RF_SR_Ensemble
import logging
import joblib

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize components
    data_loader = TennisDataLoader()
    feature_engine = TennisFeatureEngine()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    matches = data_loader.load_raw_data()
    
    # Build features
    logger.info("Building features...")
    feature_engine.build_features(matches)
    
    # Prepare training data
    logger.info("Preparing training data...")
    X_train, y_train, X_test, y_test = ModelTrainer().prepare_training_data(
        feature_engine.features['train'], 
        feature_engine.features['test']
    )
    
    # Create and train ensemble
    logger.info("Training RF+SR ensemble...")
    ensemble = RF_SR_Ensemble(rf_weight=0.75, sr_weight=0.25)
    ensemble.fit(X_train, y_train)
    
    # Evaluate ensemble
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    y_pred = ensemble.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba)
    }
    
    # Print results
    print("\n" + "="*50)
    print("RF+SR ENSEMBLE PERFORMANCE")
    print("="*50)
    print(f"\nWeights: RF={ensemble.rf_weight:.2f}, SR={ensemble.sr_weight:.2f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    
    # Save ensemble model
    logger.info("Saving ensemble model...")
    joblib.dump(ensemble, 'models/RFSR_ensemble.pkl')
    print("\n✅ Ensemble model saved as 'models/RFSR_ensemble.pkl'")

if __name__ == "__main__":
    main() 