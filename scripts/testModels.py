import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def evaluate_ensemble(y_true, y_pred_proba, y_pred):
    """Evaluate ensemble predictions"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba)
    }

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
    
    # Build features
    logger.info("Building features...")
    feature_engine.build_features(matches)
    
    # Prepare training data
    logger.info("Preparing training data...")
    X_train, y_train, X_test, y_test = model_trainer.prepare_training_data(
        feature_engine.features['train'], 
        feature_engine.features['test']
    )
    
    # Train models
    logger.info("Training models...")
    model_trainer.train_models(X_train, y_train)
    
    # Get predictions from all models
    logger.info("Getting predictions from all models...")
    rf_pred_proba = model_trainer.rf_model.predict_proba(X_test)[:, 1]
    xgb_pred_proba = model_trainer.xgb_model.predict_proba(X_test)[:, 1]
    elo_pred_proba = model_trainer.elo_model.predict_proba(X_test)
    sr_pred_proba = model_trainer.sr_model.predict_proba(X_test)
    
    # Create DataFrame with all predictions
    probs_df = pd.DataFrame({
        'random_forest': rf_pred_proba,
        'xgboost': xgb_pred_proba,
        'elo': elo_pred_proba,
        'sr': sr_pred_proba
    })
    
    # Calculate correlation matrix
    correlation_matrix = probs_df.corr()
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PREDICTION CORRELATION ANALYSIS")
    print("="*50)
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Print model performance metrics
    metrics = model_trainer.evaluate_models(X_test, y_test)
    print("\nModel Performance Metrics:")
    print("-"*30)
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.upper()}:")
        for metric_name, value in model_metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Analyze correlations and performance
    print("\nAnalysis:")
    print("-"*30)
    for model1 in probs_df.columns:
        for model2 in probs_df.columns:
            if model1 < model2:  # Avoid duplicate comparisons
                corr = correlation_matrix.loc[model1, model2]
                perf1 = metrics[model1]['auc_roc']
                perf2 = metrics[model2]['auc_roc']
                
                print(f"\n{model1} vs {model2}:")
                print(f"Correlation: {corr:.4f}")
                print(f"Performance: {model1} (AUC-ROC: {perf1:.4f}) vs {model2} (AUC-ROC: {perf2:.4f})")
                
                if corr > 0.8:
                    print("⚠️ High correlation detected!")
                    if perf1 > perf2:
                        print(f"Recommendation: Consider dropping {model2} as {model1} performs better")
                    else:
                        print(f"Recommendation: Consider dropping {model1} as {model2} performs better")
                elif corr < 0.3:
                    print("✅ Low correlation - good for ensemble diversity!")
    
    # Ensemble Analysis
    print("\n" + "="*50)
    print("ENSEMBLE ANALYSIS")
    print("="*50)
    
    # 1. Simple Weighted Average Ensembles
    print("\n1. Simple Weighted Average Ensembles:")
    print("-"*30)
    
    # Test different weight combinations
    weight_combinations = [
        (0.75, 0.25),  # RF heavy
        (0.5, 0.5),    # Equal weights
        (0.25, 0.75),  # SR heavy
    ]
    
    for rf_weight, sr_weight in weight_combinations:
        ensemble_probs = rf_weight * rf_pred_proba + sr_weight * sr_pred_proba
        ensemble_preds = (ensemble_probs > 0.5).astype(int)
        
        metrics = evaluate_ensemble(y_test, ensemble_probs, ensemble_preds)
        print(f"\nRF:{rf_weight:.2f} + SR:{sr_weight:.2f} Ensemble:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"Log Loss: {metrics['log_loss']:.4f}")
    
    # 2. Logistic Regression Meta-model
    print("\n2. Logistic Regression Meta-model:")
    print("-"*30)
    
    # Prepare meta-features
    meta_features_train = np.column_stack([
        model_trainer.rf_model.predict_proba(X_train)[:, 1],
        model_trainer.sr_model.predict_proba(X_train)
    ])
    
    meta_features_test = np.column_stack([
        rf_pred_proba,
        sr_pred_proba
    ])
    
    # Train meta-model
    meta_model = LogisticRegression()
    meta_model.fit(meta_features_train, y_train)
    
    # Get meta-model predictions
    meta_probs = meta_model.predict_proba(meta_features_test)[:, 1]
    meta_preds = (meta_probs > 0.5).astype(int)
    
    # Evaluate meta-model
    metrics = evaluate_ensemble(y_test, meta_probs, meta_preds)
    print("\nLogistic Regression Meta-model:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Meta-model weights: RF={meta_model.coef_[0][0]:.4f}, SR={meta_model.coef_[0][1]:.4f}")

if __name__ == "__main__":
    main()
