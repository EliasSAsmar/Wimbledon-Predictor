import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from model_train import ModelTrainer

class RF_SR_Ensemble(BaseEstimator, ClassifierMixin):
    """Ensemble model combining Random Forest and Serve/Return models"""
    
    def __init__(self, rf_weight=0.75, sr_weight=0.25):
        self.rf_weight = rf_weight
        self.sr_weight = sr_weight
        self.rf_model = None
        self.sr_model = None
        
    def fit(self, X, y):
        """Train the component models"""
        # Initialize and train models
        model_trainer = ModelTrainer()
        model_trainer.train_models(X, y)
        
        # Store the trained models
        self.rf_model = model_trainer.rf_model
        self.sr_model = model_trainer.sr_model
        return self
    
    def predict_proba(self, X):
        """Get probability predictions from the ensemble"""
        rf_probs = self.rf_model.predict_proba(X)[:, 1]
        sr_probs = self.sr_model.predict_proba(X)
        
        # Weighted average of probabilities
        ensemble_probs = (self.rf_weight * rf_probs + 
                         self.sr_weight * sr_probs)
        
        # Convert to 2D array format expected by sklearn
        return np.column_stack([1 - ensemble_probs, ensemble_probs])
    
    def predict(self, X):
        """Get class predictions from the ensemble"""
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int) 