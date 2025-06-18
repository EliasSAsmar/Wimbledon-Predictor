import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb
import joblib
import logging
from typing import Dict, Tuple, List

from helpers.ServeReturnModelHelper import ServeReturnModelHelper
from helpers.EloLogisticModelelper import EloLogisticModelHelper

class ModelTrainer:
    """Handles model training, evaluation, and persistence"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rf_model = None
        self.xgb_model = None
        self.rfsr_ensemble = None  # Add RFSR ensemble model
        #self.bayes_model = BayesianModelHelper()     
        #self.voting_model = None

        self.elo_model = EloLogisticModelHelper()
        self.sr_model = ServeReturnModelHelper()
        
        self.feature_columns = [
            'elo_diff',                    # General Elo difference
            'elo_grass_diff',              # Grass-specific Elo difference  
            'serve_rating_diff',           # General serve rating difference
            'serve_rating_grass_diff',     # Grass-specific serve rating difference
            'return_rating_diff',          # General return rating difference
            'return_rating_grass_diff',    # Grass-specific return rating difference
            'grass_winrate_last10_diff',   # Grass win rate last 10 matches
            'rank_diff',                   # ATP ranking difference
            'seed_diff',                   # Tournament seeding difference
            'age_diff',                    # Age difference
            'last_10_diff'                 # Overall recent form difference
        ]
    
    def create_binary_dataset(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary classification dataset with flipping
        Each match becomes 2 data points:
        1. Winner vs Loser (target = 1) 
        2. Loser vs Winner (target = 0, features flipped)
        """
        binary_data = []
        
        for _, row in features_df.iterrows():
            # Original match: Winner vs Loser (target = 1)
            original_features = {}
            for col in self.feature_columns:
                original_features[col] = row[col]
            
            binary_data.append({
                **original_features,
                'target': 1,  # Winner wins
                'match_id': row['match_id'],
                'date': row['date'],
                'tournament': row['tournament'],
                'surface': row['surface'],
                'player_1_id': row['winner_id'],
                'player_2_id': row['loser_id']
            })
            
            # Flipped match: Loser vs Winner (target = 0)
            flipped_features = {}
            for col in self.feature_columns:
                # Flip the sign for difference features
                flipped_features[col] = -row[col]
            
            binary_data.append({
                **flipped_features,
                'target': 0,  # Loser loses 
                'match_id': row['match_id'] + '_flipped',
                'date': row['date'],
                'tournament': row['tournament'],
                'surface': row['surface'],
                'player_1_id': row['loser_id'],
                'player_2_id': row['winner_id']
            })
        
        return pd.DataFrame(binary_data)
    
    def prepare_training_data(self, train_features: Dict, test_features: Dict) -> Tuple:
        """Prepare training and test datasets"""
        # Convert to DataFrames
        train_rows = []
        for match_id, match_data in train_features.items():
            # Only include grass court matches
            if match_data['surface'] == 'Grass':
                train_rows.append({'match_id': match_id, **match_data})
        
        test_rows = []
        for match_id, match_data in test_features.items():
            # Only include grass court matches
            if match_data['surface'] == 'Grass':
                test_rows.append({'match_id': match_id, **match_data})
        
        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows)
        
        # Create binary datasets
        train_binary = self.create_binary_dataset(train_df)
        test_binary = self.create_binary_dataset(test_df)
        
        # Prepare features and targets
        X_train = train_binary[self.feature_columns]
        y_train = train_binary['target']
        
        X_test = test_binary[self.feature_columns]
        y_test = test_binary['target']
        
        return X_train, y_train, X_test, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train both RandomForest and XGBoost models"""
        # Train RandomForest
        self.logger.info("Training RandomForest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # Train XGBoost
        self.logger.info("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.xgb_model.fit(X_train, y_train)

        self.elo_model.train(X_train, y_train)
        self.sr_model.train(X_train, y_train)

    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate both models and return metrics"""
        # Get predictions
        rf_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
        rf_pred = self.rf_model.predict(X_test)
        
        xgb_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = self.xgb_model.predict(X_test)
        
        elo_pred_proba = self.elo_model.predict_proba(X_test)
        elo_pred = self.elo_model.predict(X_test)
        
        sr_pred_proba = self.sr_model.predict_proba(X_test)
        sr_pred = self.sr_model.predict(X_test)
        

        # Calculate metrics
        metrics = {
            'random_forest': {
                'accuracy': accuracy_score(y_test, rf_pred),
                'auc_roc': roc_auc_score(y_test, rf_pred_proba),
                'log_loss': log_loss(y_test, rf_pred_proba)
            },
            'xgboost': {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'auc_roc': roc_auc_score(y_test, xgb_pred_proba),
                'log_loss': log_loss(y_test, xgb_pred_proba)
            },
            'elo': {
                'accuracy': accuracy_score(y_test, elo_pred),
                'auc_roc': roc_auc_score(y_test, elo_pred_proba),
                'log_loss': log_loss(y_test, elo_pred_proba)
            },
            'sr': {
                'accuracy': accuracy_score(y_test, sr_pred),   
                'auc_roc': roc_auc_score(y_test, sr_pred_proba),
                'log_loss': log_loss(y_test, sr_pred_proba)
            }
        }

        
        return metrics
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from both models"""
        rf_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        xgb_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'random_forest': rf_importance,
            'xgboost': xgb_importance
        }
    
    def save_models(self, rf_path: str = 'wimbledon_rf_model.pkl', 
                   xgb_path: str = 'wimbledon_xgb_model.pkl',
                   rfsr_path: str = 'RFSR_ensemble.pkl') -> None:
        """Save trained models to disk"""
        joblib.dump(self.rf_model, rf_path)
        joblib.dump(self.xgb_model, xgb_path)
        if self.rfsr_ensemble is not None:
            joblib.dump(self.rfsr_ensemble, rfsr_path)
        self.logger.info(f"Models saved to {rf_path}, {xgb_path}, and {rfsr_path}")
    
    def load_models(self, rf_path: str = 'wimbledon_rf_model.pkl', 
                   xgb_path: str = 'wimbledon_xgb_model.pkl',
                   rfsr_path: str = 'RFSR_ensemble.pkl') -> None:
        """Load trained models from disk"""
        self.rf_model = joblib.load(rf_path)
        self.xgb_model = joblib.load(xgb_path)
        try:
            self.rfsr_ensemble = joblib.load(rfsr_path)
            self.logger.info("Models loaded successfully (including RFSR ensemble)")
        except FileNotFoundError:
            self.logger.warning(f"RFSR ensemble not found at {rfsr_path}, skipping...")
            self.rfsr_ensemble = None
        except Exception as e:
            self.logger.warning(f"Error loading RFSR ensemble: {e}")
            self.rfsr_ensemble = None 