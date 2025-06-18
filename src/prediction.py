import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

class PredictionInterface:
    """Interface for making tennis match predictions"""
    
    def __init__(self, feature_engine, model_trainer):
        self.logger = logging.getLogger(__name__)
        self.feature_engine = feature_engine
        self.model_trainer = model_trainer
    
    def get_player_features(self, player_name: str, matches_df: pd.DataFrame) -> Tuple[Dict, str]:
        """Get all relevant features for a player from our tracking dictionaries"""
        # First find the player ID
        player_matches = matches_df[
            (matches_df['winner_name'] == player_name) | 
            (matches_df['loser_name'] == player_name)
        ]
        
        if len(player_matches) == 0:
            raise ValueError(f"No matches found for {player_name}")
        
        # Get the player ID
        if len(player_matches[player_matches['winner_name'] == player_name]) > 0:
            player_id = player_matches[player_matches['winner_name'] == player_name].iloc[0]['winner_id']
        else:
            player_id = player_matches[player_matches['loser_name'] == player_name].iloc[0]['loser_id']
        
        # Get latest rank and seed if available
        latest_match = player_matches.sort_values('tourney_date').iloc[-1]
        rank = latest_match['winner_rank'] if latest_match['winner_name'] == player_name else latest_match['loser_rank']
        seed = latest_match['winner_seed'] if latest_match['winner_name'] == player_name else latest_match['loser_seed']
        
        # Get all features from our tracking
        features = {
            'elo': self.feature_engine.player_elos[player_id],
            'elo_grass': self.feature_engine.player_grass_elos[player_id],
            'serve_rating': self.feature_engine.player_serve_ratings[player_id],
            'serve_rating_grass': self.feature_engine.player_grass_serve_ratings[player_id],
            'return_rating': self.feature_engine.player_return_ratings[player_id],
            'return_rating_grass': self.feature_engine.player_grass_return_ratings[player_id],
            'grass_winrate_last10': sum(self.feature_engine.player_grass_last_10[player_id])/len(self.feature_engine.player_grass_last_10[player_id]) if self.feature_engine.player_grass_last_10[player_id] else 0.5,
            'rank': rank if not pd.isna(rank) else 999,
            'seed': seed if not pd.isna(seed) else 0,
            'last_10': sum(self.feature_engine.player_last_10[player_id])/len(self.feature_engine.player_last_10[player_id]) if self.feature_engine.player_last_10[player_id] else 0.5
        }
        
        return features, player_id
    
    def _calculate_feature_differences(self, player1_features: Dict, player2_features: Dict) -> Dict:
        """
        Calculate feature differences (player1 - player2) in the original format
        This matches the training data format exactly
        """
        return {
            'elo_diff': player1_features['elo'] - player2_features['elo'],
            'elo_grass_diff': player1_features['elo_grass'] - player2_features['elo_grass'],
            'serve_rating_diff': player1_features['serve_rating'] - player2_features['serve_rating'],
            'serve_rating_grass_diff': player1_features['serve_rating_grass'] - player2_features['serve_rating_grass'],
            'return_rating_diff': player1_features['return_rating'] - player2_features['return_rating'],
            'return_rating_grass_diff': player1_features['return_rating_grass'] - player2_features['return_rating_grass'],
            'grass_winrate_last10_diff': player1_features['grass_winrate_last10'] - player2_features['grass_winrate_last10'],
            'rank_diff': player2_features['rank'] - player1_features['rank'],  # Reverse as lower rank is better
            'seed_diff': player2_features['seed'] - player1_features['seed'],  # Reverse as lower seed is better
            'age_diff': 0,  # We don't have reliable age data
            'last_10_diff': player1_features['last_10'] - player2_features['last_10']
        }
    
    def _calculate_betting_odds(self, prob1: float, prob2: float, vig: float = 0.07) -> Dict:
        """
        Convert probabilities to betting odds with vig
        Args:
            prob1: Probability for player 1
            prob2: Probability for player 2  
            vig: Bookmaker's vig/juice (default 7%)
        Returns:
            Dictionary with various odds formats
        """
        # Apply vig by reducing probabilities proportionally
        total_prob = prob1 + prob2
        vig_factor = (1 + vig) / total_prob
        
        prob1_with_vig = prob1 * vig_factor
        prob2_with_vig = prob2 * vig_factor
        
        # Ensure probabilities don't exceed 1
        prob1_with_vig = min(prob1_with_vig, 0.99)
        prob2_with_vig = min(prob2_with_vig, 0.99)
        
        def prob_to_american_odds(prob):
            """Convert probability to American odds"""
            if prob >= 0.5:
                return int(-100 * prob / (1 - prob))
            else:
                return int(100 * (1 - prob) / prob)
        
        def prob_to_decimal_odds(prob):
            """Convert probability to decimal odds"""
            return round(1 / prob, 2)
        
        def prob_to_fractional_odds(prob):
            """Convert probability to fractional odds"""
            decimal = 1 / prob
            numerator = decimal - 1
            
            # Find simple fraction representation
            for denom in range(1, 21):
                num = round(numerator * denom)
                if abs(num/denom - numerator) < 0.01:
                    return f"{num}/{denom}"
            
            # Fallback to decimal if no simple fraction found
            return f"{numerator:.2f}/1"
        
        return {
            'player1': {
                'american': prob_to_american_odds(prob1_with_vig),
                'decimal': prob_to_decimal_odds(prob1_with_vig),
                'fractional': prob_to_fractional_odds(prob1_with_vig),
                'implied_prob': f"{prob1_with_vig:.1%}"
            },
            'player2': {
                'american': prob_to_american_odds(prob2_with_vig),
                'decimal': prob_to_decimal_odds(prob2_with_vig),
                'fractional': prob_to_fractional_odds(prob2_with_vig),
                'implied_prob': f"{prob2_with_vig:.1%}"
            },
            'vig_applied': f"{vig:.1%}",
            'total_implied_prob': f"{prob1_with_vig + prob2_with_vig:.1%}"
        }
    
    def predict_match_outcome(self, player1_name: str, player2_name: str, 
                            matches_df: pd.DataFrame, model: str = 'both') -> Optional[Dict]:
        """
        Predict match outcome using the AVERAGING  to eliminate naming bias:
        1. Calculate P(player1 beats player2)
        2. Calculate P(player2 beats player1) 
        3. Average: final_prob = (prob1 + (1 - prob2)) / 2
        
        This ensures zero bias regardless of player name order.
        """
        self.logger.info(f"Predicting match: {player1_name} vs {player2_name} (using averaging trick)")
        
        # Get features for both players
        try:
            player1_features, player1_id = self.get_player_features(player1_name, matches_df)
            player2_features, player2_id = self.get_player_features(player2_name, matches_df)
        except ValueError as e:
            self.logger.error(f"Error: {e}")
            return None
        
        # AVERAGING TRICK: Run both orientations
        
        # Orientation 1: Player1 vs Player2
        features_1v2 = self._calculate_feature_differences(player1_features, player2_features)
        X_pred_1v2 = pd.DataFrame([features_1v2])
        
        # Orientation 2: Player2 vs Player1  
        features_2v1 = self._calculate_feature_differences(player2_features, player1_features)
        X_pred_2v1 = pd.DataFrame([features_2v1])
        
        predictions = {}
        
        if model in ['xgb', 'both']:
            # Get probabilities for both orientations
            prob_1v2 = self.model_trainer.xgb_model.predict_proba(X_pred_1v2)[0, 1]  # P(player1 wins)
            prob_2v1 = self.model_trainer.xgb_model.predict_proba(X_pred_2v1)[0, 1]  # P(player2 wins)
            
            # Average the predictions: (prob1 + (1 - prob2)) / 2
            final_prob_player1 = (prob_1v2 + (1 - prob_2v1)) / 2
            final_prob_player2 = 1 - final_prob_player1
            
            # Calculate betting odds
            betting_odds = self._calculate_betting_odds(final_prob_player1, final_prob_player2)
            
            predictions['xgboost'] = {
                'player1_win_prob': final_prob_player1,
                'player2_win_prob': final_prob_player2,
                'raw_prob_1v2': prob_1v2,
                'raw_prob_2v1': prob_2v1,
                'betting_odds': betting_odds
            }
        
        if model in ['rf', 'both']:
            # Get probabilities for both orientations
            prob_1v2 = self.model_trainer.rf_model.predict_proba(X_pred_1v2)[0, 1]  # P(player1 wins)
            prob_2v1 = self.model_trainer.rf_model.predict_proba(X_pred_2v1)[0, 1]  # P(player2 wins)
            
            # Average the predictions: (prob1 + (1 - prob2)) / 2
            final_prob_player1 = (prob_1v2 + (1 - prob_2v1)) / 2
            final_prob_player2 = 1 - final_prob_player1
            
            # Calculate betting odds
            betting_odds = self._calculate_betting_odds(final_prob_player1, final_prob_player2)
            
            predictions['random_forest'] = {
                'player1_win_prob': final_prob_player1,
                'player2_win_prob': final_prob_player2,
                'raw_prob_1v2': prob_1v2,
                'raw_prob_2v1': prob_2v1,
                'betting_odds': betting_odds
            }
        
        # Add RFSR ensemble predictions if available
        if model in ['rfsr', 'both', 'ensemble'] and self.model_trainer.rfsr_ensemble is not None:
            # Get probabilities for both orientations
            prob_1v2 = self.model_trainer.rfsr_ensemble.predict_proba(X_pred_1v2)[0, 1]  # P(player1 wins)
            prob_2v1 = self.model_trainer.rfsr_ensemble.predict_proba(X_pred_2v1)[0, 1]  # P(player2 wins)
            
            # Average the predictions: (prob1 + (1 - prob2)) / 2
            final_prob_player1 = (prob_1v2 + (1 - prob_2v1)) / 2
            final_prob_player2 = 1 - final_prob_player1
            
            # Calculate betting odds
            betting_odds = self._calculate_betting_odds(final_prob_player1, final_prob_player2)
            
            predictions['rfsr_ensemble'] = {
                'player1_win_prob': final_prob_player1,
                'player2_win_prob': final_prob_player2,
                'raw_prob_1v2': prob_1v2,
                'raw_prob_2v1': prob_2v1,
                'betting_odds': betting_odds,
                'ensemble_weights': 'RF: 75%, SR: 25%'
            }
        
        # Add feature differences for display (using player1 - player2)
        predictions['feature_differences'] = features_1v2
        
        # Add bias check information
        bias_models = []
        if 'xgboost' in predictions:
            xgb_bias = abs(predictions['xgboost']['raw_prob_1v2'] + predictions['xgboost']['raw_prob_2v1'] - 1.0)
            bias_models.append(('xgboost_bias', xgb_bias))
        if 'random_forest' in predictions:
            rf_bias = abs(predictions['random_forest']['raw_prob_1v2'] + predictions['random_forest']['raw_prob_2v1'] - 1.0)
            bias_models.append(('random_forest_bias', rf_bias))
        if 'rfsr_ensemble' in predictions:
            rfsr_bias = abs(predictions['rfsr_ensemble']['raw_prob_1v2'] + predictions['rfsr_ensemble']['raw_prob_2v1'] - 1.0)
            bias_models.append(('rfsr_ensemble_bias', rfsr_bias))
        
        if bias_models:
            predictions['bias_check'] = {
                **dict(bias_models),
                'note': 'Bias should be close to 0. Higher values indicate model bias.'
            }
        
        return predictions 