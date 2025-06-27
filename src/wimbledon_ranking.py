import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from prediction import PredictionInterface
from data_pipeline import TennisDataLoader
from feature_engine import TennisFeatureEngine
from model_train import ModelTrainer
import pickle
import glob
import os

class WimbledonRankingPredictor:
    """Predicts top 25 players for Wimbledon 2025 using RFSR ensemble model"""
    
    def __init__(self, prediction_interface: PredictionInterface, matches_df: pd.DataFrame):
        self.predictor = prediction_interface
        self.matches_df = matches_df
        self.logger = logging.getLogger(__name__)
        
        # Top players to consider (based on current ATP rankings and recent form)
        self.top_players = [
            "Jannik Sinner", "Carlos Alcaraz", "Alexander Zverev", "Jack Draper",
            "Taylor Fritz", "Novak Djokovic", "Lorenzo Musetti", "Holger Rune",
            "Daniil Medvedev", "Ben Shelton", "Alex de Minaur", "Frances Tiafoe",
            "Tommy Paul", "Andrey Rublev", "Jakub Mensik", "Francisco Cerundolo",
            "Karen Khachanov", "Ugo Humbert", "Grigor Dimitrov", "Alexei Popyrin",
            "Tomas Machac", "Flavio Cobolli", "Jiri Lehecka", "Stefanos Tsitsipas",
            "Felix Auger Aliassime", "Alejandro Davidovich Fokina", "Denis Shapovalov",
            "Alexander Bublik", "Brandon Nakashima", "Alex Michelsen", "Tallon Griekspoor",
            "Matteo Berrettini"
        ]
    
    def get_all_player_features(self) -> Dict[str, Dict]:
        """Get features for all top players"""
        player_features = {}
        
        for player_name in self.top_players:
            try:
                features, player_id = self.predictor.get_player_features(player_name, self.matches_df)
                player_features[player_name] = {
                    'features': features,
                    'player_id': player_id
                }
                self.logger.info(f"✅ Loaded features for {player_name}")
            except Exception as e:
                self.logger.warning(f"❌ Could not load features for {player_name}: {e}")
        
        return player_features
    
    def calculate_wimbledon_score(self, features: Dict) -> float:
        """
        Calculate Wimbledon-specific score using RFSR ensemble weights
        Higher score = better chance at Wimbledon
        """
        # Grass-specific metrics (weighted heavily)
        grass_elo = features['elo_grass']
        grass_serve = features['serve_rating_grass']
        grass_return = features['return_rating_grass']
        grass_form = features['grass_winrate_last10']
        
        # General metrics (less weight for grass)
        general_elo = features['elo']
        general_serve = features['serve_rating']
        general_return = features['return_rating']
        general_form = features['last_10']
        
        # Wimbledon scoring formula (similar to RFSR ensemble weights)
        # 75% weight to grass-specific performance
        # 25% weight to general performance
        
        grass_score = (
            0.4 * grass_elo +      # Grass ELO (most important)
            0.3 * grass_serve +    # Grass serve rating
            0.2 * grass_return +   # Grass return rating
            0.1 * grass_form       # Grass form
        )
        
        general_score = (
            0.4 * general_elo +    # General ELO
            0.3 * general_serve +  # General serve
            0.2 * general_return + # General return
            0.1 * general_form     # General form
        )
        
        # Final weighted score (75% grass, 25% general)
        wimbledon_score = 0.75 * grass_score + 0.25 * general_score
        
        return wimbledon_score
    
    def predict_head_to_head_win_rate(self, player1: str, player2: str) -> float:
        """Predict win rate for player1 vs player2 using RFSR ensemble"""
        try:
            prediction = self.predictor.predict_match_outcome(
                player1, player2, self.matches_df, model='ensemble'
            )
            
            if prediction and 'rfsr_ensemble' in prediction:
                return prediction['rfsr_ensemble']['player1_win_prob']
            else:
                return 0.5  # Default to 50% if prediction fails
                
        except Exception as e:
            self.logger.warning(f"Could not predict {player1} vs {player2}: {e}")
            return 0.5
    
    def calculate_tournament_win_probability(self, player_features: Dict, 
                                          all_players: List[str]) -> float:
        """
        Calculate probability of winning Wimbledon by simulating against all other players
        """
        player_name = player_features['name']
        win_rates = []
        
        for opponent in all_players:
            if opponent != player_name:
                win_rate = self.predict_head_to_head_win_rate(player_name, opponent)
                win_rates.append(win_rate)
        
        # Average win rate against all opponents
        avg_win_rate = np.mean(win_rates) if win_rates else 0.5
        
        # Boost based on Wimbledon-specific score
        wimbledon_score = player_features['wimbledon_score']
        score_boost = (wimbledon_score - 1500) / 1000  # Normalize score boost
        
        # Final tournament win probability
        tournament_win_prob = avg_win_rate + (score_boost * 0.1)  # 10% boost max
        tournament_win_prob = max(0.01, min(0.99, tournament_win_prob))  # Clamp to [0.01, 0.99]
        
        return tournament_win_prob
    
    def generate_wimbledon_rankings(self) -> pd.DataFrame:
        """Generate comprehensive Wimbledon 2025 rankings"""
        self.logger.info("🎾 Generating Wimbledon 2025 rankings...")
        
        # Get features for all players
        player_data = self.get_all_player_features()
        
        if not player_data:
            raise ValueError("No player features could be loaded")
        
        rankings_data = []
        
        for player_name, data in player_data.items():
            features = data['features']
            player_id = data['player_id']
            
            # Calculate Wimbledon-specific score
            wimbledon_score = self.calculate_wimbledon_score(features)
            
            # Store player data
            player_info = {
                'player_name': player_name,
                'player_id': player_id,
                'wimbledon_score': wimbledon_score,
                'elo': features['elo'],
                'grass_elo': features['elo_grass'],
                'serve_rating': features['serve_rating'],
                'grass_serve_rating': features['serve_rating_grass'],
                'return_rating': features['return_rating'],
                'grass_return_rating': features['return_rating_grass'],
                'grass_form': features['grass_winrate_last10'],
                'general_form': features['last_10'],
                'atp_rank': features['rank'],
                'features': features
            }
            
            rankings_data.append(player_info)
        
        # Create DataFrame and calculate tournament win probabilities
        rankings_df = pd.DataFrame(rankings_data)
        all_player_names = rankings_df['player_name'].tolist()
        
        # Calculate tournament win probabilities for each player
        for idx, row in rankings_df.iterrows():
            player_info = {
                'name': row['player_name'],
                'wimbledon_score': row['wimbledon_score']
            }
            tournament_win_prob = self.calculate_tournament_win_probability(
                player_info, all_player_names
            )
            rankings_df.at[idx, 'tournament_win_probability'] = tournament_win_prob
        
        # Sort by tournament win probability (most likely to win Wimbledon)
        rankings_df = rankings_df.sort_values('tournament_win_probability', ascending=False)
        
        # Add ranking positions
        rankings_df['wimbledon_rank'] = range(1, len(rankings_df) + 1)
        
        # Reorder columns for better display (win probability first)
        column_order = [
            'wimbledon_rank', 'player_name', 'tournament_win_probability', 'wimbledon_score',
            'grass_elo', 'elo', 'grass_serve_rating', 'serve_rating',
            'grass_return_rating', 'return_rating', 'grass_form', 'general_form',
            'atp_rank', 'player_id'
        ]
        
        rankings_df = rankings_df[column_order]
        
        return rankings_df
    
    def display_top_25(self, rankings_df: pd.DataFrame) -> None:
        """Display the top 25 players for Wimbledon 2025"""
        print("\n" + "="*80)
        print("🎾 WIMBLEDON 2025 - TOP 25 PREDICTIONS (RFSR Ensemble Model)")
        print("="*80)
        
        top_25 = rankings_df.head(25)
        
        print(f"{'Rank':<4} {'Player':<20} {'Win Prob':<10} {'Wimbledon Score':<15} {'Grass ELO':<10} {'Grass Serve':<12} {'Grass Form':<10}")
        print("-" * 80)
        
        for _, row in top_25.iterrows():
            rank = row['wimbledon_rank']
            name = row['player_name'][:19]  # Truncate long names
            win_prob = f"{row['tournament_win_probability']:.1%}"
            score = f"{row['wimbledon_score']:.0f}"
            grass_elo = f"{row['grass_elo']:.0f}"
            grass_serve = f"{row['grass_serve_rating']:.0f}"
            grass_form = f"{row['grass_form']:.1%}"
            
            print(f"{rank:<4} {name:<20} {win_prob:<10} {score:<15} {grass_elo:<10} {grass_serve:<12} {grass_form:<10}")
        
        print("\n" + "="*80)
        print("📊 KEY INSIGHTS:")
        print("="*80)
        
        # Top 5 analysis
        top_5 = top_25.head(5)
        print(f"🏆 FAVORITE: {top_5.iloc[0]['player_name']} ({top_5.iloc[0]['tournament_win_probability']:.1%} win probability)")
        print(f"🥈 CONTENDER: {top_5.iloc[1]['player_name']} ({top_5.iloc[1]['tournament_win_probability']:.1%} win probability)")
        print(f"🥉 DARK HORSE: {top_5.iloc[2]['player_name']} ({top_5.iloc[2]['tournament_win_probability']:.1%} win probability)")
        
        # Grass specialists
        grass_specialists = top_25.nlargest(3, 'grass_elo')
        print(f"\n🌱 GRASS SPECIALISTS:")
        for _, player in grass_specialists.iterrows():
            print(f"   • {player['player_name']}: Grass ELO {player['grass_elo']:.0f}")
        
        # Serve specialists
        serve_specialists = top_25.nlargest(3, 'grass_serve_rating')
        print(f"\n🎯 SERVE SPECIALISTS:")
        for _, player in serve_specialists.iterrows():
            print(f"   • {player['player_name']}: Grass Serve {player['grass_serve_rating']:.0f}")
        
        # Form leaders
        form_leaders = top_25.nlargest(3, 'grass_form')
        print(f"\n📈 FORM LEADERS:")
        for _, player in form_leaders.iterrows():
            print(f"   • {player['player_name']}: Grass Form {player['grass_form']:.1%}")
    
    def save_rankings(self, rankings_df: pd.DataFrame, filename: str = "wimbledon_2025_rankings.csv") -> None:
        """Save rankings to CSV file"""
        # Remove features column for CSV export (if it exists)
        export_df = rankings_df.copy()
        if 'features' in export_df.columns:
            export_df = export_df.drop('features', axis=1)
        
        export_df.to_csv(filename, index=False)
        self.logger.info(f"💾 Rankings saved to {filename}")
    
    def get_player_analysis(self, player_name: str, rankings_df: pd.DataFrame) -> Dict:
        """Get detailed analysis for a specific player"""
        player_row = rankings_df[rankings_df['player_name'] == player_name]
        
        if player_row.empty:
            return {"error": f"Player {player_name} not found in rankings"}
        
        player = player_row.iloc[0]
        
        analysis = {
            'player_name': player['player_name'],
            'wimbledon_rank': int(player['wimbledon_rank']),
            'wimbledon_score': float(player['wimbledon_score']),
            'tournament_win_probability': float(player['tournament_win_probability']),
            'strengths': [],
            'weaknesses': [],
            'matchup_analysis': {}
        }
        
        # Analyze strengths and weaknesses
        if player['grass_elo'] > 2000:
            analysis['strengths'].append("Exceptional grass court ELO")
        elif player['grass_elo'] > 1800:
            analysis['strengths'].append("Strong grass court performance")
        
        if player['grass_serve_rating'] > 1800:
            analysis['strengths'].append("Powerful grass court serve")
        
        if player['grass_form'] > 0.8:
            analysis['strengths'].append("Excellent recent grass form")
        
        if player['grass_return_rating'] < 1500:
            analysis['weaknesses'].append("Below-average return game on grass")
        
        if player['grass_form'] < 0.4:
            analysis['weaknesses'].append("Poor recent grass form")
        
        # Head-to-head analysis against top 5
        top_5_players = rankings_df.head(5)['player_name'].tolist()
        for opponent in top_5_players:
            if opponent != player_name:
                win_prob = self.predict_head_to_head_win_rate(player_name, opponent)
                analysis['matchup_analysis'][opponent] = f"{win_prob:.1%}"
        
        return analysis

def main():
    """Main function to generate Wimbledon 2025 rankings"""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize prediction engine (reuse from main.py)
    from main import initialize_engine
    predictor, matches = initialize_engine()
    
    # Create ranking predictor
    ranking_predictor = WimbledonRankingPredictor(predictor, matches)
    
    # Generate rankings
    logger.info("🎾 Generating Wimbledon 2025 rankings...")
    rankings_df = ranking_predictor.generate_wimbledon_rankings()
    
    # Display results
    ranking_predictor.display_top_25(rankings_df)
    
    # Save rankings
    ranking_predictor.save_rankings(rankings_df)
    
    # Detailed analysis for top 3
    print("\n" + "="*80)
    print("🔍 DETAILED ANALYSIS - TOP 3 PLAYERS")
    print("="*80)
    
    for i, player_name in enumerate(rankings_df.head(3)['player_name']):
        analysis = ranking_predictor.get_player_analysis(player_name, rankings_df)
        print(f"\n{i+1}. {analysis['player_name']} (Rank #{analysis['wimbledon_rank']})")
        print(f"   Wimbledon Score: {analysis['wimbledon_score']:.0f}")
        print(f"   Win Probability: {analysis['tournament_win_probability']:.1%}")
        print(f"   Strengths: {', '.join(analysis['strengths'])}")
        print(f"   Weaknesses: {', '.join(analysis['weaknesses'])}")
        print(f"   vs Top 5:")
        for opponent, win_prob in analysis['matchup_analysis'].items():
            print(f"     • vs {opponent}: {win_prob}")

if __name__ == "__main__":
    main() 