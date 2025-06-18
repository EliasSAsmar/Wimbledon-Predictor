import pandas as pd
import numpy as np
import pickle
import hashlib
import os
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class FeatureConfig:
    """Configuration for feature engineering parameters"""
    base_elo: int = 1500
    k_factor: int = 32
    grass_k_factor: int = 40
    serve_rating_weights: Tuple[float, float, float] = (0.3, 0.4, 0.3)
    rating_update_factor: int = 20
    grass_rating_boost: int = 25
    form_window_size: int = 10

class DefaultElo:
    """Callable class to replace lambda for pickle compatibility"""
    def __init__(self, base_elo: int):
        self.base_elo = base_elo
    
    def __call__(self):
        return self.base_elo

class DefaultDeque:
    """Callable class to replace lambda for pickle compatibility"""
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
    
    def __call__(self):
        return deque(maxlen=self.maxlen)

class TennisFeatureEngine:
    """
    Professional-grade feature engineering pipeline for tennis predictions with:
    - Elo rating systems (general and surface-specific)
    - Serve/return performance metrics
    - Recent form tracking
    - Temporal feature validation
    - Intelligent caching system
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None, cache_dir: str = "cache"):
        self.config = config if config else FeatureConfig()
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize rating systems
        self._init_rating_systems()
        self._init_feature_storage()
        
    def _get_features_hash(self, matches_df: pd.DataFrame, split_date: str) -> str:
        """Generate hash for feature cache validation"""
        # Include data shape, split date, and config in hash
        hash_input = f"{matches_df.shape}:{split_date}:{str(self.config)}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _load_features_from_cache(self, cache_hash: str) -> bool:
        """Load features and ratings from cache if available"""
        cache_path = os.path.join(self.cache_dir, f"features_{cache_hash}.pkl")
        ratings_path = os.path.join(self.cache_dir, f"ratings_{cache_hash}.pkl")
        
        if not os.path.exists(cache_path) or not os.path.exists(ratings_path):
            return False
        
        try:
            # Load features
            with open(cache_path, 'rb') as f:
                self.features = pickle.load(f)
            
            # Load ratings
            with open(ratings_path, 'rb') as f:
                ratings_data = pickle.load(f)
                self.player_elos = ratings_data['player_elos']
                self.player_serve_ratings = ratings_data['player_serve_ratings']
                self.player_return_ratings = ratings_data['player_return_ratings']
                self.player_grass_elos = ratings_data['player_grass_elos']
                self.player_grass_serve_ratings = ratings_data['player_grass_serve_ratings']
                self.player_grass_return_ratings = ratings_data['player_grass_return_ratings']
                self.player_last_10 = ratings_data['player_last_10']
                self.player_grass_last_10 = ratings_data['player_grass_last_10']
            
            self.logger.info(f"✅ Loaded from cache: Train={len(self.features['train'])}, Test={len(self.features['test'])}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Cache load failed: {e}")
            return False
    
    def _save_features_to_cache(self, cache_hash: str):
        """Save features and ratings to cache"""
        cache_path = os.path.join(self.cache_dir, f"features_{cache_hash}.pkl")
        ratings_path = os.path.join(self.cache_dir, f"ratings_{cache_hash}.pkl")
        
        try:
            # Save features
            with open(cache_path, 'wb') as f:
                pickle.dump(self.features, f)
            
            # Save ratings (convert defaultdicts to regular dicts for pickle)
            ratings_data = {
                'player_elos': dict(self.player_elos),
                'player_serve_ratings': dict(self.player_serve_ratings),
                'player_return_ratings': dict(self.player_return_ratings),
                'player_grass_elos': dict(self.player_grass_elos),
                'player_grass_serve_ratings': dict(self.player_grass_serve_ratings),
                'player_grass_return_ratings': dict(self.player_grass_return_ratings),
                'player_last_10': dict(self.player_last_10),
                'player_grass_last_10': dict(self.player_grass_last_10)
            }
            with open(ratings_path, 'wb') as f:
                pickle.dump(ratings_data, f)
            
            self.logger.info(f"💾 Saved to cache: Train={len(self.features['train'])}, Test={len(self.features['test'])}")
            
        except Exception as e:
            self.logger.warning(f"Cache save failed: {e}")
        
    def _init_rating_systems(self):
        """Initialize all rating trackers with pickle-compatible defaults"""
        # Use pickle-compatible callable classes instead of lambdas
        default_elo = DefaultElo(self.config.base_elo)
        default_deque = DefaultDeque(self.config.form_window_size)
        
        self.player_elos = defaultdict(default_elo)
        self.player_serve_ratings = defaultdict(default_elo)
        self.player_return_ratings = defaultdict(default_elo)
        
        # Grass-specific ratings
        self.player_grass_elos = defaultdict(default_elo)
        self.player_grass_serve_ratings = defaultdict(default_elo)
        self.player_grass_return_ratings = defaultdict(default_elo)
        
        # Rolling form tracking
        self.player_last_10 = defaultdict(default_deque)
        self.player_grass_last_10 = defaultdict(default_deque)
        
        # Keep the new structure for compatibility
        self.ratings = {
            'elo': self.player_elos,
            'grass_elo': self.player_grass_elos,
            'serve': self.player_serve_ratings,
            'grass_serve': self.player_grass_serve_ratings,
            'return': self.player_return_ratings,
            'grass_return': self.player_grass_return_ratings
        }
        
        self.form_trackers = {
            'all_surfaces': self.player_last_10,
            'grass': self.player_grass_last_10
        }
    
    def _init_feature_storage(self):
        """Initialize feature storage systems"""
        self.features = {
            'train': {},
            'test': {},
            'metadata': {
                'min_date': None,
                'max_date': None,
                'player_count': 0
            }
        }
    
    def calculate_win_probability(self, rating1: float, rating2: float) -> float:
        """Calculate expected win probability between two ratings"""
        return 1 / (1 + 10**((rating2 - rating1)/400))
    
    def update_rating(self, player_id: str, opponent_id: str, rating_type: str, 
                    actual_result: float, is_grass: bool = False) -> None:
        """
        Generic rating update function for all rating systems
        Args:
            player_id: ID of player to update
            opponent_id: ID of opponent
            rating_type: Type of rating ('elo', 'serve', 'return')
            actual_result: 1 for win, 0 for loss
            is_grass: Whether match was on grass
        """
        try:
            # Determine rating system and parameters
            system_key = f"grass_{rating_type}" if is_grass else rating_type
            k_factor = self.config.grass_k_factor if is_grass else self.config.k_factor
            
            # Get current ratings
            player_rating = self.ratings[system_key][player_id]
            opponent_rating = self.ratings[system_key][opponent_id]
            
            # Calculate rating change
            expected = self.calculate_win_probability(player_rating, opponent_rating)
            rating_change = k_factor * (actual_result - expected)
            
            # Apply update
            self.ratings[system_key][player_id] += rating_change
            self.ratings[system_key][opponent_id] -= rating_change
            
        except KeyError as e:
            self.logger.error(f"Invalid rating system: {str(e)}")
            raise
    
    def update_serve_return_stats(self, row: pd.Series, winner_id: str, 
                                loser_id: str, is_grass: bool) -> None:
        """
        Update serve and return ratings based on match statistics
        """
        # Process winner's serve stats
        if not pd.isna(row.get('w_1stIn', np.nan)):
            serve_stats = self._calculate_serve_performance(
                row['w_1stIn'], 
                row['w_svpt'],
                row['w_1stWon'],
                row['w_2ndWon']
            )
            self._update_serve_metrics(winner_id, serve_stats, is_grass)
            self._update_return_metrics(loser_id, 1 - serve_stats, is_grass)
        
        # Process loser's serve stats
        if not pd.isna(row.get('l_1stIn', np.nan)):
            serve_stats = self._calculate_serve_performance(
                row['l_1stIn'],
                row['l_svpt'],
                row['l_1stWon'],
                row['l_2ndWon']
            )
            self._update_serve_metrics(loser_id, serve_stats, is_grass)
            self._update_return_metrics(winner_id, 1 - serve_stats, is_grass)
    
    def _calculate_serve_performance(self, first_in: int, total_serves: int,
                                   first_won: int, second_won: int) -> float:
        """Calculate composite serve performance metric"""
        if total_serves == 0:
            return 0.5
            
        serve_pct = first_in / total_serves
        first_win_pct = first_won / first_in if first_in > 0 else 0
        second_win_pct = second_won / (total_serves - first_in) if (total_serves - first_in) > 0 else 0
        
        weights = self.config.serve_rating_weights
        return (weights[0] * serve_pct + 
                weights[1] * first_win_pct + 
                weights[2] * second_win_pct)
    
    def _update_serve_metrics(self, player_id: str, performance: float, 
                            is_grass: bool) -> None:
        """Update serve ratings based on performance"""
        update_factor = self.config.grass_rating_boost if is_grass else self.config.rating_update_factor
        self.ratings['grass_serve' if is_grass else 'serve'][player_id] += update_factor * (performance - 0.5)
    
    def _update_return_metrics(self, player_id: str, performance: float, 
                             is_grass: bool) -> None:
        """Update return ratings based on performance"""
        update_factor = self.config.grass_rating_boost if is_grass else self.config.rating_update_factor
        self.ratings['grass_return' if is_grass else 'return'][player_id] += update_factor * (performance - 0.5)
    
    def build_features(self, matches_df: pd.DataFrame, split_date: str = '2024-07-01', use_cache: bool = True) -> None:
        """
        Process matches chronologically and build features with temporal validation and caching
        Args:
            matches_df: DataFrame of match data
            split_date: Date to split train/test sets
            use_cache: Whether to use cached features if available
        """
        # Try to load from cache first
        if use_cache:
            cache_hash = self._get_features_hash(matches_df, split_date)
            if self._load_features_from_cache(cache_hash):
                return
        
        # Build features from scratch
        self.logger.info("🔄 Building features from scratch...")
        sorted_matches = matches_df.sort_values('tourney_date').copy()
        split_date = pd.to_datetime(split_date)
        
        total_matches = len(sorted_matches)
        self.logger.info(f"📊 Processing {total_matches} matches chronologically")
        
        for idx, (_, row) in enumerate(sorted_matches.iterrows()):
            # Feature extraction before updates to prevent leakage
            features = self._extract_match_features(row)
            
            # Store features with temporal validation
            self._store_features(row, features, split_date)
            
            # Update all rating systems
            self._update_all_ratings(row)
            
            # Less frequent logging
            if idx % 10000 == 0 and idx > 0:
                progress = (idx / total_matches) * 100
                self.logger.info(f"⚡ Progress: {progress:.1f}% ({idx}/{total_matches})")
        
        self._log_feature_stats()
        
        # Save to cache
        if use_cache:
            self._save_features_to_cache(cache_hash)
    
    def clear_cache(self):
        """Clear all cached features"""
        import glob
        cache_files = glob.glob(os.path.join(self.cache_dir, "features_*.pkl"))
        ratings_files = glob.glob(os.path.join(self.cache_dir, "ratings_*.pkl"))
        
        for file in cache_files + ratings_files:
            try:
                os.remove(file)
                self.logger.info(f"🗑️ Removed cache file: {file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {file}: {e}")
    
    def _calculate_seed_diff(self, row: pd.Series) -> float:
        """Calculate the difference in tournament seeding between players"""
        winner_seed = row.get('winner_seed', 0)
        loser_seed = row.get('loser_seed', 0)
        
        # Handle NaN values
        if pd.isna(winner_seed):
            winner_seed = 0
        if pd.isna(loser_seed):
            loser_seed = 0
            
        return loser_seed - winner_seed  # Reverse as lower seed is better

    def _extract_match_features(self, row: pd.Series) -> Dict:
        """Extract all features for a match - EXACT copy from tempy.py"""
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        
        return {
            'elo_diff': self.player_elos[winner_id] - self.player_elos[loser_id],
            'elo_grass_diff': self.player_grass_elos[winner_id] - self.player_grass_elos[loser_id],
            'serve_rating_diff': self.player_serve_ratings[winner_id] - self.player_serve_ratings[loser_id],
            'serve_rating_grass_diff': self.player_grass_serve_ratings[winner_id] - self.player_grass_serve_ratings[loser_id],
            'return_rating_diff': self.player_return_ratings[winner_id] - self.player_return_ratings[loser_id],
            'return_rating_grass_diff': self.player_grass_return_ratings[winner_id] - self.player_grass_return_ratings[loser_id],
            'grass_winrate_last10_diff': self._get_grass_form(winner_id) - self._get_grass_form(loser_id),
            'rank_diff': row.get('loser_rank', 999) - row.get('winner_rank', 999),
            'seed_diff': (row.get('loser_seed', 0) - row.get('winner_seed', 0)) if pd.notna(row.get('winner_seed')) and pd.notna(row.get('loser_seed')) else 0,
            'age_diff': 0,  # Age data not consistently available
            'last_10_diff': self._get_recent_form(winner_id) - self._get_recent_form(loser_id)
        }

    def _update_metadata(self, date: pd.Timestamp) -> None:
        """Update metadata with date information"""
        if self.features['metadata']['min_date'] is None or date < self.features['metadata']['min_date']:
            self.features['metadata']['min_date'] = date
        if self.features['metadata']['max_date'] is None or date > self.features['metadata']['max_date']:
            self.features['metadata']['max_date'] = date

    def _store_features(self, row: pd.Series, features: Dict, split_date: pd.Timestamp) -> None:
        """Store features with proper temporal split"""
        match_id = f"{row['tourney_id']}_{row['match_num']}"
        feature_data = {
            **features,
            'date': row['tourney_date'],
            'tournament': row['tourney_name'],
            'surface': row['surface'],
            'winner_id': row['winner_id'],
            'loser_id': row['loser_id']
        }
        
        if row['tourney_date'] < split_date:
            self.features['train'][match_id] = feature_data
        else:
            self.features['test'][match_id] = feature_data
        
        # Update metadata
        self._update_metadata(row['tourney_date'])
    
    def _update_all_ratings(self, row: pd.Series) -> None:
        """Update all rating systems after feature extraction - EXACT copy from tempy.py"""
        winner_id = row['winner_id']
        loser_id = row['loser_id']
        is_grass = row['surface'] == 'Grass'
        
        # Update general Elo - EXACT copy from tempy.py
        winner_elo, loser_elo = self.player_elos[winner_id], self.player_elos[loser_id]
        expected = 1 / (1 + 10**((loser_elo - winner_elo)/400))
        k = 32
        
        self.player_elos[winner_id] += k * (1 - expected)
        self.player_elos[loser_id] += k * (0 - (1 - expected))
        
        # Update grass Elo (if grass match) - EXACT copy from tempy.py
        if is_grass:
            winner_grass_elo = self.player_grass_elos[winner_id]
            loser_grass_elo = self.player_grass_elos[loser_id]
            expected_grass = 1 / (1 + 10**((loser_grass_elo - winner_grass_elo)/400))
            k_grass = 40
            
            self.player_grass_elos[winner_id] += k_grass * (1 - expected_grass)
            self.player_grass_elos[loser_id] += k_grass * (0 - (1 - expected_grass))
        
        # Update form tracking - EXACT copy from tempy.py
        self.player_last_10[winner_id].append(1)
        self.player_last_10[loser_id].append(0)
        
        if is_grass:
            self.player_grass_last_10[winner_id].append(1)
            self.player_grass_last_10[loser_id].append(0)
        
        # Update serve/return ratings if stats available - EXACT copy from tempy.py
        self._update_serve_return_ratings_tempy_style(row, winner_id, loser_id, is_grass)
    
    def _get_recent_form(self, player_id: str) -> float:
        """Get win rate from last 10 matches overall - match tempy.py exactly"""
        recent_results = self.player_last_10[player_id]
        return sum(recent_results) / len(recent_results) if recent_results else 0.5
    
    def _get_grass_form(self, player_id: str) -> float:
        """Get player's grass court win rate from last 10 grass matches - match tempy.py exactly"""
        grass_results = self.player_grass_last_10[player_id]
        return sum(grass_results) / len(grass_results) if grass_results else 0.5

    def _update_serve_return_ratings_tempy_style(self, row: pd.Series, winner_id: str, loser_id: str, is_grass: bool) -> None:
        """Update serve and return ratings based on match statistics - EXACT copy from tempy.py"""
        # Winner serve performance
        if not pd.isna(row.get('w_1stIn', np.nan)):
            serve_pct = row['w_1stIn'] / row['w_svpt'] if row['w_svpt'] > 0 else 0
            first_won = row['w_1stWon'] / row['w_1stIn'] if row['w_1stIn'] > 0 else 0
            second_won = row['w_2ndWon'] / (row['w_svpt'] - row['w_1stIn']) if (row['w_svpt'] - row['w_1stIn']) > 0 else 0
            
            serve_performance = serve_pct * 0.3 + first_won * 0.4 + second_won * 0.3
            
            self.player_serve_ratings[winner_id] += 20 * (serve_performance - 0.5)
            if is_grass:
                self.player_grass_serve_ratings[winner_id] += 25 * (serve_performance - 0.5)
            
            # Loser return performance
            return_performance = 1 - serve_performance
            self.player_return_ratings[loser_id] += 20 * (return_performance - 0.5)
            if is_grass:
                self.player_grass_return_ratings[loser_id] += 25 * (return_performance - 0.5)
        
        # Similar for loser serve stats
        if not pd.isna(row.get('l_1stIn', np.nan)):
            serve_pct = row['l_1stIn'] / row['l_svpt'] if row['l_svpt'] > 0 else 0
            first_won = row['l_1stWon'] / row['l_1stIn'] if row['l_1stIn'] > 0 else 0
            second_won = row['l_2ndWon'] / (row['l_svpt'] - row['l_1stIn']) if (row['l_svpt'] - row['l_1stIn']) > 0 else 0
            
            serve_performance = serve_pct * 0.3 + first_won * 0.4 + second_won * 0.3
            
            self.player_serve_ratings[loser_id] += 20 * (serve_performance - 0.5)
            if is_grass:
                self.player_grass_serve_ratings[loser_id] += 25 * (serve_performance - 0.5)
            
            return_performance = 1 - serve_performance
            self.player_return_ratings[winner_id] += 20 * (return_performance - 0.5)
            if is_grass:
                self.player_grass_return_ratings[winner_id] += 25 * (return_performance - 0.5)

    def _log_feature_stats(self) -> None:
        """Log feature statistics"""
        self.logger.info(
            f"✅ Features complete! Train: {len(self.features['train'])}, "
            f"Test: {len(self.features['test'])}. "
            f"Date range: {self.features['metadata']['min_date'].date()} to {self.features['metadata']['max_date'].date()}"
        )

    def save_features(self, train_file: str = 'tennis_train_features.pkl', 
                     test_file: str = 'tennis_test_features.pkl') -> None:
        """Save features to pickle files"""
        with open(train_file, 'wb') as f:
            pickle.dump(self.features['train'], f)
        with open(test_file, 'wb') as f:
            pickle.dump(self.features['test'], f)
        self.logger.info(f"Features saved: {train_file}, {test_file}")

    # Additional helper methods would be implemented here...