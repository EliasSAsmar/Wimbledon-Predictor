import pandas as pd
import glob
import pickle
import hashlib
import os
from typing import List, Dict, Tuple
from pathlib import Path
import logging

class TennisDataLoader:
    """ATP Match Data Loader and Preprocessor with caching"""
    
    def __init__(self, data_dir: str = "data/raw", cache_dir: str = "cache"):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        self.essential_columns = [
            'winner_name', 'loser_name', 'surface', 
            'score', 'tourney_date', 'round',
            'winner_id', 'loser_id', 'draw_size'
        ]
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_data_hash(self, years: List[int] = None) -> str:
        """Generate hash of data files to detect changes"""
        files = glob.glob(f"{self.data_dir}/*.csv")
        if years:
            files = [f for f in files if any(str(year) in f for year in years)]
        
        # Sort files for consistent hashing
        files.sort()
        
        # Create hash from file paths and modification times
        hash_input = ""
        for file in files:
            stat = os.stat(file)
            hash_input += f"{file}:{stat.st_mtime}:{stat.st_size}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cache_path(self, cache_type: str, years: List[int] = None) -> str:
        """Get cache file path"""
        years_str = "_".join(map(str, sorted(years))) if years else "all"
        return os.path.join(self.cache_dir, f"{cache_type}_{years_str}.pkl")
    
    def _load_from_cache(self, cache_type: str, years: List[int] = None) -> pd.DataFrame:
        """Load data from cache if available and valid"""
        cache_path = self._get_cache_path(cache_type, years)
        hash_path = cache_path.replace('.pkl', '_hash.txt')
        
        if not os.path.exists(cache_path) or not os.path.exists(hash_path):
            return None
        
        # Check if cache is still valid
        with open(hash_path, 'r') as f:
            cached_hash = f.read().strip()
        
        current_hash = self._get_data_hash(years)
        if cached_hash != current_hash:
            self.logger.info("Data files changed, cache invalid")
            return None
        
        # Load cached data
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.info(f"✅ Loaded {cache_type} from cache: {len(data)} records")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_type: str, years: List[int] = None):
        """Save data to cache"""
        cache_path = self._get_cache_path(cache_type, years)
        hash_path = cache_path.replace('.pkl', '_hash.txt')
        
        try:
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save hash
            current_hash = self._get_data_hash(years)
            with open(hash_path, 'w') as f:
                f.write(current_hash)
            
            self.logger.info(f"💾 Saved {cache_type} to cache: {len(data)} records")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
        
    def load_raw_data(self, years: List[int] = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Load raw ATP match data from CSV files with caching
        Args:
            years: List of years to load (None loads all available)
            use_cache: Whether to use cached data if available
        Returns:
            Combined DataFrame of match data
        """
        # Try to load from cache first
        if use_cache:
            cached_data = self._load_from_cache("raw_data", years)
            if cached_data is not None:
                return cached_data
        
        # Load from CSV files
        self.logger.info(" Loading data from CSV files...")
        try:
            files = glob.glob(f"{self.data_dir}/*.csv")
            if years:
                files = [f for f in files if any(str(year) in f for year in years)]
                
            if not files:
                raise FileNotFoundError(f"No matching files found in {self.data_dir}")
                
            dfs = []
            for file in files:
                df = pd.read_csv(file)
                dfs.append(df)
                # Only log every 5th file to reduce spam
                if len(dfs) % 5 == 0 or len(dfs) == len(files):
                    self.logger.info(f"Loaded {len(dfs)}/{len(files)} files...")
                
            matches = pd.concat(dfs, ignore_index=True)
            # Handle date conversion more robustly
            # First, clean up any non-numeric values and convert properly
            date_series = matches['tourney_date'].fillna(0)  # Replace NaN with 0 temporarily
            date_series = pd.to_numeric(date_series, errors='coerce')  # Convert to numeric, NaN for invalid
            date_series = date_series.fillna(0).astype(int).astype(str)  # Convert to string
            # Replace '0' back with NaN for proper handling
            date_series = date_series.replace('0', pd.NaT)
            matches['tourney_date'] = pd.to_datetime(date_series, format='%Y%m%d', errors='coerce')
            
            self.logger.info(
                f"Loaded {len(matches)} matches from {len(files)} files. "
                f"Years: {matches['tourney_date'].dt.year.min()}-"
                f"{matches['tourney_date'].dt.year.max()}"
            )
            
            # Save to cache
            if use_cache:
                self._save_to_cache(matches, "raw_data", years)
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame, use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and split data into grass/non-grass matches with caching
        Args:
            df: Raw match DataFrame
            use_cache: Whether to use cached data if available
        Returns:
            Tuple of (grass_matches, non_grass_matches)
        """
        # Generate a hash for the input data to check cache validity
        data_hash = hashlib.md5(str(df.shape).encode()).hexdigest()
        
        # Try to load from cache
        if use_cache:
            grass_cache = self._load_from_cache(f"grass_data_{data_hash}")
            non_grass_cache = self._load_from_cache(f"non_grass_data_{data_hash}")
            
            if grass_cache is not None and non_grass_cache is not None:
                self.logger.info(" Loaded preprocessed data from cache")
                return grass_cache, non_grass_cache
        
        # Process data
        self.logger.info("🔄 Preprocessing data...")
        try:
            # Validate essential columns
            missing_cols = [col for col in self.essential_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Clean data
            clean_df = df.dropna(subset=self.essential_columns).copy()
            clean_df = clean_df[clean_df['draw_size'] >= 32]  # Only main draws
            
            # Split by surface
            grass_matches = clean_df[clean_df['surface'] == 'Grass'].copy()
            non_grass_matches = clean_df[clean_df['surface'] != 'Grass'].copy()
            
            self.logger.info(
                f" Preprocessing complete. Grass: {len(grass_matches)}, "
                f"Non-grass: {len(non_grass_matches)}"
            )
            
            # Save to cache
            if use_cache:
                self._save_to_cache(grass_matches, f"grass_data_{data_hash}")
                self._save_to_cache(non_grass_matches, f"non_grass_data_{data_hash}")
            
            return grass_matches, non_grass_matches
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear all cached data"""
        cache_files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        hash_files = glob.glob(os.path.join(self.cache_dir, "*_hash.txt"))
        
        for file in cache_files + hash_files:
            try:
                os.remove(file)
                self.logger.info(f"Removed cache file: {file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {file}: {e}")
                
    def get_year_distribution(self, df: pd.DataFrame) -> Dict[int, int]:
        """Get match count by year"""
        return df['tourney_date'].dt.year.value_counts().sort_index().to_dict()
        
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Basic data validation checks"""
        checks = {
            'has_matches': len(df) > 0,
            'has_dates': df['tourney_date'].notnull().all(),
            'valid_surfaces': df['surface'].isin(['Grass', 'Clay', 'Hard']).all()
        }
        return all(checks.values())