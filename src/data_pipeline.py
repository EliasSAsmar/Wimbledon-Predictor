import pandas as pd
import glob
from typing import List, Dict, Tuple
from pathlib import Path
import logging

class TennisDataLoader:
    """ATP Match Data Loader and Preprocessor"""
    
    def __init__(self, data_dir: str = "atp_matches"):
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        self.essential_columns = [
            'winner_name', 'loser_name', 'surface', 
            'score', 'tourney_date', 'round',
            'winner_id', 'loser_id', 'draw_size'
        ]
        
    def load_raw_data(self, years: List[int] = None) -> pd.DataFrame:
        """
        Load raw ATP match data from CSV files
        Args:
            years: List of years to load (None loads all available)
        Returns:
            Combined DataFrame of match data
        """
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
                self.logger.info(f"Loaded {len(df)} matches from {Path(file).name}")
                
            matches = pd.concat(dfs, ignore_index=True)
            matches['tourney_date'] = pd.to_datetime(
                matches['tourney_date'].astype(str), 
                format='%Y%m%d'
            )
            
            self.logger.info(
                f"Loaded {len(matches)} matches from {len(files)} files. "
                f"Year range: {matches['tourney_date'].dt.year.min()}-"
                f"{matches['tourney_date'].dt.year.max()}"
            )
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and split data into grass/non-grass matches
        Args:
            df: Raw match DataFrame
        Returns:
            Tuple of (grass_matches, non_grass_matches)
        """
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
                f"Preprocessing complete. Grass matches: {len(grass_matches)}, "
                f"Non-grass matches: {len(non_grass_matches)}"
            )
            
            return grass_matches, non_grass_matches
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise
            
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