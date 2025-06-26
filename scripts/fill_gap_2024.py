#!/usr/bin/env python3
"""
Fill the missing data gap from July 1, 2024 to December 18, 2024
This script processes the 2024 data that wasn't included in model training
to update player ratings before running twenty_five.py
"""

import pandas as pd
import pickle
import os
import sys
from pathlib import Path
import logging

# Add the src directory to the path to import other modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feature_engine import TennisFeatureEngine

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_existing_ratings(feature_engine: TennisFeatureEngine) -> bool:
    """
    Load existing ratings from cache files (as of July 1, 2024)
    """
    import glob
    
    # Find existing ratings files in cache
    rating_files = glob.glob(f"{feature_engine.cache_dir}/ratings_*.pkl")
    
    if not rating_files:
        print("No existing rating files found")
        return False
    
    # Sort by modification time to get the most recent
    rating_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_ratings_file = rating_files[0]
    
    print(f"Loading ratings from: {latest_ratings_file}")
    
    try:
        with open(latest_ratings_file, 'rb') as f:
            ratings_data = pickle.load(f)
        
        # Load all the rating systems
        feature_engine.player_elos.update(ratings_data['player_elos'])
        feature_engine.player_serve_ratings.update(ratings_data['player_serve_ratings'])
        feature_engine.player_return_ratings.update(ratings_data['player_return_ratings'])
        feature_engine.player_grass_elos.update(ratings_data['player_grass_elos'])
        feature_engine.player_grass_serve_ratings.update(ratings_data['player_grass_serve_ratings'])
        feature_engine.player_grass_return_ratings.update(ratings_data['player_grass_return_ratings'])
        feature_engine.player_last_10.update(ratings_data['player_last_10'])
        feature_engine.player_grass_last_10.update(ratings_data['player_grass_last_10'])
        
        print(f"Successfully loaded ratings for {len(feature_engine.player_elos)} players")
        print(f"Successfully loaded grass ratings for {len(feature_engine.player_grass_elos)} players")
        return True
        
    except Exception as e:
        print(f"Error loading ratings: {e}")
        return False

def save_updated_ratings(feature_engine: TennisFeatureEngine) -> None:
    """
    Save updated ratings to a new cache file
    """
    import time
    
    # Create a new cache filename with timestamp
    timestamp = int(time.time())
    ratings_path = os.path.join(feature_engine.cache_dir, f"ratings_gap_filled_{timestamp}.pkl")
    
    try:
        # Save ratings (convert defaultdicts to regular dicts for pickle)
        ratings_data = {
            'player_elos': dict(feature_engine.player_elos),
            'player_serve_ratings': dict(feature_engine.player_serve_ratings),
            'player_return_ratings': dict(feature_engine.player_return_ratings),
            'player_grass_elos': dict(feature_engine.player_grass_elos),
            'player_grass_serve_ratings': dict(feature_engine.player_grass_serve_ratings),
            'player_grass_return_ratings': dict(feature_engine.player_grass_return_ratings),
            'player_last_10': {k: list(v) for k, v in feature_engine.player_last_10.items()},
            'player_grass_last_10': {k: list(v) for k, v in feature_engine.player_grass_last_10.items()}
        }
        
        with open(ratings_path, 'wb') as f:
            pickle.dump(ratings_data, f)
        
        print(f"Saved updated ratings to: {ratings_path}")
        
    except Exception as e:
        print(f"Error saving ratings: {e}")
        raise

def process_gap_data(df: pd.DataFrame) -> None:
    """
    Process the gap data from July 1, 2024 to December 18, 2024
    """
    print("Processing gap data from July 1, 2024 to December 18, 2024...")
    
    try:
        # Initialize feature engine
        feature_engine = TennisFeatureEngine()
        
        # Try to load existing ratings (as of July 1, 2024)
        if not load_existing_ratings(feature_engine):
            print("No existing ratings found. Starting with fresh ratings...")
        
        # Filter for gap period: July 1, 2024 to December 18, 2024
        gap_start = pd.to_datetime('2024-07-01')
        gap_end = pd.to_datetime('2024-12-18')
        
        gap_data = df[
            (df['tourney_date'] >= gap_start) & 
            (df['tourney_date'] <= gap_end)
        ].copy()
        
        print(f"Found {len(gap_data)} matches in gap period")
        print(f"Date range: {gap_data['tourney_date'].min()} to {gap_data['tourney_date'].max()}")
        
        # Sort by date to process chronologically
        gap_data = gap_data.sort_values('tourney_date')
        
        # Process matches to update ratings
        matches_processed = 0
        errors = []
        
        for _, row in gap_data.iterrows():
            try:
                # Update all ratings using the feature engine method
                feature_engine._update_all_ratings(row)
                matches_processed += 1
                
                if matches_processed % 100 == 0:
                    print(f"Processed {matches_processed} matches...")
                    
            except Exception as e:
                error_msg = f"Error processing match {row.get('match_num', 'unknown')}: {e}"
                errors.append(error_msg)
                if len(errors) <= 5:  # Only print first 5 errors
                    print(error_msg)
                continue
        
        print(f"Successfully processed {matches_processed} matches")
        if errors:
            print(f"Encountered {len(errors)} errors during processing")
        
        # Save updated ratings
        save_updated_ratings(feature_engine)
        
        # Print some statistics
        print(f"Total players with ELO ratings: {len(feature_engine.player_elos)}")
        print(f"Total players with grass ELO ratings: {len(feature_engine.player_grass_elos)}")
        
        # Show top 10 players by ELO
        if len(feature_engine.player_elos) > 0:
            top_players = sorted(
                feature_engine.player_elos.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            print("\nTop 10 players by ELO rating:")
            for i, (player_id, rating) in enumerate(top_players, 1):
                print(f"{i:2d}. {player_id}: {rating:.1f}")
                
        # Show top 10 players by grass ELO
        if len(feature_engine.player_grass_elos) > 0:
            top_grass_players = sorted(
                feature_engine.player_grass_elos.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            print("\nTop 10 players by Grass ELO rating:")
            for i, (player_id, rating) in enumerate(top_grass_players, 1):
                print(f"{i:2d}. {player_id}: {rating:.1f}")
    
    except Exception as e:
        print(f"Error processing gap data: {e}")
        raise

def main():
    """
    Main function to fill the 2024 data gap
    """
    print("=== Filling 2024 Data Gap (July 1 - December 18) ===")
    
    # Set up logging
    setup_logging()
    
    # Set up paths
    data_file = Path("data/raw/atp_matches_2024.csv")
    
    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        return
    
    try:
        # Load 2024 data
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        
        # Convert tourney_date to datetime (handle YYYYMMDD integer format)
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
        
        print(f"Loaded {len(df)} matches from 2024")
        print(f"Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
        
        # Process the gap data
        process_gap_data(df)
        
        print("\n=== Gap Filling Complete ===")
        print("✅ Ratings updated with July 1 - December 18, 2024 data")
        print("✅ Now you can run twenty_five.py to add 2025 data")
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    main() 