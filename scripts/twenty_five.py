import pandas as pd
import glob
import pickle
import hashlib
import os
from typing import List, Dict, Tuple
from pathlib import Path
import logging
from collections import defaultdict

# Add the src directory to the path to import other modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from feature_engine import TennisFeatureEngine

def build_name_mapping(existing_data_dir: str = "atp_matches") -> Dict[str, str]:
    """
    Build a mapping from "LastName FirstInitial." format to "FirstName LastName" format
    using existing ATP data
    """
    print("Building name mapping from existing data...")
    
    name_mapping = {}
    
    # Load existing ATP data to build mapping
    existing_files = glob.glob(f"{existing_data_dir}/atp_matches_*.csv")
    
    for file_path in existing_files:
        if "2025" in file_path:  # Skip the 2025 file we're processing
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # Get all unique names from winner_name and loser_name
            if 'winner_name' in df.columns and 'loser_name' in df.columns:
                all_names = set(df['winner_name'].dropna().tolist() + df['loser_name'].dropna().tolist())
                
                for full_name in all_names:
                    if pd.isna(full_name) or full_name == '':
                        continue
                    
                    # Extract last name and first initial
                    parts = full_name.strip().split()
                    if len(parts) >= 2:
                        first_name = parts[0]
                        last_name = ' '.join(parts[1:])  # Handle multi-part last names
                        
                        # Create the key in "LastName FirstInitial." format
                        first_initial = first_name[0].upper() if first_name else ''
                        key = f"{last_name} {first_initial}."
                        
                        name_mapping[key] = full_name
                        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Built mapping for {len(name_mapping)} players")
    return name_mapping

def load_and_filter_2025_data(file_path: str) -> pd.DataFrame:
    """
    Load ATP2025.csv and filter for 2025 data only
    """
    print(f"Loading data from {file_path}...")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for 2025 data only
    df_2025 = df[df['Date'].dt.year == 2025].copy()
    
    print(f"2025 data shape: {df_2025.shape}")
    print(f"Date range: {df_2025['Date'].min()} to {df_2025['Date'].max()}")
    
    return df_2025

def normalize_names_with_mapping(df: pd.DataFrame, name_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Normalize names using the name mapping from existing data
    """
    print("Normalizing player names with mapping...")
    
    def map_name(name_str):
        if pd.isna(name_str):
            return name_str
        
        name_str = str(name_str).strip()
        
        # First try direct mapping
        if name_str in name_mapping:
            return name_mapping[name_str]
        
        # If not found, try some variations
        # Handle cases where the format might be slightly different
        variations = [
            name_str,
            name_str.replace('.', ''),  # Remove periods
            name_str + '.' if not name_str.endswith('.') else name_str[:-1]  # Toggle period
        ]
        
        for variation in variations:
            if variation in name_mapping:
                return name_mapping[variation]
        
        # If still not found, create a best guess
        parts = name_str.replace('.', '').split()
        if len(parts) == 2:
            last_name, first_initial = parts[0], parts[1]
            # This is a fallback - would need manual review
            return f"{first_initial} {last_name}"
        
        return name_str
    
    # Apply mapping to player names
    df['Player_1'] = df['Player_1'].apply(map_name)
    df['Player_2'] = df['Player_2'].apply(map_name)
    df['Winner'] = df['Winner'].apply(map_name)
    
    # Count successful mappings
    mapped_count = 0
    unmapped_names = set()
    
    for col in ['Player_1', 'Player_2', 'Winner']:
        for name in df[col].dropna():
            if name in name_mapping.values():
                mapped_count += 1
            else:
                unmapped_names.add(name)
    
    print(f"Successfully mapped names, {len(unmapped_names)} unique names need review")
    if unmapped_names and len(unmapped_names) <= 20:
        print(f"Unmapped names: {sorted(list(unmapped_names))}")
    
    return df

def map_to_existing_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the ATP2025 format to match the existing ATP match data format
    """
    print("Mapping to existing data format...")
    
    # Create a mapping dictionary for the columns
    column_mapping = {
        'Tournament': 'tourney_name',
        'Date': 'tourney_date', 
        'Surface': 'surface',
        'Round': 'round',
        'Player_1': 'player1_name',
        'Player_2': 'player2_name',
        'Winner': 'winner_name',
        'Rank_1': 'player1_rank',
        'Rank_2': 'player2_rank',
        'Score': 'score'
    }
    
    # Rename columns to match existing format
    df_mapped = df.rename(columns=column_mapping)
    
    # Add missing columns that exist in the current format
    df_mapped['tourney_id'] = '2025'  # Use year as ID
    df_mapped['match_num'] = range(1, len(df_mapped) + 1)
    
    # Create winner/loser columns based on Winner field
    df_mapped['loser_name'] = df_mapped.apply(
        lambda row: row['player2_name'] if row['winner_name'] == row['player1_name'] else row['player1_name'], 
        axis=1
    )
    
    # Create winner/loser IDs (simplified version)
    df_mapped['winner_id'] = df_mapped['winner_name'].str.replace(' ', '_').str.lower()
    df_mapped['loser_id'] = df_mapped['loser_name'].str.replace(' ', '_').str.lower()
    
    # Handle missing data
    df_mapped['player1_rank'] = pd.to_numeric(df_mapped['player1_rank'], errors='coerce')
    df_mapped['player2_rank'] = pd.to_numeric(df_mapped['player2_rank'], errors='coerce')
    
    return df_mapped

def load_existing_ratings(feature_engine: TennisFeatureEngine) -> bool:
    """
    Load existing ratings from cache files
    """
    import glob
    
    # Find existing ratings files in cache
    rating_files = glob.glob(f"{feature_engine.cache_dir}/ratings_*.pkl")
    
    if not rating_files:
        print("No existing rating files found")
        return False
    
    # Use the most recent ratings file
    latest_ratings_file = max(rating_files, key=os.path.getmtime)
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
    ratings_path = os.path.join(feature_engine.cache_dir, f"ratings_2025_updated_{timestamp}.pkl")
    
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

def update_elo_ratings(df: pd.DataFrame) -> None:
    """
    Load existing ELO ratings and update them with 2025 match data
    """
    print("Updating ELO ratings with 2025 data...")
    
    try:
        # Initialize feature engine
        feature_engine = TennisFeatureEngine()
        
        # Try to load existing ratings
        if not load_existing_ratings(feature_engine):
            print("Starting with fresh ratings...")
        
        # Process 2025 matches to update ratings
        matches_processed = 0
        errors = []
        
        # Convert DataFrame to match the expected format
        for _, row in df.iterrows():
            try:
                # Convert to pandas Series with the expected format
                match_row = pd.Series({
                    'winner_id': row['winner_id'],
                    'loser_id': row['loser_id'],
                    'surface': row['surface'],
                    'tourney_date': row['tourney_date'],
                    'winner_rank': row.get('player1_rank', 999) if row['winner_name'] == row['player1_name'] else row.get('player2_rank', 999),
                    'loser_rank': row.get('player2_rank', 999) if row['winner_name'] == row['player1_name'] else row.get('player1_rank', 999),
                    'winner_seed': None,  # Not available in 2025 data
                    'loser_seed': None,   # Not available in 2025 data
                    # Match stats not available in 2025 data, so set to NaN
                    'w_1stIn': None,
                    'w_svpt': None,
                    'w_1stWon': None,
                    'w_2ndWon': None,
                    'l_1stIn': None,
                    'l_svpt': None,
                    'l_1stWon': None,
                    'l_2ndWon': None
                })
                
                # Update all ratings using the correct method
                feature_engine._update_all_ratings(match_row)
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
        print(f"Error updating ELO ratings: {e}")
        raise

def main():
    """
    Main function to process ATP2025 data
    """
    print("=== ATP 2025 Data Processing ===")
    
    # Set up paths
    data_file = Path("data/raw/atp2025.csv")
    
    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        return
    
    try:
        # Step 1: Build name mapping from existing data
        name_mapping = build_name_mapping()
        
        # Step 2: Load and filter 2025 data
        df_2025 = load_and_filter_2025_data(data_file)
        
        if len(df_2025) == 0:
            print("No 2025 data found in the file!")
            return
        
        # Step 3: Normalize names using mapping
        df_normalized = normalize_names_with_mapping(df_2025, name_mapping)
        
        # Step 4: Map to existing format
        df_mapped = map_to_existing_format(df_normalized)
        
        # Step 5: Save processed 2025 data
        output_file = Path("atp_matches/atp_matches_2025_processed.csv")
        df_mapped.to_csv(output_file, index=False)
        print(f"Saved processed 2025 data to {output_file}")
        
        # Step 6: Update ELO ratings
        update_elo_ratings(df_mapped)
        
        print("\n=== Processing Complete ===")
        print(f"Processed {len(df_mapped)} matches from 2025")
        print(f"Updated ELO ratings for players")
        
        # Show sample of processed data
        print("\nSample of processed data:")
        sample_cols = ['tourney_date', 'tourney_name', 'winner_name', 'loser_name', 'surface']
        print(df_mapped[sample_cols].head())
        
    except Exception as e:
        print(f"Error in main processing: {e}")
        raise

if __name__ == "__main__":
    main()

