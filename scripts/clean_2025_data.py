import pandas as pd
import os
from pathlib import Path

def clean_2025_data():
    """
    Clean ATP2025.csv to only keep 2025 matches
    This makes it easier to add new grass tournament data
    """
    print("=== Cleaning ATP2025.csv to only keep 2025 matches ===")
    
    # Set up paths
    input_file = Path("data/raw/ATP2025.csv")
    output_file = Path("data/raw/ATP2025_cleaned.csv")
    backup_file = Path("data/raw/ATP2025_backup.csv")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return
    
    try:
        # Load the data
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Show date range
        print(f"Original date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Filter for 2025 data only
        df_2025 = df[df['Date'].dt.year == 2025].copy()
        
        print(f"2025 data shape: {df_2025.shape}")
        print(f"2025 date range: {df_2025['Date'].min()} to {df_2025['Date'].max()}")
        
        # Show surface breakdown
        print("\nSurface breakdown:")
        print(df_2025['Surface'].value_counts())
        
        # Show tournament breakdown
        print("\nTournament breakdown:")
        print(df_2025['Tournament'].value_counts().head(10))
        
        # Create backup of original file
        print(f"\nCreating backup: {backup_file}")
        df.to_csv(backup_file, index=False)
        
        # Save cleaned 2025 data
        print(f"Saving cleaned 2025 data: {output_file}")
        df_2025.to_csv(output_file, index=False)
        
        # Also replace the original file
        print(f"Replacing original file with cleaned data...")
        df_2025.to_csv(input_file, index=False)
        
        print("\n=== Cleaning Complete ===")
        print(f"✅ Original file backed up to: {backup_file}")
        print(f"✅ Cleaned file saved to: {output_file}")
        print(f"✅ Original file replaced with cleaned data")
        print(f"📊 Kept {len(df_2025)} matches from 2025")
        print(f"🗑️ Removed {len(df) - len(df_2025)} non-2025 matches")
        
        # Show what was removed
        removed_data = df[df['Date'].dt.year != 2025]
        if len(removed_data) > 0:
            print(f"\nRemoved data years: {sorted(removed_data['Date'].dt.year.unique())}")
        
    except Exception as e:
        print(f"Error cleaning data: {e}")
        raise

if __name__ == "__main__":
    clean_2025_data() 