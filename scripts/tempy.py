# Wimbledon Predictor - Consolidated Code

import pandas as pd 
import numpy as np 
import glob
import pickle
from collections import defaultdict, deque
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import xgboost as xgb

# ==============================================
# DATA LOADING AND PREPARATION
# ==============================================

# Load ATP match data
print("Loading ATP match data...")
files = sorted(glob.glob("atp_matches/*.csv"))
dfs = [pd.read_csv(file) for file in files]
matches = pd.concat(dfs, ignore_index=True)

# Convert date and show summary
matches['tourney_date'] = pd.to_datetime(matches['tourney_date'].astype(str), format='%Y%m%d')
print(f"Loaded {len(matches)} matches from {len(files)} files (2010-2024)")
print(f"Year distribution: {matches['tourney_date'].dt.year.value_counts().sort_index().to_dict()}")

# Filter and clean data
essential_columns = ['winner_name', 'loser_name', 'surface', 'score', 'tourney_date', 'round']

# Grass court matches (for Wimbledon-specific analysis)
grass_matches = matches[
    (matches['surface'] == 'Grass') & 
    (matches['draw_size'] >= 32)
].dropna(subset=essential_columns).copy()

# Non-grass matches (for general tennis features)
non_grass_matches = matches[
    (matches['surface'] != 'Grass') & 
    (matches['draw_size'] >= 32)
].dropna(subset=essential_columns).copy()

print(f"Grass matches: {len(grass_matches)}")
print(f"Non-grass matches: {len(non_grass_matches)}")
print(f"Total matches for analysis: {len(grass_matches) + len(non_grass_matches)}")

# ==============================================
# FEATURE ENGINEERING
# ==============================================

class TennisFeatureEngine:
    """
    Feature engineering for tennis match prediction with temporal safeguards
    """
    def __init__(self):
        # General surface Elo ratings (all surfaces except grass)
        self.player_elos = defaultdict(lambda: 1500)
        self.player_serve_ratings = defaultdict(lambda: 1500)
        self.player_return_ratings = defaultdict(lambda: 1500)
        
        # Grass-specific ratings
        self.player_grass_elos = defaultdict(lambda: 1500)
        self.player_grass_serve_ratings = defaultdict(lambda: 1500)
        self.player_grass_return_ratings = defaultdict(lambda: 1500)
        
        # Rolling form tracking
        self.player_last_10 = defaultdict(lambda: deque(maxlen=10))
        self.player_grass_last_10 = defaultdict(lambda: deque(maxlen=10))
        
        # Feature storage
        self.train_features = {}
        self.test_features = {}
    
    def get_grass_winrate(self, player_id):
        """Get player's grass court win rate from last 10 grass matches"""
        grass_results = self.player_grass_last_10[player_id]
        return sum(grass_results) / len(grass_results) if grass_results else 0.5
    
    def get_recent_form(self, player_id):
        """Get win rate from last 10 matches overall"""
        recent_results = self.player_last_10[player_id]
        return sum(recent_results) / len(recent_results) if recent_results else 0.5
    
    def extract_features(self, row):
        """Extract features BEFORE updating any statistics"""
        winner_id, loser_id = row['winner_id'], row['loser_id']
        
        return {
            'elo_diff': self.player_elos[winner_id] - self.player_elos[loser_id],
            'elo_grass_diff': self.player_grass_elos[winner_id] - self.player_grass_elos[loser_id],
            'serve_rating_diff': self.player_serve_ratings[winner_id] - self.player_serve_ratings[loser_id],
            'serve_rating_grass_diff': self.player_grass_serve_ratings[winner_id] - self.player_grass_serve_ratings[loser_id],
            'return_rating_diff': self.player_return_ratings[winner_id] - self.player_return_ratings[loser_id],
            'return_rating_grass_diff': self.player_grass_return_ratings[winner_id] - self.player_grass_return_ratings[loser_id],
            'grass_winrate_last10_diff': self.get_grass_winrate(winner_id) - self.get_grass_winrate(loser_id),
            'rank_diff': row.get('loser_rank', 999) - row.get('winner_rank', 999),
            'seed_diff': (row.get('loser_seed', 0) - row.get('winner_seed', 0)) if pd.notna(row.get('winner_seed')) and pd.notna(row.get('loser_seed')) else 0,
            'age_diff': 0,  # Age data not consistently available
            'last_10_diff': self.get_recent_form(winner_id) - self.get_recent_form(loser_id)
        }
    
    def update_ratings(self, row):
        """Update all ratings AFTER feature extraction"""
        winner_id, loser_id = row['winner_id'], row['loser_id']
        is_grass = row['surface'] == 'Grass'
        
        # Update general Elo
        winner_elo, loser_elo = self.player_elos[winner_id], self.player_elos[loser_id]
        expected = 1 / (1 + 10**((loser_elo - winner_elo)/400))
        k = 32
        
        self.player_elos[winner_id] += k * (1 - expected)
        self.player_elos[loser_id] += k * (0 - (1 - expected))
        
        # Update grass Elo (if grass match)
        if is_grass:
            winner_grass_elo = self.player_grass_elos[winner_id]
            loser_grass_elo = self.player_grass_elos[loser_id]
            expected_grass = 1 / (1 + 10**((loser_grass_elo - winner_grass_elo)/400))
            k_grass = 40
            
            self.player_grass_elos[winner_id] += k_grass * (1 - expected_grass)
            self.player_grass_elos[loser_id] += k_grass * (0 - (1 - expected_grass))
        
        # Update form tracking
        self.player_last_10[winner_id].append(1)
        self.player_last_10[loser_id].append(0)
        
        if is_grass:
            self.player_grass_last_10[winner_id].append(1)
            self.player_grass_last_10[loser_id].append(0)
        
        # Update serve/return ratings if stats available
        self._update_serve_return_ratings(row, winner_id, loser_id, is_grass)
    
    def _update_serve_return_ratings(self, row, winner_id, loser_id, is_grass):
        """Update serve and return ratings based on match statistics"""
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
    
    def build_features(self, all_matches_df, split_date='2024-07-01'):
        """Build features with proper temporal split"""
        sorted_matches = all_matches_df.sort_values('tourney_date').copy()
        split_date = pd.to_datetime(split_date)
        
        print(f"Processing {len(sorted_matches)} matches chronologically...")
        print(f"Train: before {split_date}, Test: from {split_date} onwards")
        
        for idx, (_, row) in enumerate(sorted_matches.iterrows()):
            # Extract features BEFORE updating statistics
            features = self.extract_features(row)
            
            # Store features based on time split
            match_id = f"{row['tourney_id']}_{row['match_num']}"
            feature_data = {
                **features,
                'date': row['tourney_date'],
                'tournament': row['tourney_name'],
                'winner_id': row['winner_id'],
                'loser_id': row['loser_id'],
                'surface': row['surface']
            }
            
            if row['tourney_date'] < split_date:
                self.train_features[match_id] = feature_data
            else:
                self.test_features[match_id] = feature_data
            
            # Update statistics AFTER feature extraction
            self.update_ratings(row)
            
            if idx % 2000 == 0:
                print(f"Processed {idx} matches...")
        
        print(f"Complete! Train: {len(self.train_features)}, Test: {len(self.test_features)}")
    
    def save_features(self, train_file='tennis_train_features.pkl', test_file='tennis_test_features.pkl'):
        """Save features to pickle files"""
        with open(train_file, 'wb') as f:
            pickle.dump(self.train_features, f)
        with open(test_file, 'wb') as f:
            pickle.dump(self.test_features, f)
        print(f"Features saved: {train_file}, {test_file}")

print("TennisFeatureEngine class defined successfully!")

# Build features using the TennisFeatureEngine
feature_engine = TennisFeatureEngine()

# Combine all matches for chronological processing
all_matches = pd.concat([non_grass_matches, grass_matches], ignore_index=True)

# Build features with proper temporal split (before Wimbledon 2024)
feature_engine.build_features(all_matches, split_date='2024-07-01')

# Save features to files
feature_engine.save_features()

# Show sample features
if feature_engine.train_features:
    sample_train = list(feature_engine.train_features.values())[0]
    print(f"\nSample TRAIN features: {sample_train['tournament']} on {sample_train['date'].date()}")
    
if feature_engine.test_features:
    sample_test = list(feature_engine.test_features.values())[0]
    print(f"Sample TEST features: {sample_test['tournament']} on {sample_test['date'].date()}")

# ==============================================
# MODEL TRAINING AND EVALUATION
# ==============================================

# Load SEPARATE train and test features (no leakage!)
with open('tennis_train_features.pkl', 'rb') as f:
    train_features = pickle.load(f)
    
with open('tennis_test_features.pkl', 'rb') as f:
    test_features = pickle.load(f)

print(f"Loaded TRAIN features: {len(train_features)} matches")
print(f"Loaded TEST features: {len(test_features)} matches")

# Convert to DataFrames
train_rows = []
for match_id, match_data in train_features.items():
    train_rows.append({'match_id': match_id, **match_data})

test_rows = []
for match_id, match_data in test_features.items():
    test_rows.append({'match_id': match_id, **match_data})

train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)

print(f"Train DataFrame: {len(train_df)} matches")
print(f"Test DataFrame: {len(test_df)} matches")

# Define our exact feature columns - GRASS-FOCUSED for Wimbledon!
FEATURE_COLUMNS = [
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

print(f"\nUsing {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}")
print(f"Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")

def create_binary_dataset(features_df):
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
        for col in FEATURE_COLUMNS:
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
        for col in FEATURE_COLUMNS:
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

# Create binary datasets for BOTH train and test
print("Creating binary datasets with flipping...")
train_binary = create_binary_dataset(train_df)
test_binary = create_binary_dataset(test_df)

print(f"Original train matches: {len(train_df)}")
print(f"Binary train dataset: {len(train_binary)} (should be 2x original)")
print(f"Original test matches: {len(test_df)}")
print(f"Binary test dataset: {len(test_binary)} (should be 2x original)")

# Prepare features and targets
X_train = train_binary[FEATURE_COLUMNS]
y_train = train_binary['target']

X_test = test_binary[FEATURE_COLUMNS]
y_test = test_binary['target']

print(f"\nFeature shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"Train target balance: {y_train.value_counts(normalize=True)}")
print(f"Test target balance: {y_test.value_counts(normalize=True)}")

# Check for data leakage by ensuring no overlap in dates
print(f"\nDATA LEAKAGE CHECK:")
print(f"Train date range: {train_binary['date'].min()} to {train_binary['date'].max()}")
print(f"Test date range: {test_binary['date'].min()} to {test_binary['date'].max()}")
print(f"No overlap: {train_binary['date'].max() < test_binary['date'].min()}")

# Train RandomForest
print("Training RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Train XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Predictions
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_pred = rf_model.predict(X_test)

xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict(X_test)

# Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

print("\nRandomForest Results:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, rf_pred_proba):.4f}")
print(f"Log Loss: {log_loss(y_test, rf_pred_proba):.4f}")

print("\nXGBoost Results:")
print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, xgb_pred_proba):.4f}")
print(f"Log Loss: {log_loss(y_test, xgb_pred_proba):.4f}")

# Feature importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)

print("\nRandomForest Feature Importance:")
rf_importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(rf_importance)

print("\nXGBoost Feature Importance:")
xgb_importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(xgb_importance)

# Save models for future use
joblib.dump(rf_model, 'wimbledon_rf_model.pkl')
joblib.dump(xgb_model, 'wimbledon_xgb_model.pkl')
print("Models saved!")

# ==============================================
# PREDICTION FUNCTIONS
# ==============================================

def get_player_features(player_name):
    """Get all relevant features for a player from our tracking dictionaries"""
    
    # First find the player ID
    player_matches = matches[
        (matches['winner_name'] == player_name) | 
        (matches['loser_name'] == player_name)
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
        'elo': feature_engine.player_elos[player_id],
        'elo_grass': feature_engine.player_grass_elos[player_id],
        'serve_rating': feature_engine.player_serve_ratings[player_id],
        'serve_rating_grass': feature_engine.player_grass_serve_ratings[player_id],
        'return_rating': feature_engine.player_return_ratings[player_id],
        'return_rating_grass': feature_engine.player_grass_return_ratings[player_id],
        'grass_winrate_last10': sum(feature_engine.player_grass_last_10[player_id])/len(feature_engine.player_grass_last_10[player_id]) if feature_engine.player_grass_last_10[player_id] else 0.5,
        'rank': rank if not pd.isna(rank) else 999,
        'seed': seed if not pd.isna(seed) else 0,
        'last_10': sum(feature_engine.player_last_10[player_id])/len(feature_engine.player_last_10[player_id]) if feature_engine.player_last_10[player_id] else 0.5
    }
    
    return features, player_id

def predict_match_outcome(player1_name, player2_name, model='both'):
    """
    Predict match outcome between two players using both models
    Returns win probability for player1
    """
    print(f"\nPredicting match: {player1_name} vs {player2_name}")
    print("-" * 50)
    
    # Get features for both players
    try:
        player1_features, player1_id = get_player_features(player1_name)
        player2_features, player2_id = get_player_features(player2_name)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Calculate feature differences (player1 - player2)
    feature_diffs = {
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
    
    # Make predictions
    X_pred = pd.DataFrame([feature_diffs])
    
    if model in ['xgb', 'both']:
        xgb_prob = xgb_model.predict_proba(X_pred)[0, 1]
        print(f"\nXGBoost prediction:")
        print(f"{player1_name} win probability: {xgb_prob:.1%}")
        print(f"{player2_name} win probability: {(1-xgb_prob):.1%}")
    
    if model in ['rf', 'both']:
        rf_prob = rf_model.predict_proba(X_pred)[0, 1]
        print(f"\nRandomForest prediction:")
        print(f"{player1_name} win probability: {rf_prob:.1%}")
        print(f"{player2_name} win probability: {(1-rf_prob):.1%}")
    
    # Print key feature differences
    print("\nKey feature differences (Player 1 - Player 2):")
    print(f"Elo rating diff: {feature_diffs['elo_diff']:.1f}")
    print(f"Grass Elo diff: {feature_diffs['elo_grass_diff']:.1f}")
    print(f"Serve rating diff: {feature_diffs['serve_rating_diff']:.1f}")
    print(f"Return rating diff: {feature_diffs['return_rating_diff']:.1f}")
    print(f"Recent form diff: {feature_diffs['last_10_diff']:.2f}")
    print(f"Ranking diff: {-feature_diffs['rank_diff']:.0f}")  # Reverse back for display

# Example usage
print("Predicting example match...")
predict_match_outcome("Ben Shelton", "Jiri Lehecka")
predict_match_outcome("Jiri Lehecka", "Ben Shelton")

print("\n" + "="*50)
print("READY FOR 2025 PREDICTIONS!")
print("="*50)
print("✅ Features built with no data leakage")
print("✅ Models trained on pre-Wimbledon 2024 data") 
print("✅ Tested on Wimbledon 2024")
print("✅ Binary flipping implemented")
print("✅ Models saved for future use")
print("✅ Can now predict any Player A vs Player B match")


import random, itertools, numpy as np, pandas as pd

def orientation_gap(p1, p2):
    """Return |P(A beats B) – (1-P(B beats A))| for both models."""
    x = predict_match_outcome(p1, p2, model='xgb', silent=True)
    y = predict_match_outcome(p2, p1, model='xgb', silent=True)
    return abs(x - (1 - y))

players = ["Carlos Alcaraz","Jannik Sinner","Daniil Medvedev",
           "Alexander Zverev","Hubert Hurkacz","Ben Shelton",
           "Matteo Berrettini","Tommy Paul","Alex De Minaur","Grigor Dimitrov"]

pairs = random.sample(list(itertools.combinations(players, 2)), 40)
gaps  = [orientation_gap(a,b) for a,b in pairs]
print(f"Mean gap  : {np.mean(gaps):.4f}")
print(f"Max  gap  : {np.max(gaps):.4f}")
