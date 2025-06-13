# 🎾 Wimbledon Tennis Prediction Engine

A professional-grade machine learning system for predicting tennis match outcomes, specifically optimized for Wimbledon grass court matches. This system uses advanced feature engineering, multiple ML models, and sophisticated bias elimination techniques.

## 🏆 Key Features

- **Bias-Free Predictions**: Implements averaging trick to eliminate player name ordering bias
- **Advanced Feature Engineering**: ELO ratings, surface-specific metrics, serve/return analysis
- **Multiple Models**: XGBoost and Random Forest ensemble
- **Professional Architecture**: Modular, cached, and production-ready
- **Betting Odds**: Converts predictions to various betting formats with bookmaker vig
- **Temporal Validation**: Proper train/test split respecting match chronology

## 📊 System Performance

### Model Metrics (Bias-Corrected)
- **XGBoost**: 65.2% accuracy, 0.712 AUC-ROC, 0.623 log loss
- **Random Forest**: 64.8% accuracy, 0.708 AUC-ROC, 0.631 log loss
- **Bias Test**: 0.000000 naming bias (perfect elimination)

### Key Features by Importance
1. **ELO Rating Difference**: General player strength
2. **Grass ELO Difference**: Surface-specific performance
3. **Serve Rating Difference**: Serving ability comparison
4. **Return Rating Difference**: Return game analysis
5. **Recent Form**: Last 10 matches performance
6. **ATP Ranking**: Official world rankings

## 🏗️ Architecture

```
tennis-prediction-engine/
├── src/
│   ├── data_pipeline.py      # Data loading & preprocessing
│   ├── feature_engine.py     # Feature engineering pipeline
│   ├── model_train.py        # Model training & evaluation
│   ├── prediction.py         # Prediction interface (bias-free)
│   ├── main.py              # Main execution script
│   └── train_models.py      # Model retraining script
├── cache/                   # Cached features & data
├── models/                  # Trained model files
├── atp_matches/            # Raw ATP match data
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd Wimbledon-Predictor

# Create virtual environment
python -m venv myproject_venv
source myproject_venv/bin/activate  # On Windows: myproject_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Predictions
```bash
# Run the main prediction system
python src/main.py
```

### 3. Make Custom Predictions
```python
from src.main import initialize_engine, predict_match

# Initialize system
predictor, matches = initialize_engine()

# Predict any match
predict_match(predictor, matches, "Novak Djokovic", "Carlos Alcaraz")
```

## 🔧 Core Components

### 1. Data Pipeline (`data_pipeline.py`)
- **Purpose**: Load and preprocess ATP match data
- **Features**: 
  - Intelligent caching system
  - Data validation and cleaning
  - Surface-specific filtering
- **Key Methods**:
  - `load_raw_data()`: Load ATP CSV files
  - `preprocess_data()`: Clean and split by surface

### 2. Feature Engine (`feature_engine.py`)
- **Purpose**: Advanced feature engineering for tennis predictions
- **Features**:
  - ELO rating systems (general + grass-specific)
  - Serve/return performance tracking
  - Recent form analysis (rolling windows)
  - Temporal feature validation
- **Key Metrics**:
  - Player ELO ratings (updated after each match)
  - Surface-specific performance ratings
  - Head-to-head statistics
  - Recent form (last 10 matches)

### 3. Model Training (`model_train.py`)
- **Purpose**: Train and evaluate ML models
- **Models**: 
  - XGBoost Classifier
  - Random Forest Classifier
- **Features**:
  - Binary classification with data augmentation
  - Feature importance analysis
  - Cross-validation metrics
  - Model persistence

### 4. Prediction Interface (`prediction.py`)
- **Purpose**: Bias-free match outcome predictions
- **Key Innovation**: **Averaging Trick** for bias elimination
  ```python
  # Calculate both orientations
  prob_1v2 = model.predict_proba(player1_vs_player2)
  prob_2v1 = model.predict_proba(player2_vs_player1)
  
  # Average for unbiased result
  final_prob = (prob_1v2 + (1 - prob_2v1)) / 2
  ```
- **Output Formats**:
  - Win probabilities
  - American odds (+150, -200)
  - Decimal odds (2.50, 1.50)
  - Fractional odds (3/2, 1/2)

## 🧪 Bias Elimination

### The Problem
Traditional tennis prediction models suffer from **naming bias** - predictions change based on player name input order:
- `predict("Player A", "Player B")` ≠ `predict("Player B", "Player A")`
- This occurs because features are calculated as `player1 - player2`

### Our Solution: Averaging Trick
1. **Calculate both orientations**:
   - P(A beats B) when A is "player1"
   - P(B beats A) when B is "player1"
2. **Average the results**:
   - Final P(A beats B) = [P₁ + (1 - P₂)] / 2
3. **Result**: Perfect symmetry (0.000000 bias)

### Verification
```bash
# Run bias test
python test_bias_check.py
```

## 📈 Feature Engineering Details

### ELO Rating System
- **Base ELO**: 1500 (starting rating)
- **K-Factor**: 32 (general), 40 (grass-specific)
- **Update Formula**: `new_rating = old_rating + K × (actual - expected)`

### Serve/Return Ratings
- **Serve Performance**: Based on aces, first serve %, points won
- **Return Performance**: Break points converted, return games won
- **Surface Adjustment**: Grass-specific bonuses for serve-and-volley

### Recent Form Tracking
- **Rolling Windows**: Last 10 matches (all surfaces + grass-only)
- **Form Score**: Win percentage in recent matches
- **Temporal Decay**: More recent matches weighted higher

## 🎯 Usage Examples

### Basic Prediction
```python
# Simple match prediction
predict_match(predictor, matches, "Rafael Nadal", "Roger Federer")
```

### Advanced Usage
```python
# Get detailed prediction data
prediction = predictor.predict_match_outcome(
    "Novak Djokovic", 
    "Andy Murray", 
    matches_df, 
    model='both'  # Use both XGBoost and Random Forest
)

# Access probabilities
xgb_prob = prediction['xgboost']['player1_win_prob']
rf_prob = prediction['random_forest']['player1_win_prob']

# Get betting odds
odds = prediction['xgboost']['betting_odds']
american_odds = odds['player1']['american']
```

### Model Retraining
```bash
# Retrain models with latest data
python src/train_models.py
```

## 🔍 Model Validation

### Temporal Split
- **Training**: Matches before July 1, 2024
- **Testing**: Matches after July 1, 2024
- **Rationale**: Prevents data leakage, simulates real-world usage

### Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **AUC-ROC**: Area under ROC curve (discrimination ability)
- **Log Loss**: Probability calibration quality
- **Bias Test**: Naming order independence

### Feature Importance
```
1. elo_diff              (0.234) - General player strength
2. elo_grass_diff        (0.187) - Grass court performance  
3. serve_rating_diff     (0.156) - Serving ability
4. return_rating_diff    (0.142) - Return game quality
5. last_10_diff          (0.098) - Recent form
6. rank_diff             (0.089) - ATP ranking
7. grass_winrate_last10  (0.067) - Grass-specific form
8. serve_rating_grass    (0.027) - Grass serving
```

## 🛠️ Technical Implementation

### Caching System
- **Feature Cache**: Avoid recomputing expensive features
- **Data Cache**: Cache preprocessed match data
- **Hash Validation**: Detect data changes automatically

### Error Handling
- **Player Validation**: Check if players exist in dataset
- **Data Quality**: Handle missing values and outliers
- **Model Loading**: Graceful fallback for missing models

### Logging
- **Structured Logging**: Detailed operation tracking
- **Performance Monitoring**: Cache hits, computation times
- **Error Tracking**: Comprehensive error reporting

## 🔮 Future Enhancements

### Planned Models for Ensemble
1. **LSTM Networks**: Capture temporal patterns in player performance
2. **Bayesian Models**: Provide uncertainty estimates
3. **Graph Neural Networks**: Model player interaction networks
4. **Transformer Models**: Attention-based sequence modeling

### Advanced Features
- **Weather Integration**: Wind, temperature, humidity effects
- **Injury Tracking**: Player fitness and injury history
- **Playing Style**: Aggressive vs defensive classification
- **Head-to-Head**: Historical matchup analysis

## 📋 Requirements

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.6.0
joblib>=1.1.0
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ATP for providing comprehensive match data
- Tennis analytics community for insights
- Open source ML libraries (scikit-learn, XGBoost)

---

**Built with ❤️ for tennis analytics and machine learning** 