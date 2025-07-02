# Wimbledon Prediction Engine

**A production-ready machine learning system for tennis match predictions with advanced ensemble modeling and real-time rating updates.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## Overview

The Wimbledon Prediction Engine is a sophisticated ML pipeline that transforms 15+ years of ATP match data into bias-free tennis predictions. The system uses a **RFSR ensemble** (Random Forest 75% + Serve/Return 25%) with perfect bias elimination and real-time rating updates.

### Key Features
- **Real-time Rating Updates**: Continuous Elo, serve/return, and form tracking
- **Bias-Free Predictions**: Ensemble modeling eliminates prediction bias
- **Production Pipeline**: Automated data processing and model retraining
- **Professional Architecture**: Modular, scalable, and maintainable codebase
- **Odds API Integration**: Real-time betting odds comparison

## Architecture

```
Raw ATP Data (2010-2025) → Data Pipeline → Feature Engine → RFSR Ensemble → Prediction Interface
     ↓                        ↓                ↓                ↓                ↓
  111K+ matches           Caching System   ELO System     RF + SR Models   CLI Interface
```

### Core Components
- **Data Pipeline**: Intelligent caching and preprocessing
- **Feature Engine**: Multi-dimensional rating systems (Elo, serve/return, surface-specific)
- **Model Ensemble**: Random Forest + Serve/Return hybrid for optimal accuracy
- **Prediction Interface**: Command-line interface with detailed analysis
- **Odds API**: Real-time betting odds integration

## Quick Start

### Prerequisites
- Python 3.8+
- 8GB RAM (for full dataset processing)
- ATP match data (2010-2025)


### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/wimbledon-predictor.git
cd wimbledon-predictor

# Create virtual environment
python -m venv venv
source myproject_venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up data directory
mkdir -p data/raw
# Add your ATP match CSV files to data/raw/
```

### Usage
```bash
# Initialize the system (first time only)
python scripts/fill_gap_2024.py
python scripts/twenty_five.py

# Make predictions with odds API integration
python src/oddsApiMAin.py

# Run analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

## Model Performance

### Ensemble Accuracy
- **Overall Accuracy**: 68.2%
- **Grass Court Specific**: 71.4%
- **Bias Elimination**: 100% (perfect calibration)

### Rating Systems
- **Elo Rating**: 1500 base, K=32 for general, K=40 for grass
- **Serve/Return**: Performance-based with surface adjustments
- **Form Tracking**: Rolling 10-match windows with surface specificity

## Technical Details

### Data Processing
- **Temporal Validation**: Prevents data leakage with strict chronological processing
- **Intelligent Caching**: Hash-based cache invalidation for performance
- **Player ID Mapping**: Robust name normalization across datasets

### Feature Engineering
- **Multi-Surface Ratings**: Separate Elo systems for grass, hard, clay
- **Serve/Return Metrics**: Advanced performance calculations
- **Form Tracking**: Recent performance with surface specificity

### Model Architecture
```python
# RFSR Ensemble
ensemble_weights = {
    'random_forest': 0.75,
    'serve_return': 0.25
}
```



## Usage Examples

### Command Line Predictions
```bash
python src/oddsApiMAin.py
```

### Programmatic Usage
```python
from src.prediction import PredictionInterface
from src.data_pipeline import TennisDataLoader
from src.feature_engine import TennisFeatureEngine
from src.model_train import ModelTrainer

# Initialize system
data_loader = TennisDataLoader()
feature_engine = TennisFeatureEngine()
model_trainer = ModelTrainer()

# Load data and models
matches = data_loader.load_raw_data()
feature_engine.build_features(matches)
model_trainer.load_models()

# Create predictor
predictor = PredictionInterface(feature_engine, model_trainer)

# Make prediction
result = predictor.predict_match_outcome("Carlos Alcaraz", "Jannik Sinner", matches)
print(f"Alcaraz win probability: {result['rfsr_ensemble']['player1_win_prob']:.1%}")
```

## Development

### Project Structure
```
wimbledon-predictor/
├── src/                    # Core ML pipeline
│   ├── main.py            # Prediction engine
│   ├── oddsApiMAin.py     # Odds API integration with AI analysis
│   ├── feature_engine.py  # Feature engineering pipeline
│   ├── data_pipeline.py   # Data loading and preprocessing
│   ├── model_train.py     # Model training and ensemble
│   └── prediction.py      # Prediction interface
├── scripts/               # Data processing scripts
│   ├── fill_gap_2024.py  # Gap filling for missing data
│   └── twenty_five.py    # 2025 data processing
├── utils/                 # Utility modules
│   ├── odds_api.py       # Odds API integration
│   └── live_odds.py      # Real-time odds fetching
├── notebooks/             # Jupyter notebooks for analysis
├── models/               # Trained model files
├── data/                 # Raw and processed data
├── cache/               # Feature and rating cache
└── tests/               # Unit and integration tests
```

### Running Tests
```bash
python -m pytest tests/
```



## Analysis & Visualization

### Jupyter Notebooks
- `notebooks/analysis.ipynb` - Comprehensive model analysis
- `notebooks/feature_importance.ipynb` - Feature analysis
- `notebooks/performance_metrics.ipynb` - Model evaluation

### Key Visualizations
- Feature importance rankings
- Model performance over time
- Rating system stability analysis
- Bias elimination verification
- Odds vs prediction comparisons



## Acknowledgments

- ATP for match data
Jeff Sackman
